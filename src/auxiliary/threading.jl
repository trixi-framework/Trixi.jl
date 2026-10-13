# Shared-memory parallelism for CPU loops: implementation of `@threaded`.
#
# `@threaded` splits the iteration range into `nchunks` contiguous chunks (at most one
# per thread of the default thread pool) and runs one chunk per thread. The way the
# chunks are run is selected by the `backend` preference (see `set_threading_backend!`):
#
# - `:pool` (default): a small pool of persistent worker tasks, one sticky task per
#   thread of the default thread pool, which spin for a short while after finishing
#   a chunk before going to sleep. Waking up spinning workers is much cheaper than
#   scheduling new tasks, which matters for the many small loops in `rhs!`.
#   This replaces the role Polyester.jl's `@batch` used to play.
# - `:static`: `Threads.@threads :static` over the chunks.
# - `:dynamic`: `Threads.@threads :dynamic` over the chunks (tasks may run on any
#   thread of the default pool, but not migrate while they do not yield).
# - `:kernelabstractions`: like `:static` for loops that are not written as
#   KernelAbstractions.jl kernels.
# - `:serial`: no threading at all.
#
# The chunk-to-thread mapping is fixed for `:pool` and `:static`, so scratch buffers
# indexed by `Threads.threadid()` are safe. They are also safe for `:dynamic` as long
# as loop bodies do not yield, which is the case for all numerical kernels in Trixi.jl.

# `_THREADING_BACKEND` is defined in `src/Trixi.jl`.

# Number of chunks a threaded loop over `range` is split into.
# Returns `1` if the loop should be run serially, e.g., because only a single thread
# is available, or when called from inside another threaded loop (nested threading).
@inline function threaded_nchunks(range)
    @static if _THREADING_BACKEND === :serial
        return 1
    else
        n = Threads.threadpoolsize(:default)
        n == 1 && return 1
        threaded_nested() && return 1
        return max(1, min(n, length(range)))
    end
end

@inline function threaded_nested()
    @static if _THREADING_BACKEND === :pool
        p = POOL[]
        return p !== nothing && (p.busy[] != 0 || p.owner != Threads.threadid())
    else
        return ccall(:jl_in_threaded_region, Cint, ()) != 0
    end
end

# The `chunk`-th of `nchunks` contiguous parts of `range`
@inline function chunk_range(range::AbstractUnitRange, chunk, nchunks)
    len = length(range)
    lo = first(range) + div((chunk - 1) * len, nchunks)
    hi = first(range) + div(chunk * len, nchunks) - 1
    return lo:hi
end
@inline function chunk_range(range, chunk, nchunks)
    len = length(range)
    lo = div((chunk - 1) * len, nchunks) + 1
    hi = div(chunk * len, nchunks)
    return view(range, lo:hi)
end

# Run `f(chunk)` for `chunk in 1:nchunks`, in parallel
@inline function threaded_run(f::F, nchunks) where {F}
    if nchunks == 1
        f(1)
        return nothing
    end
    @static if _THREADING_BACKEND === :pool
        pool_run(f, nchunks)
    elseif _THREADING_BACKEND === :dynamic
        Threads.@threads :dynamic for chunk in 1:nchunks
            f(chunk)
        end
    else
        Threads.@threads :static for chunk in 1:nchunks
            f(chunk)
        end
    end
    return nothing
end

# Run `f(chunk)` for `chunk in 1:nchunks` in parallel and return the results of the
# chunks combined with the reduction operators `ops` (one per entry of the tuple `init`).
# `f` must return a tuple of the same type as `init`.
@inline function threaded_mapreduce_chunks(f::F, ops::Ops, init, nchunks) where {F, Ops}
    if nchunks == 1
        return combine_reductions(ops, init, f(1))
    end
    partials = Vector{typeof(init)}(undef, nchunks)
    threaded_run(nchunks) do chunk
        @inbounds partials[chunk] = f(chunk)
        return nothing
    end
    result = init
    for chunk in 1:nchunks
        result = combine_reductions(ops, result, @inbounds partials[chunk])
    end
    return result
end

# Neutral elements of the reduction operators supported by `@threaded reduction = ...`
@inline reduction_init(::typeof(+), x) = zero(x)
@inline reduction_init(::typeof(*), x) = one(x)
@inline reduction_init(::typeof(max), x) = typemin(x)
@inline reduction_init(::typeof(min), x) = typemax(x)
# `Trixi.max` and `Trixi.min` are different from `Base.max` and `Base.min`
@inline reduction_init(::typeof(Base.max), x) = typemin(x)
@inline reduction_init(::typeof(Base.min), x) = typemax(x)
@inline reduction_init(::typeof(&), x) = true
@inline reduction_init(::typeof(|), x) = false

@inline function combine_reductions(ops::Tuple, partials1::Tuple, partials2::Tuple)
    return map((op, a, b) -> op(a, b), ops, partials1, partials2)
end

"""
    @threaded for ... end
    @threaded reduction=(op, var) for ... end
    @threaded reduction=((op1, var1), (op2, var2), ...) for ... end

Semantically the same as `Threads.@threads` when iterating over a `AbstractUnitRange`
but without guarantee that the underlying implementation uses `Threads.@threads`
or works for more general `for` loops.
In particular, there may be an additional check whether only one thread is used
to reduce the overhead of serial execution. Nested `@threaded` loops run the inner
loop serially.

The optional `reduction` argument (same syntax as in Polyester.jl's `@batch`) reduces
the local variables `var1, var2, ...` with the operators `op1, op2, ...`, which must be
one of `+`, `*`, `max`, `min`, `&`, `|`. Each chunk starts from the neutral element of
the operator and the chunk results are combined with the initial value of the variable
in a fixed order.

!!! warn
    This macro does not necessarily work for general `for` loops. For example,
    it does not necessarily support general iterables such as `eachline(filename)`.
    `break` and `return` inside the loop body do not work as in a serial loop.

Some discussion can be found at [https://discourse.julialang.org/t/overhead-of-threads-threads/53964](https://discourse.julialang.org/t/overhead-of-threads-threads/53964)
and [https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435](https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435).
"""
macro threaded(args...)
    expr = args[end]
    reductions = Tuple{Any, Symbol}[]
    for arg in args[1:(end - 1)]
        if Meta.isexpr(arg, :(=)) && arg.args[1] === :reduction
            spec = arg.args[2]
            @assert Meta.isexpr(spec, :tuple)
            if all(x -> Meta.isexpr(x, :tuple), spec.args)
                for s in spec.args
                    push!(reductions, (s.args[1], s.args[2]))
                end
            else
                push!(reductions, (spec.args[1], spec.args[2]))
            end
        else
            throw(ArgumentError("unsupported argument $(arg) to @threaded"))
        end
    end

    if !Meta.isexpr(expr, :for)
        throw(ArgumentError("@threaded requires a `for` loop expression"))
    end
    loop_spec = expr.args[1]
    if !Meta.isexpr(loop_spec, :(=))
        # e.g., `for i in 1:n, j in 1:m`
        throw(ArgumentError("@threaded only supports a single loop variable"))
    end

    @static if _THREADING_BACKEND === :serial
        return esc(expr)
    end

    loopvar = loop_spec.args[1]
    range_expr = loop_spec.args[2]
    body = expr.args[2]

    range = gensym(:range)
    nchunks = gensym(:nchunks)
    chunk = gensym(:chunk)

    # We keep a plain serial loop for the case of a single chunk (one thread or nested
    # threading). Running the loop body in a closure is slower for some kernels (up to
    # 1.5x), e.g., since facts established by the caller such as the array sizes checked
    # by `check_axes` do not propagate into the closure.
    serial_loop = Expr(:for, Expr(:(=), loopvar, range), body)

    if isempty(reductions)
        ex = quote
            let $range = $range_expr
                $nchunks = $Trixi.threaded_nchunks($range)
                if $nchunks == 1
                    $serial_loop
                else
                    $Trixi.threaded_run($nchunks) do $chunk
                        for $loopvar in $Trixi.chunk_range($range, $chunk, $nchunks)
                            $body
                        end
                        nothing
                    end
                end
            end
        end
    else
        ops = [r[1] for r in reductions]
        vars = [r[2] for r in reductions]
        inits = [gensym(Symbol(v, :_init)) for v in vars]
        op_tuple = Expr(:tuple, ops...)
        var_tuple = Expr(:tuple, vars...)
        init_tuple = Expr(:tuple, inits...)
        local_inits = [:(local $v = $Trixi.reduction_init($op, $i))
                       for (op, v, i) in zip(ops, vars, inits)]
        let_bindings = [:($range = $range_expr)]
        append!(let_bindings, [:($i = $v) for (i, v) in zip(inits, vars)])
        ex = quote
            let $(let_bindings...)
                $nchunks = $Trixi.threaded_nchunks($range)
                if $nchunks == 1
                    $serial_loop
                else
                    $var_tuple = $Trixi.threaded_mapreduce_chunks($op_tuple, $init_tuple,
                                                                  $nchunks) do $chunk
                        $(local_inits...)
                        for $loopvar in $Trixi.chunk_range($range, $chunk, $nchunks)
                            $body
                        end
                        $var_tuple
                    end
                end
            end
        end
    end
    # Use `esc(quote ... end)` for nested macro calls as suggested in
    # https://github.com/JuliaLang/julia/issues/23221
    return esc(ex)
end

# A minimal pool of persistent worker tasks used by the `:pool` backend.
#
# There is one sticky worker task per thread of the default thread pool except one;
# the calling task runs the first chunk itself. After finishing a chunk, workers spin
# (yielding to other tasks regularly) for a short while before going to sleep on an
# `Event`, so that back-to-back threaded loops do not pay for task creation and
# scheduling. The pool is created lazily by the first threaded loop on the thread
# `owner`; threaded loops started from other threads or while the pool is busy
# (nested threading) run serially.
mutable struct ThreadingPool
    @atomic work::Any
    const nworkers::Int
    const generation::Threads.Atomic{Int}
    const done::Threads.Atomic{Int}
    const sleeping::Vector{Threads.Atomic{Int}}
    const events::Vector{Base.Event}
    const owner::Int
    const busy::Threads.Atomic{Int}
    @atomic err::Any
end

const POOL = Ref{Union{Nothing, ThreadingPool}}(nothing)
# Number of busy-waiting iterations of a worker before it goes to sleep
const POOL_SPIN = 1 << 14

function pool_worker_loop(pool::ThreadingPool, worker::Int)
    last_generation = 0
    while true
        spins = 0
        while pool.generation[] == last_generation
            spins += 1
            if spins > POOL_SPIN
                pool.sleeping[worker][] = 1
                if pool.generation[] == last_generation
                    wait(pool.events[worker])
                end
                pool.sleeping[worker][] = 0
                spins = 0
            elseif spins % 1024 == 0
                yield()
            else
                ccall(:jl_cpu_pause, Cvoid, ())
                GC.safepoint()
            end
        end
        last_generation = pool.generation[]
        try
            Base.invokelatest(@atomic(pool.work), worker + 1)
        catch err
            @atomic pool.err = CapturedException(err, catch_backtrace())
        end
        Threads.atomic_add!(pool.done, 1)
    end
end

function create_pool()
    owner = Threads.threadid()
    nworkers = Threads.threadpoolsize(:default) - 1
    tids = [tid for tid in Threads.threadpooltids(:default) if tid != owner]
    resize!(tids, nworkers)
    pool = ThreadingPool(nothing, nworkers, Threads.Atomic{Int}(0),
                         Threads.Atomic{Int}(0),
                         [Threads.Atomic{Int}(0) for _ in 1:nworkers],
                         [Base.Event(true) for _ in 1:nworkers], owner,
                         Threads.Atomic{Int}(0), nothing)
    for (worker, tid) in enumerate(tids)
        task = Task(() -> pool_worker_loop(pool, worker))
        task.sticky = true
        ccall(:jl_set_task_tid, Cint, (Any, Cint), task, tid - 1)
        schedule(task)
    end
    return pool
end

@noinline function get_pool()
    pool = POOL[]
    if pool === nothing
        pool = create_pool()
        POOL[] = pool
    end
    return pool
end

function pool_run(f::F, nchunks) where {F}
    # Never start worker tasks while precompiling
    if ccall(:jl_generating_output, Cint, ()) == 1
        for chunk in 1:nchunks
            f(chunk)
        end
        return nothing
    end
    pool = get_pool()
    if nchunks > pool.nworkers + 1 || pool.owner != Threads.threadid() ||
       Threads.atomic_cas!(pool.busy, 0, 1) != 0
        for chunk in 1:nchunks
            f(chunk)
        end
        return nothing
    end
    try
        # Chunks `2:nchunks` are run by workers `1:(nchunks - 1)`, the remaining
        # workers run a no-op.
        @atomic pool.work = chunk -> (chunk <= nchunks && f(chunk); nothing)
        @atomic pool.err = nothing
        pool.done[] = 0
        Threads.atomic_add!(pool.generation, 1)
        for worker in 1:(pool.nworkers)
            if pool.sleeping[worker][] == 1
                notify(pool.events[worker])
            end
        end
        f(1)
        while pool.done[] < pool.nworkers
            ccall(:jl_cpu_pause, Cvoid, ())
            GC.safepoint()
        end
        @atomic pool.work = nothing
        err = @atomic pool.err
        if err !== nothing
            throw(err)
        end
    finally
        pool.busy[] = 0
    end
    return nothing
end
