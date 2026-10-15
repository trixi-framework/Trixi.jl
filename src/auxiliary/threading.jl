# Shared-memory parallelism for CPU loops: implementation of `@threaded`.
#
# `@threaded` splits the iteration range into `nchunks` contiguous chunks (at most one
# per thread of the default thread pool) and runs one chunk per thread. The way the
# chunks are run is selected by the `backend` preference (see `set_threading_backend!`):
#
# - `:pool` (default): OhMyThreads.jl's `tforeach` and `tmapreduce` with the
#   `TeamScheduler` of ThreadTeams.jl, which runs the chunks on a persistent team of
#   workers: one sticky worker task per thread of the default thread pool, which spins
#   for a short while after finishing a chunk before going to sleep. Handing chunks to
#   spinning workers is much cheaper than scheduling new tasks, which matters for the
#   many small loops in `rhs!`. The calling task runs the first chunk, and also every
#   chunk that a worker has not picked up yet (e.g., because another process occupies
#   its core). This replaces the role Polyester.jl's `@batch` used to play.
# - `:static`: `Threads.@threads :static` over the chunks.
# - `:dynamic`: `Threads.@threads :dynamic` over the chunks (tasks may run on any
#   thread of the default pool, but not migrate while they do not yield).
# - `:kernelabstractions`: like `:static` for loops that are not written as
#   KernelAbstractions.jl kernels.
# - `:serial`: no threading at all.
#
# Chunks that run at the same time always run on different threads, so scratch buffers
# indexed by `Threads.threadid()` are safe as long as loop bodies do not yield, which is
# the case for all numerical kernels in Trixi.jl.
#
# TODO: index scratch buffers by the chunk index instead of `Threads.threadid()`.
# Within one `@threaded` loop, every chunk has a unique index in `1:nchunks` (with
# `nchunks <= Threads.nthreads(:default)`), so this stays correct even if a loop body
# yields, works the same for all backends, and costs nothing at runtime (unlike
# TaskLocalValues.jl, whose lookups are not free and whose values would be re-created
# for every loop with `Threads.@threads`). `@threaded` could expose the chunk index, e.g.,
# `@threaded chunk = c for element in ...`, using `enumerate(chunks(...))` and
# `@task_index` with the `TeamScheduler`. The buffers could then have
# `Threads.nthreads(:default)` instead of `Threads.maxthreadid()` entries, and the
# `:static` and `:dynamic` backends could use OhMyThreads.jl's schedulers (see
# `threaded_foreach`).

# `_THREADING_BACKEND` is defined in `src/Trixi.jl`.

# Number of chunks a threaded loop over `range` is split into.
# Returns `1` if the loop should be run serially, e.g., because only a single thread
# is available, or when called from inside another threaded loop (nested threading).
# This is inlined in front of the serial loop of `@threaded`, so it must not contain
# acquire (or stronger) atomic loads: LLVM's MemoryDependenceAnalysis treats them as
# clobbers ("FIXME: This is overly conservative"), so that GVN cannot merge loads before
# and after them, e.g., array sizes checked by `check_axes`. This cost 4-8% in several
# kernels on Julia v1.13. `ThreadTeams.nchunks` only uses relaxed loads.
@inline function threaded_nchunks(range)
    @static if _THREADING_BACKEND === :serial
        return 1
    elseif _THREADING_BACKEND === :pool
        # Also `1` while precompiling and while another task runs a threaded loop
        return ThreadTeams.nchunks(range)
    else
        n = Threads.threadpoolsize(:default)
        n == 1 && return 1
        ccall(:jl_in_threaded_region, Cint, ()) != 0 && return 1
        return max(1, min(n, length(range)))
    end
end

# The `nchunks` contiguous parts of `range` (a `chunks` object of ChunkSplitters.jl,
# re-exported by OhMyThreads.jl)
@inline threaded_parts(range, nchunks) = OhMyThreads.chunks(range; n = nchunks)

# Run `f(part)` for each of the `nchunks` contiguous parts of `range`, in parallel.
# The loop body of `@threaded` stays a plain `for` loop over a part.
@inline function threaded_foreach(f::F, range, nchunks) where {F}
    parts = threaded_parts(range, nchunks)
    @static if _THREADING_BACKEND === :pool
        # One part per worker of the team
        OhMyThreads.tforeach(f, parts; scheduler = TeamScheduler())
        # The other backends deliberately use `Threads.@threads` instead of OhMyThreads.jl's
        # `StaticScheduler`/`DynamicScheduler`: those would spawn tasks for nested loops
        # instead of running them serially (`threaded_nchunks` relies on
        # `jl_in_threaded_region`, which only `Threads.@threads` sets). Waiting for nested
        # tasks yields, so another chunk could run on the same thread and overwrite the
        # scratch buffers indexed by `Threads.threadid()`.
    elseif _THREADING_BACKEND === :dynamic
        Threads.@threads :dynamic for chunk in 1:nchunks
            f(@inbounds parts[chunk])
        end
    else
        Threads.@threads :static for chunk in 1:nchunks
            f(@inbounds parts[chunk])
        end
    end
    return nothing
end

# Run `f(part)` for each of the `nchunks` contiguous parts of `range` in parallel and
# return the results of the parts combined with the reduction operators `ops` (one per
# entry of the tuple `init`), starting from `init`. `f` must return a tuple with one entry
# per entry of `init`. The results of the parts are combined in a fixed order.
@inline function threaded_mapreduce(f::F, ops::Ops, init, range, nchunks) where {F, Ops}
    parts = threaded_parts(range, nchunks)
    @static if _THREADING_BACKEND === :pool
        # Combines the results of the parts in order, without allocating
        result = OhMyThreads.tmapreduce(f, CombineReductions(ops), parts;
                                        scheduler = TeamScheduler())
        return combine_reductions(ops, init, result)
    else
        partials = Vector{typeof(init)}(undef, nchunks)
        # `1:nchunks` split into `nchunks` parts: one chunk index per part
        threaded_foreach(1:nchunks, nchunks) do chunks
            for chunk in chunks
                @inbounds partials[chunk] = f(parts[chunk])
            end
            return nothing
        end
        result = init
        for chunk in 1:nchunks
            result = combine_reductions(ops, result, @inbounds partials[chunk])
        end
        return result
    end
end

# Neutral elements of the reduction operators supported by `@threaded reduction = ...`
@inline reduction_init(::typeof(+), x) = zero(x)
@inline reduction_init(::typeof(*), x) = one(x)
@inline reduction_init(::typeof(max), x) = typemin(x)
@inline reduction_init(::typeof(min), x) = typemax(x)
# `Trixi.max` and `Trixi.min` are different from `Base.max` and `Base.min`
@inline reduction_init(::typeof(Base.max), x) = typemin(x)
@inline reduction_init(::typeof(Base.min), x) = typemax(x)
@inline reduction_init(::typeof(&), x::Bool) = true
@inline reduction_init(::typeof(&), x::Integer) = ~zero(x)
@inline reduction_init(::typeof(|), x) = zero(x)

@inline function combine_reductions(ops::Tuple, partials1::Tuple, partials2::Tuple)
    return map((op, a, b) -> op(a, b), ops, partials1, partials2)
end

# `combine_reductions` as a reduction operator for `OhMyThreads.tmapreduce`
struct CombineReductions{Ops} <: Function
    ops::Ops
end
@inline (c::CombineReductions)(partials1, partials2) = combine_reductions(c.ops, partials1,
                                                                          partials2)

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
    # !!! danger "Heisenbug"
    #     `wrap_array` used to return `PtrArray`s only for Polyester.jl since
    #     `Threads.@threads` combined with them caused sporadic, huge slowdowns.
    #     The hand-written `@turbo` kernels are still only selected for the `:pool`
    #     backend (see `TurboArray`). Keep this in mind when considering to change this
    #     macro.
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
    part = gensym(:part)

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
                    $Trixi.threaded_foreach($range, $nchunks) do $part
                        for $loopvar in $part
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
        serial_bindings = [:($v = $i) for (v, i) in zip(vars, inits)]
        # The result is assigned outside of the `let` blocks, so that this also works at
        # top level (assigning globals).
        ex = quote
            $var_tuple = let $(let_bindings...)
                $nchunks = $Trixi.threaded_nchunks($range)
                if $nchunks == 1
                    let $(serial_bindings...)
                        $serial_loop
                        $var_tuple
                    end
                else
                    $Trixi.threaded_mapreduce($op_tuple, $init_tuple, $range,
                                              $nchunks) do $part
                        $(local_inits...)
                        for $loopvar in $part
                            $body
                        end
                        $var_tuple
                    end
                end
            end
            nothing
        end
    end
    # Use `esc(quote ... end)` for nested macro calls as suggested in
    # https://github.com/JuliaLang/julia/issues/23221
    return esc(ex)
end
