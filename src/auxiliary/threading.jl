# Shared-memory parallelism for CPU loops: implementation of `@threaded`.
#
# `@threaded` splits the iteration range into `nchunks` contiguous chunks (at most one
# per thread of the default thread pool) and runs one chunk per thread. The way the
# chunks are run is selected by the `backend` preference (see `set_threading_backend!`):
#
# - `:pool` (default): the persistent worker pool of the vendored ParkedWorkers.jl package
#   (`src/auxiliary/vendored/ParkedWorkers`): one sticky worker task per thread of the
#   default thread pool, which spins for a short while after finishing a chunk before
#   going to sleep. Handing chunks to spinning workers is much cheaper than scheduling new
#   tasks, which matters for the many small loops in `rhs!`. The calling task runs the
#   first chunk, and also every chunk that a worker has not picked up yet (e.g., because
#   another process occupies its core). This replaces the role Polyester.jl's `@batch`
#   used to play.
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

# `_THREADING_BACKEND` is defined in `src/Trixi.jl`.

# Number of chunks a threaded loop over `range` is split into.
# Returns `1` if the loop should be run serially, e.g., because only a single thread
# is available, or when called from inside another threaded loop (nested threading).
@inline function threaded_nchunks(range)
    @static if _THREADING_BACKEND === :serial
        return 1
    elseif _THREADING_BACKEND === :pool
        # Also `1` while precompiling and while another task runs a threaded loop
        return ParkedWorkers.nchunks(range)
    else
        n = Threads.threadpoolsize(:default)
        n == 1 && return 1
        ccall(:jl_in_threaded_region, Cint, ()) != 0 && return 1
        return max(1, min(n, length(range)))
    end
end

# The `chunk`-th of `nchunks` contiguous parts of `range`
@inline chunk_range(range, chunk, nchunks) = ParkedWorkers.chunk(range, chunk, nchunks)

# Run `f(chunk)` for `chunk in 1:nchunks`, in parallel
@inline function threaded_run(f::F, nchunks) where {F}
    if nchunks == 1
        f(1)
        return nothing
    end
    @static if _THREADING_BACKEND === :pool
        ParkedWorkers.run_chunks(f, nchunks)
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
# `f` must return a tuple with one entry per entry of `init`.
@inline function threaded_mapreduce_chunks(f::F, ops::Ops, init, nchunks) where {F, Ops}
    if nchunks == 1
        return combine_reductions(ops, init, f(1))
    end
    @static if _THREADING_BACKEND === :pool
        # Combines the chunk results in chunk order, without allocating
        return combine_reductions(ops, init,
                                  ParkedWorkers.run_fold(f, ParkedWorkers.TupleOp(ops),
                                                         nchunks))
    else
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
                    $Trixi.threaded_mapreduce_chunks($op_tuple, $init_tuple,
                                                     $nchunks) do $chunk
                        $(local_inits...)
                        for $loopvar in $Trixi.chunk_range($range, $chunk, $nchunks)
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
