# Vendored from ParkedWorkers.jl by Valentin Churavy (not yet published or registered),
# commit b91941470a5a29dcbc6cd9e55131924a92ecaa10,
# file `src/loops.jl`, formatted with the JuliaFormatter settings of Trixi.jl and otherwise
# unchanged. Do not edit here; change the package and copy it again. Replace this copy by
# a package dependency once ParkedWorkers.jl is registered. See `LICENSE` in this
# directory.
#
# MIT License. Copyright (c) 2026 Valentin Churavy.

# Splitting iteration spaces into chunks, and the user-facing loop API.

# Number of elements of an iteration space
@inline plength(itr::AbstractUnitRange) = length(itr)
@inline plength(itr) = lastindex(itr) - firstindex(itr) + 1

"""
    ParkedWorkers.nchunks(itr) -> Int

Number of chunks a parallel loop over `itr` would be split into right now: at most one per
thread of the `:default` thread pool and at most one per element. `1` means that the loop
runs serially, which is the case with a single thread, inside another parallel loop, and
while another task runs a parallel loop. (During precompilation, loops also run serially,
but that is decided later.)
"""
@inline nchunks(itr) = nchunks_for_length(plength(itr))

# Inlined in front of the serial loop: only plain and relaxed loads here (see
# `pool_busy`). Precompilation is detected later, in `acquire_pool`.
@inline function nchunks_for_length(len::Int)
    len <= 1 && return 1
    nw = nworkers()
    nw == 1 && return 1
    pool_busy() && return 1
    return min(nw, len)
end

# The `chunk`-th of `n` contiguous parts of `1:len`, shifted by `offset`
@inline function chunk_indices(offset::Integer, len::Integer, chunk::Integer, n::Integer)
    lo = offset + div((chunk - 1) * len, n)
    hi = offset + div(chunk * len, n) - 1
    return lo:hi
end

"""
    ParkedWorkers.chunk(itr, chunk, n)

The `chunk`-th of `n` contiguous parts of the iteration space `itr`:

- a sub-range for an `AbstractUnitRange` (with the same offset, e.g. `0:10` or `2:7`);
- a `view` (using linear indices) for other `AbstractArray`s, e.g. `CartesianIndices`,
  `StepRange`s or arrays to loop over by value;
- a generator over `itr[j]` for other indexable collections such as `Tuple`s.
"""
@inline function chunk(itr::AbstractUnitRange, chunk::Integer, n::Integer)
    r = chunk_indices(first(itr), length(itr), chunk, n)
    return oftype(first(itr), first(r)):oftype(first(itr), last(r))
end
@inline function chunk(itr::AbstractArray, chunk::Integer, n::Integer)
    return view(itr, chunk_indices(firstindex(itr), plength(itr), chunk, n))
end
@inline function chunk(itr, chunk::Integer, n::Integer)
    return (itr[j] for j in chunk_indices(firstindex(itr), plength(itr), chunk, n))
end

struct ForeachChunk{F, I}
    f::F
    itr::I
    n::Int
end
@inline function (c::ForeachChunk)(i::Int)
    for x in chunk(c.itr, i, c.n)
        c.f(x)
    end
    return nothing
end

"""
    pforeach(f, itr)

Call `f(x)` for every element `x` of `itr` in parallel and return `nothing`. `itr` must be
an `AbstractUnitRange`, an `AbstractArray`, or a collection with `firstindex`, `lastindex`
and `getindex` (e.g. a `Tuple`).

`itr` is split into one contiguous chunk per thread of the `:default` thread pool (see
[`ParkedWorkers.nchunks`](@ref)); chunk 1 runs on the calling task, the others on the
worker tasks. The call returns when all chunks are done. If chunks throw, the exception of
the first failing chunk is rethrown after all chunks have finished.

```julia
x = rand(1000)
pforeach(eachindex(x)) do i
    x[i] = sqrt(x[i])
end
```
"""
function pforeach(f::F, itr) where {F}
    n = nchunks(itr)
    if n == 1
        for x in itr
            f(x)
        end
    else
        run_chunks(ForeachChunk(f, itr, n), n)
    end
    return nothing
end

struct NoInit end

struct MapReduceChunk{F, Op, I, Init}
    f::F
    op::Op
    itr::I
    n::Int
    init::Init
end
@inline function (c::MapReduceChunk{F, Op, I, NoInit})(i::Int) where {F, Op, I}
    return mapreduce(c.f, c.op, chunk(c.itr, i, c.n))
end
@inline function (c::MapReduceChunk)(i::Int)
    return mapreduce(c.f, c.op, chunk(c.itr, i, c.n); init = c.init)
end

"""
    pmapreduce(f, op, itr; [init])

Parallel `mapreduce(f, op, itr; [init])`. Each chunk of `itr` (see [`pforeach`](@ref))
is reduced with `mapreduce`, then the chunk results are combined with `op` in chunk order.
`op` must be associative. `init`, if given, must be a neutral element of `op`, since it is
used once per chunk. With floating-point numbers, the result can differ from a serial
`mapreduce` in the last digits, and depends on the number of chunks, but not on timing.

```julia
pmapreduce(abs2, +, x)          # sum of squares
pmapreduce(i -> f(x[i]), max, eachindex(x))
```
"""
function pmapreduce(f::F, op::Op, itr; init = NoInit()) where {F, Op}
    n = nchunks(itr)
    if n == 1
        return init isa NoInit ? mapreduce(f, op, itr) : mapreduce(f, op, itr; init)
    end
    return run_fold(MapReduceChunk(f, op, itr, n, init), op, n)
end

"""
    preduce(op, itr; [init])

Parallel `reduce(op, itr; [init])`, i.e. `pmapreduce(identity, op, itr; [init])`.
"""
preduce(op::Op, itr; kw...) where {Op} = pmapreduce(identity, op, itr; kw...)

"""
    ParkedWorkers.neutral(op, x)

Neutral element of the reduction operator `op` with the same type as `x`, used as the
initial value of the per-chunk accumulators of `@pfor reduction = (op, var)`. Defined for
`+`, `*`, `max`, `min`, `&`, `|` and `xor`. Add a method for a custom operator, or pass the
neutral element explicitly with `reduction = (op, var, neutral)`.
"""
function neutral end
neutral(::typeof(+), x) = zero(x)
neutral(::typeof(*), x) = one(x)
neutral(::typeof(max), x) = typemin(x)
neutral(::typeof(min), x) = typemax(x)
neutral(::typeof(&), x::Bool) = true
neutral(::typeof(&), x::Integer) = ~zero(x)
neutral(::typeof(|), x) = zero(x)
neutral(::typeof(xor), x) = zero(x)
function neutral(op, x)
    throw(ArgumentError("no neutral element of `$op` known; add a method " *
                        "to `ParkedWorkers.neutral` or use " *
                        "`reduction = (op, var, neutral)`"))
end

# Combine tuples of partial reduction results
struct TupleOp{Ops <: Tuple}
    ops::Ops
end
@inline (t::TupleOp)(a::Tuple, b::Tuple) = map((op, x, y) -> op(x, y), t.ops, a, b)

"""
    @pfor for x in itr
        ...
    end
    @pfor reduction = (op, var) for x in itr ... end
    @pfor reduction = (op, var, neutral) for x in itr ... end
    @pfor reduction = ((op1, var1), (op2, var2), ...) for x in itr ... end

Run a `for` loop in parallel on the persistent worker pool. `itr` is split into one
contiguous chunk per thread of the `:default` thread pool, chunk 1 runs on the calling
task (see [`pforeach`](@ref) for the supported iteration spaces). When the loop would get a
single chunk (one thread or element, a nested `@pfor`, another task currently running a
parallel loop, precompilation), the plain serial loop is run instead.

`reduction` reduces local variables with the given operators, with the same syntax as
Polyester.jl's `@batch`: every chunk starts its own copy of `var` from the neutral element of
`op` (see [`ParkedWorkers.neutral`](@ref)), and after the loop `var` is the initial value
combined with the chunk results in chunk order. `op` must be associative.

```julia
s = 0.0
m = -Inf
@pfor reduction = ((+, s), (max, m)) for i in eachindex(x)
    s += x[i]
    m = max(m, x[i])
end
```

!!! warning
    The loop body runs in a closure. `break` and `return` only end the current chunk, so
    do not use them. A variable of the enclosing function that is assigned more than once
    (e.g. in both branches of an `if`) and is used in the body is captured as a
    `Core.Box`, which makes the loop type-unstable and allocating; assign it once (e.g. by
    a single destructuring assignment) or move the loop into a function.
"""
macro pfor(args...)
    isempty(args) && throw(ArgumentError("@pfor requires a `for` loop"))
    expr = args[end]
    reductions = Any[]
    for arg in args[1:(end - 1)]
        if Meta.isexpr(arg, :(=)) && arg.args[1] === :reduction
            spec = arg.args[2]
            Meta.isexpr(spec, :tuple) ||
                throw(ArgumentError("`reduction` must be `(op, var)` or a tuple of those"))
            if all(x -> Meta.isexpr(x, :tuple), spec.args)
                append!(reductions, spec.args)
            else
                push!(reductions, spec)
            end
        else
            throw(ArgumentError("unsupported argument `$(arg)` to @pfor"))
        end
    end
    for r in reductions
        (2 <= length(r.args) <= 3 && r.args[2] isa Symbol) ||
            throw(ArgumentError("invalid reduction `$(r)`, expected `(op, var)` or " *
                                "`(op, var, neutral)`"))
    end
    Meta.isexpr(expr, :for) || throw(ArgumentError("@pfor requires a `for` loop"))
    loop_spec = expr.args[1]
    Meta.isexpr(loop_spec, :(=)) ||
        throw(ArgumentError("@pfor only supports a single loop variable"))
    loopvar = loop_spec.args[1]
    itr_expr = loop_spec.args[2]
    body = expr.args[2]

    PW = ParkedWorkers
    itr = gensym(:itr)
    n = gensym(:n)
    c = gensym(:chunk)

    if isempty(reductions)
        ex = quote
            let $itr = $itr_expr
                $n = $PW.nchunks($itr)
                if $n == 1
                    # A plain loop: faster than calling the closure for some kernels
                    for $loopvar in $itr
                        $body
                    end
                else
                    $PW.run_chunks($n) do $c
                        for $loopvar in $PW.chunk($itr, $c, $n)
                            $body
                        end
                        nothing
                    end
                end
            end
            nothing
        end
    else
        ops = [r.args[1] for r in reductions]
        vars = Symbol[r.args[2] for r in reductions]
        inits = [gensym(Symbol(v, :_init)) for v in vars]
        local_inits = map(reductions, vars, inits) do r, v, i
            nv = length(r.args) == 3 ? r.args[3] : :($PW.neutral($(r.args[1]), $i))
            :(local $v = $nv)
        end
        ops_tuple = Expr(:tuple, ops...)
        vars_tuple = Expr(:tuple, vars...)
        inits_tuple = Expr(:tuple, inits...)
        let_bindings = Any[:($itr = $itr_expr)]
        append!(let_bindings, [:($i = $v) for (i, v) in zip(inits, vars)])
        serial_bindings = [:($v = $i) for (v, i) in zip(vars, inits)]
        ops_var = gensym(:ops)
        # The result is assigned outside of the `let` blocks, so that this also works at
        # top level (assigning globals) and does not make `var` a captured variable.
        ex = quote
            $vars_tuple = let $(let_bindings...)
                $n = $PW.nchunks($itr)
                if $n == 1
                    let $(serial_bindings...)
                        for $loopvar in $itr
                            $body
                        end
                        $vars_tuple
                    end
                else
                    $ops_var = $PW.TupleOp($ops_tuple)
                    $ops_var($inits_tuple,
                             $PW.run_fold($ops_var, $n) do $c
                                 $(local_inits...)
                                 for $loopvar in $PW.chunk($itr, $c, $n)
                                     $body
                                 end
                                 $vars_tuple
                             end)
                end
            end
            nothing
        end
    end
    return esc(ex)
end
