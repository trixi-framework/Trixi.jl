@muladd begin
#! format: noindent

"""
    TrixiStateVector(data::AbstractVector)

A thin wrapper around an ODE state vector that provides MPI-distributed linear algebra
operations, enabling compatibility with matrix-free iterative solvers such as those in
[Krylov.jl](https://github.com/JuliaSmoothOptimizers/Krylov.jl).

`TrixiStateVector` subclasses `AbstractVector` and forwards all standard operations to the
underlying `data` field. The two overridden operations are:
- `LinearAlgebra.dot(x, y)`: global MPI reduction via `MPI.Allreduce`
- `LinearAlgebra.norm(x)`: global MPI reduction via `MPI.Allreduce`

Broadcast operations (e.g. `@. u = α * x + y`) preserve the `TrixiStateVector` wrapper
so that the state stays wrapped throughout the time integration loop.

`wrap_array` transparently unwraps the vector to its `data` before passing it to Trixi's
interior spatial-discretization routines, so no changes to `rhs!` or callbacks are needed.

# Examples
```julia
ode = semidiscretize(semi, tspan)  # wrap_state=true by default
sol = solve(ode, alg; ode_default_options()...)
```
"""
struct TrixiStateVector{T, A <: AbstractVector{T}} <: AbstractVector{T}
    data::A
end

# -------------------------------------------------------------------------
# AbstractVector interface – delegate everything to .data
# -------------------------------------------------------------------------

Base.size(v::TrixiStateVector) = size(v.data)
@inline Base.getindex(v::TrixiStateVector, i::Integer) = v.data[i]
@inline Base.setindex!(v::TrixiStateVector, x, i::Integer) = setindex!(v.data, x, i)
Base.IndexStyle(::Type{<:TrixiStateVector}) = IndexLinear()

# Krylov.jl workspace interface: allocate an uninitialized vector of length n.
# Krylov.jl calls `S(undef, n)` where S = typeof(b) to build GMRES workspaces.
function (::Type{TrixiStateVector{T, A}})(::UndefInitializer,
                                          n::Integer) where {T, A <: AbstractVector{T}}
    return TrixiStateVector{T, A}(A(undef, n))
end

# Allocation / copy: always return TrixiStateVector so the wrapper is preserved.
Base.similar(v::TrixiStateVector) = TrixiStateVector(similar(v.data))
function Base.similar(v::TrixiStateVector, ::Type{S}) where {S}
    return TrixiStateVector(similar(v.data, S))
end
function Base.similar(v::TrixiStateVector, ::Type{S}, dims::Dims{1}) where {S}
    return TrixiStateVector(similar(v.data, S, dims))
end
# TrixiStateVector can only wrap vectors, so multi-dimensional `similar` has to
# return a plain array.
function Base.similar(v::TrixiStateVector, ::Type{S}, dims::Dims) where {S}
    return similar(v.data, S, dims)
end
Base.copy(v::TrixiStateVector) = TrixiStateVector(copy(v.data))
function Base.copyto!(dst::TrixiStateVector, src::TrixiStateVector)
    copyto!(dst.data, src.data)
    return dst
end
Base.fill!(v::TrixiStateVector, x) = (fill!(v.data, x); v)
Base.resize!(v::TrixiStateVector, n::Integer) = (resize!(v.data, n); v)

# Pointer / memory layout: lets the existing `unsafe_wrap`-based `wrap_array` implementations
# in dg.jl work without any modifications.
Base.pointer(v::TrixiStateVector{T}) where {T} = pointer(v.data)
function Base.unsafe_convert(::Type{Ptr{T}}, v::TrixiStateVector{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, v.data)
end

# storage_type: required by wrap_array in dg.jl to determine the concrete array constructor.
storage_type(::Type{<:TrixiStateVector{T, A}}) where {T, A} = storage_type(A)

# Wrap a freshly created/loaded state vector for `semidiscretize` if requested.
# Only plain `Vector`s are wrapped: GPU arrays and other special storage types
# (e.g. `VectorOfArray` used by DGMulti) pass through unchanged.
function maybe_wrap_state(u0_ode, wrap_state::Bool)
    if wrap_state && u0_ode isa Vector
        return TrixiStateVector(u0_ode)
    end
    return u0_ode
end

# -------------------------------------------------------------------------
# Distributed linear algebra – MPI global reductions
# -------------------------------------------------------------------------

"""
    LinearAlgebra.dot(x::TrixiStateVector, y::TrixiStateVector)

Compute the global dot product across all MPI ranks.
"""
function LinearAlgebra.dot(x::TrixiStateVector, y::TrixiStateVector)
    local_dot = dot(x.data, y.data)
    if mpi_isparallel()
        return MPI.Allreduce(local_dot, +, mpi_comm())
    else
        return local_dot
    end
end

"""
    LinearAlgebra.norm(v::TrixiStateVector, p=2)

Compute the global L2 norm across all MPI ranks.
Only the 2-norm is supported.
"""
function LinearAlgebra.norm(v::TrixiStateVector, p::Real = 2)
    if p != 2
        throw(ArgumentError("Only the 2-norm (p=2) is supported for TrixiStateVector, got p=$p"))
    end
    local_sum = sum(abs2, v.data)
    if mpi_isparallel()
        global_sum = MPI.Allreduce(local_sum, +, mpi_comm())
        return sqrt(global_sum)
    else
        return sqrt(local_sum)
    end
end

# -------------------------------------------------------------------------
# Broadcast: preserve TrixiStateVector wrapper across stage updates
#
# Uses Broadcast.ArrayStyle so that broadcasting with a TrixiStateVector
# (and scalars / Refs) always produces another TrixiStateVector.  The
# implementation unwraps all TrixiStateVector arguments to their .data
# before delegating to the underlying array's broadcast machinery.
# -------------------------------------------------------------------------

Base.BroadcastStyle(::Type{<:TrixiStateVector}) = Broadcast.ArrayStyle{TrixiStateVector}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TrixiStateVector}},
                      ::Type{ElType}) where {ElType}
    v = _tsv_find(bc)
    return TrixiStateVector(similar(v.data, ElType, axes(bc)))
end

# Materialise a broadcast into a TrixiStateVector.  After stripping all
# TrixiStateVector wrappers from the Broadcasted tree, the result is a plain-
# array Broadcasted that is then evaluated element-wise in a @threaded loop so
# that stage-update expressions (e.g. `@. u = a * u + b * k`) respect the same
# threading backend as the rest of Trixi.jl.
function Base.copyto!(dst::TrixiStateVector,
                      bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TrixiStateVector}})
    bc_plain = Broadcast.instantiate(_tsv_unwrap(bc))
    @threaded for i in eachindex(dst.data)
        @inbounds dst.data[i] = bc_plain[i]
    end
    return dst
end

# Locate the first TrixiStateVector leaf in a Broadcasted tree.
function _tsv_find(bc::Broadcast.Broadcasted)
    for arg in bc.args
        result = _tsv_find_in(arg)
        result !== nothing && return result
    end
    error("No TrixiStateVector found in broadcast expression")
end
_tsv_find_in(v::TrixiStateVector) = v
_tsv_find_in(bc::Broadcast.Broadcasted) = _tsv_find(bc)
_tsv_find_in(::Any) = nothing

# Recursively strip TrixiStateVector wrappers from a Broadcasted tree so that the
# result can be applied to the underlying plain array.
function _tsv_unwrap(bc::Broadcast.Broadcasted{<:Broadcast.ArrayStyle{TrixiStateVector}})
    return Broadcast.Broadcasted{Nothing}(bc.f, map(_tsv_unwrap_arg, bc.args))
end
_tsv_unwrap_arg(v::TrixiStateVector) = v.data
function _tsv_unwrap_arg(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TrixiStateVector}})
    return _tsv_unwrap(bc)
end
_tsv_unwrap_arg(x) = x

# -------------------------------------------------------------------------
# wrap_array integration
#
# Unwrap at the semidiscretization level so the interior DGSEM/FDSBP wrap_array
# implementations receive the plain underlying vector (which they expect).
# The mesh/solver-level wrap_array methods already work via the pointer/storage_type
# overloads above, but explicit methods here document intent clearly.
# -------------------------------------------------------------------------

function wrap_array(u_ode::TrixiStateVector, semi::AbstractSemidiscretization)
    return wrap_array(u_ode.data, mesh_equations_solver_cache(semi)...)
end

function wrap_array_native(u_ode::TrixiStateVector, semi::AbstractSemidiscretization)
    return wrap_array_native(u_ode.data, mesh_equations_solver_cache(semi)...)
end

# -------------------------------------------------------------------------
# KernelAbstractions and Adapt integration
# -------------------------------------------------------------------------

# Required by trixi_backend (containers.jl) which is called from save_solution_file.
KernelAbstractions.get_backend(v::TrixiStateVector) = KernelAbstractions.get_backend(v.data)

# Required by trixi_adapt so that GPU/type adaption propagates through the wrapper.
function Adapt.adapt_structure(to, v::TrixiStateVector)
    return TrixiStateVector(Adapt.adapt(to, v.data))
end
end # @muladd
