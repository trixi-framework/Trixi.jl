
# TODO: MPI. Keep this module inside Trixi or move it to another repo as
#            external dependency with simple test suite and documentation?
module TrixiMPIArrays

using ArrayInterface: ArrayInterface
using MPI: MPI

import ..Trixi: mpi_comm, mpi_rank

export TrixiMPIArray, local_length


"""
    TrixiMPIArray{T, N} <: AbstractArray{T, N}

A thin wrapper of arrays distributed via MPI used in Trixi.jl. The idea is that
these arrays behave as much as possible as plain arrays would in an SPMD-style
distributed MPI setting with exception of reductions, which are performed
globally. This allows to use these arrays in ODE solvers such as the ones from
OrdinaryDiffEq.jl, since vector space operations, broadcasting, and reductions
are the only operations required for explicit time integration methods with
fixed step sizes or adaptive step sizes based on CFL or error estimates.

!!! warning "Experimental code"
    This code is experimental and may be changed or removed in any future release.
"""
struct TrixiMPIArray{T, N, Parent<:AbstractArray{T, N}} <: AbstractArray{T, N}
  u_local::Parent
  mpi_comm::MPI.Comm
  mpi_rank::Int
  # TODO: MPI. Shall we also include something like the following fields
  #       and remove them from the global state? Do we ever need something
  #       from the global MPI state without having a state vector `u`? Does
  #       including these fields here have a performance impact since it
  #       increases the size of these arrays?
  # mpi_size::Int
  # mpi_isroot::Bool
  # mpi_isparallel::Bool
end

function TrixiMPIArray(u_local::AbstractArray{T, N}) where {T, N}
  # TODO: MPI. Hard-coded to MPI.COMM_WORLD for now
  mpi_comm = MPI.COMM_WORLD
  mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
  TrixiMPIArray{T, N, typeof(u_local)}(u_local, mpi_comm, mpi_rank)
end


# TODO: MPI. Adapt
# - wrap_array - done
# - wrap_array_native - should not be changed since it should return a plain `Array`
# - return type of initialization stuff when setting an IC with MPI - handled
#   by allocate_coefficients, done
# - dispatch on this array type instead of parallel trees etc. and use
#   `parent(u)` to get local versions instead of `invoke`
#   TODO: MPI. When do we want to dispatch on `u::TrixiMPIArray` and when on
#              an MPI parallel mesh type? Right now, we dispatch a lot on
#              something like `mesh::ParallelTreeMesh{2}`, which can be nice
#              since it is consistent with dispatch on `mesh::TreeMesh{2}`
#              in the general case. At first, I just dispatched on TrixiMPIArray
#              a few times, in particular when using `parent` instead of `invoke`
#              machinery.


# Custom interface and general Base interface not covered by other parts below
Base.parent(u::TrixiMPIArray) = u.u_local
Base.resize!(u::TrixiMPIArray, new_size) = resize!(parent(u), new_size)

mpi_comm(u::TrixiMPIArray) = u.mpi_comm
mpi_rank(u::TrixiMPIArray) = u.mpi_rank
# TODO: MPI. What about the following interface functions?
# mpi_nranks(u::TrixiMPIArray) = MPI_SIZE[]
# mpi_isparallel(u::TrixiMPIArray) = MPI_IS_PARALLEL[]


# Implementation of the abstract array interface of Base
# See https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array
Base.size(u::TrixiMPIArray) = size(parent(u))
Base.getindex(u::TrixiMPIArray, idx) = getindex(parent(u), idx)
Base.setindex!(u::TrixiMPIArray, v, idx) = setindex!(parent(u), v, idx)
Base.IndexStyle(::Type{TrixiMPIArray{T, N, Parent}}) where {T, N, Parent} = IndexStyle(Parent)
Base.similar(u::TrixiMPIArray, ::Type{S}, dims::NTuple{N, Int}) where {S, N}	= TrixiMPIArray(similar(parent(u), S, dims))
Base.axes(u::TrixiMPIArray)	= axes(parent(u))


# Implementation of the strided array interface of Base
# See https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-strided-arrays
Base.strides(u::TrixiMPIArray) = strides(parent(u))
Base.unsafe_convert(::Type{Ptr{T}}, u::TrixiMPIArray{T}) where {T} = Base.unsafe_convert(Ptr{T}, parent(u))
Base.elsize(::Type{TrixiMPIArray{T, N, Parent}}) where {T, N, Parent} = elsize(Parent)


# TODO: MPI. Do we need customized broadcasting? What about FastBroadcast.jl and
#            threaded execution with `@.. thread=true`?
# See https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting


# TODO: MPI. How shall we handle specializations such as `split_form_kernel!(_du::PtrArray, u_cons::PtrArray,`
#            for `flux_ranocha_turbo` and `flux_shima_etal_turbo`?


# Implementation of methods from ArrayInterface.jl for use with
# LoopVectorization.jl etc.
# See https://juliaarrays.github.io/ArrayInterface.jl/stable/
ArrayInterface.parent_type(::Type{TrixiMPIArray{T, N, Parent}}) where {T, N, Parent} = Parent


# TODO: MPI. Do we need LinearAlgebra methods such as `norm` or `dot`?


# `mapreduce` functionality from Base using global reductions via MPI communication
function Base.mapreduce(f::F, op::Op, u::TrixiMPIArray; kwargs...) where {F, Op}
  local_value = mapreduce(f, op, parent(u); kwargs...)
  return MPI.Allreduce(local_value, op, mpi_comm(u))
end


# TODO: MPI. Default settings of OrdinaryDiffEq etc.
# Interesting options could be
# - ODE_DEFAULT_UNSTABLE_CHECK
# - ODE_DEFAULT_ISOUTOFDOMAIN
# - ODE_DEFAULT_NORM
# See https://github.com/SciML/DiffEqBase.jl/blob/master/src/common_defaults.jl

# Problems and inconsistencies with TrixiMPIArrays
# A basic question is how to handle `length`. We want `TrixiMPIArray`s to behave
# like regular `Array`s in most code, e.g., when looping over an array (which
# should use `eachindex`). At the same time, we want to be able to use adaptive
# time stepping using error estimates in OrdinaryDiffEq.jl. There, the default
# norm `ODE_DEFAULT_NORM` is the one described in the book of Hairer & Wanner,
# i.e., it includes a division by the global `length` of the array. We could
# specialize `ODE_DEFAULT_NORM` accordingly, but that requires depending on
# DiffEqBase.jl (instead of SciMLBase.jl). Alternatively, we could implement
# this via Requires.jl, but that will prevent precompilation and maybe trigger
# invalidations. Alternatively, could implement our own norm and pass that as
# `internalnorm`. Here, we decide to use the least intrusive approach for now
# and specialize `length` to return a global length while making sure that all
# local behavior is still working as expected (if using `eachindex` instead of
# `1:length` etc.). This means that `eachindex(u) != Base.OneTo(length(u))` for
# `u::TrixiMPIArray` in general, even if `u` and its underlying array use
# one-based indexing.
Base.eachindex(u::TrixiMPIArray) = eachindex(parent(u))

function Base.length(u::TrixiMPIArray)
  local_length = length(parent(u))
  return MPI.Allreduce(local_length, +, mpi_comm(u))
end

local_length(u) = length(u)
local_length(u::TrixiMPIArray) = length(parent(u))


end # module

using .TrixiMPIArrays: TrixiMPIArrays, TrixiMPIArray, local_length
