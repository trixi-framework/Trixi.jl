
# TODO: MPI. We keep this module inside Trixi for now. When it stabilizes and
#            turns out to be generally useful, we can consider moving it to a
#            separate package with simple test suite and documentation.
module TrixiMPIArrays

using ArrayInterface: ArrayInterface
using MPI: MPI

using ..Trixi: Trixi, mpi_comm

export TrixiMPIArray, local_length, ode_norm, ode_unstable_check


# Dispatch etc.
# The following functions have special dispatch behavior for `TrixiMPIArray`s.
# - `wrap_array`:
#   the wrapped array is wrapped again in a `TrixiMPIArray`
# - `wrap_array_native`:
#   should not be changed since it should return a plain `Array`
# - `allocate_coefficients`:
#   this handles the return type of initialization stuff when setting an IC
#   with MPI
# Besides these, we usually dispatch on MPI mesh types such as
# `mesh::ParallelTreeMesh` or ``mesh::ParallelP4eestMesh`, since this is
# consistent with other dispatches on the mesh type. However, we dispatch on
# `u::TrixiMPIArray` whenever this allows simplifying some code, e.g., because
# we can call a basic function on `parent(u)` and add some MPI stuff on top.
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
end

function TrixiMPIArray(u_local::AbstractArray{T, N}) where {T, N}
  # TODO: MPI. Hard-coded to MPI.COMM_WORLD for now
  mpi_comm = MPI.COMM_WORLD
  TrixiMPIArray{T, N, typeof(u_local)}(u_local, mpi_comm)
end


# Custom interface and general Base interface not covered by other parts below
Base.parent(u::TrixiMPIArray) = u.u_local
Base.resize!(u::TrixiMPIArray, new_size) = resize!(parent(u), new_size)
function Base.copy(u::TrixiMPIArray)
  return TrixiMPIArray(copy(parent(u)), mpi_comm(u))
end

Trixi.mpi_comm(u::TrixiMPIArray) = u.mpi_comm


# Implementation of the abstract array interface of Base
# See https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array
Base.size(u::TrixiMPIArray) = size(parent(u))
Base.getindex(u::TrixiMPIArray, idx) = getindex(parent(u), idx)
Base.setindex!(u::TrixiMPIArray, v, idx) = setindex!(parent(u), v, idx)
Base.IndexStyle(::Type{TrixiMPIArray{T, N, Parent}}) where {T, N, Parent} = IndexStyle(Parent)
function Base.similar(u::TrixiMPIArray, ::Type{S}, dims::NTuple{N, Int}) where {S, N}
  return TrixiMPIArray(similar(parent(u), S, dims), mpi_comm(u))
end
Base.axes(u::TrixiMPIArray)	= axes(parent(u))


# Implementation of the strided array interface of Base
# See https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-strided-arrays
Base.strides(u::TrixiMPIArray) = strides(parent(u))
function Base.unsafe_convert(::Type{Ptr{T}}, u::TrixiMPIArray{T}) where {T}
  return Base.unsafe_convert(Ptr{T}, parent(u))
end
Base.elsize(::Type{TrixiMPIArray{T, N, Parent}}) where {T, N, Parent} = elsize(Parent)


# We need to customize broadcasting since broadcasting expressions allocating
# an output would return plain `Array`s otherwise, losing the MPI information.
# Such allocating broadcasting calls are used for example when determining the
# initial step size in OrdinaryDiffEq.jl.
# However, everything else appears to be fine, i.e., all broadcasting calls
# with a given output storage location work fine. In particular, threaded
# broadcasting with FastBroadcast.jl works fine, e.g., when using threaded RK
# methods such as `SSPRK43(thread=OrdinaryDiffEq.True())`.
# See also
# https://github.com/YingboMa/FastBroadcast.jl
# https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting
Base.BroadcastStyle(::Type{<:TrixiMPIArray}) = Broadcast.ArrayStyle{TrixiMPIArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{TrixiMPIArray}}, ::Type{ElType}) where ElType
  # Scan the inputs for the first TrixiMPIArray and use that to create a `similar`
  # output array with MPI information
  A = find_mpi_array(bc)
  similar(A, axes(bc))
end
# `A = find_mpi_array(As)` returns the first TrixiMPIArray among the arguments
find_mpi_array(bc::Base.Broadcast.Broadcasted) = find_mpi_array(bc.args)
find_mpi_array(args::Tuple) = find_mpi_array(find_mpi_array(args[1]), Base.tail(args))
find_mpi_array(x) = x
find_mpi_array(::Tuple{}) = nothing
find_mpi_array(a::TrixiMPIArray, rest) = a
find_mpi_array(::Any, rest) = find_mpi_array(rest)


# Implementation of methods from ArrayInterface.jl for use with
# LoopVectorization.jl etc.
# See https://juliaarrays.github.io/ArrayInterface.jl/stable/
ArrayInterface.parent_type(::Type{TrixiMPIArray{T, N, Parent}}) where {T, N, Parent} = Parent


# TODO: MPI. For now, we do not implement specializations of LinearAlgebra
#            functions such as `norm` or `dot`. We might revisit this again
#            in the future.


# `mapreduce` functionality from Base using global reductions via MPI communication
# for use in, e.g., error-based step size control in OrdinaryDiffEq.jl
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
# Some consequences are that we need to implement specializations of `show`,
# since the default ones call `length`. However, this doesn't work if not all
# ranks call the same method, e.g., when showing an array only on one rank.
Base.eachindex(u::TrixiMPIArray) = eachindex(parent(u))


function Base.length(u::TrixiMPIArray)
  local_length = length(parent(u))
  return MPI.Allreduce(local_length, +, mpi_comm(u))
end

"""
    local_length(u)

Like `length(u)`, but returns the length of the local data for `u::TrixiMPIArray`.
"""
local_length(u) = length(u)
local_length(u::TrixiMPIArray) = length(parent(u))


# Specializations of `show` without global communication via a global `length`.
# This is necessary when `show`ing `TrixiMPIArray`s only on some ranks, e.g.,
# for development.
function Base.show(io::IO, u::TrixiMPIArray)
  print(io, "TrixiMPIArray wrapping ", parent(u))
end

function Base.show(io::IO, mime::MIME"text/plain", u::TrixiMPIArray)
  print(io, "TrixiMPIArray wrapping ")
  show(io, mime, parent(u))
end


# Specialization of `view`. Without these, `view`s of arrays returned by
# `wrap_array` with multiple conserved variables do not always work...
# This may also be related to the use of a global `length`?
Base.view(u::TrixiMPIArray, idx::Vararg{Any,N}) where {N} = view(parent(u), idx...)

# Specialization of `copyto!` calling `length` under the hood...
function Base.copyto!(dest::TrixiMPIArray, src::TrixiMPIArray)
  copyto!(parent(dest), parent(src))
  return dest
end


"""
    ode_norm(u, t)

Implementation of the weighted L2 norm of Hairer and Wanner used for error-based
step size control in OrdinaryDiffEq.jl. This function is aware of
[`TrixiMPIArray`](@ref)s, handling them appropriately with global MPI
communication.

You must pass this function as keyword argument
`internalnorm=Trixi.ode_norm`
of `solve` when using error-based step size control with MPI parallel execution
of Trixi.jl.
"""
ode_norm(u, t) = @fastmath abs(u)
ode_norm(u::AbstractArray, t) = sqrt(sum(abs2, u) / length(u))
function ode_norm(u::TrixiMPIArray, t)
  local_sumabs2 = sum(abs2, parent(u))
  local_length  = length(parent(u))
  # TODO: MPI. This could be fused into one call to improve parallel performance.
  global_sumabs2 = MPI.Allreduce(local_sumabs2, +, mpi_comm(u))
  global_length  = MPI.Allreduce(local_length,  +, mpi_comm(u))
  return sqrt(global_sumabs2 / global_length)
end


"""
    ode_unstable_check(dt, u, semi, t)

Implementation of the basic check for instability used in OrdinaryDiffEq.jl.
Instead of checking something like `any(isnan, u)`, this function just checks
`isnan(dt)`. This helps when using [`TrixiMPIArray`](@ref)s, since no additional
global communication is required and all ranks will return the same result.

You should pass this function as keyword argument
`unstable_check=Trixi.ode_unstable_check`
of `solve` when using error-based step size control with MPI parallel execution
of Trixi.jl.
"""
ode_unstable_check(dt, u, semi, t) = isnan(dt)


end # module

using .TrixiMPIArrays: TrixiMPIArrays, TrixiMPIArray, local_length, ode_norm, ode_unstable_check
