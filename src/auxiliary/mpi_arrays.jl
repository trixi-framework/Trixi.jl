
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

# TODO: MPI. How shall we handle `length`? We want `TrixiMPIArray`s to behave
#            like regular `Array`s in most code, e.g., for `eachindex` etc.
#            However, we need to divide by the `length` of the global array
#            for `ODE_DEFAULT_NORM`. We could specialize `ODE_DEFAULT_NORM`
#            accordingly, but that requires depending on DiffEqBase (instead of
#            SciMLBase). Alternatively, we could implement this via Requires.jl,
#            but that will prevent precompilation and maybe trigger invalidations.
#            Alternatively, we could specialize `length` to return a global
#            length and make sure that all local behavior is still working as
#            expected (if we use `eachindex` instead of `1:length` etc.).
# TODO: Document and describe this stuff
Base.eachindex(u::TrixiMPIArray) = eachindex(parent(u))

function Base.length(u::TrixiMPIArray)
  local_length = length(parent(u))
  return MPI.Allreduce(local_length, +, mpi_comm(u))
end

local_length(u) = length(u)
local_length(u::TrixiMPIArray) = length(parent(u))


end # module


using .TrixiMPIArrays


#= TODO: MPI. Delete development code and notes

# Simple serial test

julia> trixi_include("examples/tree_2d_dgsem/elixir_euler_ec.jl", tspan=(0.0, 10.0))

julia> sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              5.22s /  95.2%           17.4MiB /  96.3%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       4.24k    4.86s   97.8%  1.15ms   1.50MiB    8.9%     370B
   volume integral          4.24k    3.97s   80.0%   938μs     0.00B    0.0%    0.00B
   interface flux           4.24k    561ms   11.3%   132μs     0.00B    0.0%    0.00B
   prolong2interfaces       4.24k    115ms    2.3%  27.1μs     0.00B    0.0%    0.00B
   surface integral         4.24k    110ms    2.2%  25.9μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              4.24k   45.6ms    0.9%  10.8μs     0.00B    0.0%    0.00B
   Jacobian                 4.24k   45.2ms    0.9%  10.7μs     0.00B    0.0%    0.00B
   ~rhs!~                   4.24k   8.63ms    0.2%  2.04μs   1.50MiB    8.9%     370B
   prolong2boundaries       4.24k    343μs    0.0%  80.9ns     0.00B    0.0%    0.00B
   mortar flux              4.24k    251μs    0.0%  59.2ns     0.00B    0.0%    0.00B
   prolong2mortars          4.24k    213μs    0.0%  50.3ns     0.00B    0.0%    0.00B
   boundary flux            4.24k   88.7μs    0.0%  20.9ns     0.00B    0.0%    0.00B
   source terms             4.24k   74.8μs    0.0%  17.7ns     0.00B    0.0%    0.00B
 calculate dt                 848   52.9ms    1.1%  62.4μs     0.00B    0.0%    0.00B
 analyze solution              10   38.5ms    0.8%  3.85ms    159KiB    0.9%  15.9KiB
 I/O                           11   18.8ms    0.4%  1.71ms   15.1MiB   90.2%  1.38MiB
   save solution               10   18.6ms    0.4%  1.86ms   15.1MiB   90.0%  1.51MiB
   get element variables       10    170μs    0.0%  17.0μs   20.6KiB    0.1%  2.06KiB
   ~I/O~                       11   35.9μs    0.0%  3.26μs   7.20KiB    0.0%     671B
   save mesh                   10    554ns    0.0%  55.4ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────


julia> mpi_ode = remake(ode, u0=Trixi.TrixiMPIArray(copy(ode.u0)));

julia> mpi_sol = solve(mpi_ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              5.17s /  95.3%           17.5MiB /  96.3%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       4.24k    4.82s   97.8%  1.14ms   1.63MiB    9.6%     402B
   volume integral          4.24k    3.94s   80.0%   931μs     0.00B    0.0%    0.00B
   interface flux           4.24k    563ms   11.4%   133μs     0.00B    0.0%    0.00B
   surface integral         4.24k    108ms    2.2%  25.5μs     0.00B    0.0%    0.00B
   prolong2interfaces       4.24k    107ms    2.2%  25.2μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              4.24k   45.5ms    0.9%  10.8μs     0.00B    0.0%    0.00B
   Jacobian                 4.24k   43.7ms    0.9%  10.3μs     0.00B    0.0%    0.00B
   ~rhs!~                   4.24k   7.45ms    0.2%  1.76μs   1.63MiB    9.6%     402B
   prolong2mortars          4.24k    341μs    0.0%  80.6ns     0.00B    0.0%    0.00B
   prolong2boundaries       4.24k    311μs    0.0%  73.5ns     0.00B    0.0%    0.00B
   mortar flux              4.24k    228μs    0.0%  53.9ns     0.00B    0.0%    0.00B
   source terms             4.24k   89.5μs    0.0%  21.1ns     0.00B    0.0%    0.00B
   boundary flux            4.24k   88.1μs    0.0%  20.8ns     0.00B    0.0%    0.00B
 calculate dt                 848   52.0ms    1.1%  61.3μs     0.00B    0.0%    0.00B
 analyze solution              10   37.0ms    0.8%  3.70ms    158KiB    0.9%  15.8KiB
 I/O                           11   19.6ms    0.4%  1.78ms   15.1MiB   89.5%  1.38MiB
   save solution               10   19.4ms    0.4%  1.94ms   15.1MiB   89.3%  1.51MiB
   get element variables       10    227μs    0.0%  22.7μs   22.2KiB    0.1%  2.22KiB
   ~I/O~                       11   23.8μs    0.0%  2.16μs   7.20KiB    0.0%     671B
   save mesh                   10    795ns    0.0%  79.5ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────


julia> sol = solve(ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              2.70s /  90.2%           12.1MiB /  82.8%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    2.41s   98.6%  1.02ms    853KiB    8.3%     372B
   volume integral          2.35k    1.94s   79.5%   827μs     0.00B    0.0%    0.00B
   interface flux           2.35k    280ms   11.5%   119μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   65.3ms    2.7%  27.8μs     0.00B    0.0%    0.00B
   surface integral         2.35k   63.8ms    2.6%  27.2μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   25.6ms    1.0%  10.9μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   24.8ms    1.0%  10.6μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   5.16ms    0.2%  2.20μs    853KiB    8.3%     372B
   prolong2boundaries       2.35k    179μs    0.0%  76.4ns     0.00B    0.0%    0.00B
   prolong2mortars          2.35k    156μs    0.0%  66.4ns     0.00B    0.0%    0.00B
   mortar flux              2.35k    145μs    0.0%  61.9ns     0.00B    0.0%    0.00B
   source terms             2.35k   46.3μs    0.0%  19.7ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   45.8μs    0.0%  19.5ns     0.00B    0.0%    0.00B
 analyze solution               6   21.4ms    0.9%  3.57ms   94.7KiB    0.9%  15.8KiB
 I/O                            7   12.2ms    0.5%  1.75ms   9.08MiB   90.7%  1.30MiB
   save solution                6   12.1ms    0.5%  2.01ms   9.06MiB   90.6%  1.51MiB
   get element variables        6   82.2μs    0.0%  13.7μs   12.4KiB    0.1%  2.06KiB
   ~I/O~                        7   64.2μs    0.0%  9.18μs   5.20KiB    0.1%     761B
   save mesh                    6    516ns    0.0%  86.0ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────

julia> mpi_sol = solve(mpi_ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              2.71s /  88.7%           12.3MiB /  82.1%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    2.35s   97.9%  1.00ms    927KiB    9.0%     404B
   volume integral          2.35k    1.91s   79.2%   811μs     0.00B    0.0%    0.00B
   interface flux           2.35k    273ms   11.4%   116μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   60.7ms    2.5%  25.8μs     0.00B    0.0%    0.00B
   surface integral         2.35k   59.3ms    2.5%  25.3μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   25.3ms    1.1%  10.8μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   25.0ms    1.0%  10.7μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   4.88ms    0.2%  2.08μs    927KiB    9.0%     404B
   prolong2mortars          2.35k    187μs    0.0%  79.8ns     0.00B    0.0%    0.00B
   prolong2boundaries       2.35k    152μs    0.0%  64.5ns     0.00B    0.0%    0.00B
   mortar flux              2.35k    122μs    0.0%  51.9ns     0.00B    0.0%    0.00B
   source terms             2.35k   48.0μs    0.0%  20.5ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   45.0μs    0.0%  19.2ns     0.00B    0.0%    0.00B
 analyze solution               6   33.7ms    1.4%  5.61ms   96.5KiB    0.9%  16.1KiB
 I/O                            7   16.3ms    0.7%  2.33ms   9.08MiB   90.1%  1.30MiB
   save solution                6   16.2ms    0.7%  2.69ms   9.06MiB   89.9%  1.51MiB
   get element variables        6    142μs    0.0%  23.6μs   13.3KiB    0.1%  2.22KiB
   ~I/O~                        7   31.4μs    0.0%  4.49μs   5.20KiB    0.1%     761B
   save mesh                    6    552ns    0.0%  92.0ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────


julia> trixi_include("examples/tree_2d_dgsem/elixir_euler_ec.jl", tspan=(0.0, 10.0), volume_flux=flux_ranocha_turbo)

julia> mpi_ode = remake(ode, u0=Trixi.TrixiMPIArray(copy(ode.u0)));

julia> sol = solve(ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              1.46s /  82.7%           12.1MiB /  82.8%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    1.18s   97.5%   500μs    853KiB    8.3%     372B
   volume integral          2.35k    699ms   58.0%   298μs     0.00B    0.0%    0.00B
   interface flux           2.35k    290ms   24.0%   124μs     0.00B    0.0%    0.00B
   surface integral         2.35k   64.3ms    5.3%  27.4μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   64.0ms    5.3%  27.3μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   26.0ms    2.2%  11.1μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   25.9ms    2.1%  11.0μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   5.04ms    0.4%  2.15μs    853KiB    8.3%     372B
   prolong2boundaries       2.35k    184μs    0.0%  78.5ns     0.00B    0.0%    0.00B
   prolong2mortars          2.35k    114μs    0.0%  48.6ns     0.00B    0.0%    0.00B
   mortar flux              2.35k   75.0μs    0.0%  31.9ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   63.2μs    0.0%  26.9ns     0.00B    0.0%    0.00B
   source terms             2.35k   56.8μs    0.0%  24.2ns     0.00B    0.0%    0.00B
 analyze solution               6   18.8ms    1.6%  3.14ms   96.3KiB    0.9%  16.0KiB
 I/O                            7   11.9ms    1.0%  1.70ms   9.08MiB   90.7%  1.30MiB
   save solution                6   11.8ms    1.0%  1.96ms   9.06MiB   90.6%  1.51MiB
   get element variables        6    103μs    0.0%  17.1μs   12.4KiB    0.1%  2.06KiB
   ~I/O~                        7   16.0μs    0.0%  2.28μs   5.20KiB    0.1%     761B
   save mesh                    6    420ns    0.0%  70.0ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────

julia> mpi_sol = solve(mpi_ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              2.66s /  89.3%           12.3MiB /  82.1%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    2.34s   98.7%   997μs    927KiB    9.0%     404B
   volume integral          2.35k    1.90s   80.2%   810μs     0.00B    0.0%    0.00B
   interface flux           2.35k    269ms   11.3%   115μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   59.0ms    2.5%  25.1μs     0.00B    0.0%    0.00B
   surface integral         2.35k   57.3ms    2.4%  24.4μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   23.9ms    1.0%  10.2μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   23.8ms    1.0%  10.1μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   4.47ms    0.2%  1.90μs    927KiB    9.0%     404B
   prolong2boundaries       2.35k    158μs    0.0%  67.1ns     0.00B    0.0%    0.00B
   prolong2mortars          2.35k    104μs    0.0%  44.4ns     0.00B    0.0%    0.00B
   mortar flux              2.35k   93.3μs    0.0%  39.7ns     0.00B    0.0%    0.00B
   source terms             2.35k   62.5μs    0.0%  26.6ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   45.3μs    0.0%  19.3ns     0.00B    0.0%    0.00B
 analyze solution               6   21.2ms    0.9%  3.53ms   96.5KiB    0.9%  16.1KiB
 I/O                            7   10.5ms    0.4%  1.49ms   9.08MiB   90.1%  1.30MiB
   save solution                6   10.3ms    0.4%  1.72ms   9.06MiB   89.9%  1.51MiB
   get element variables        6    104μs    0.0%  17.3μs   13.3KiB    0.1%  2.22KiB
   ~I/O~                        7   13.9μs    0.0%  1.98μs   5.20KiB    0.1%     761B
   save mesh                    6    482ns    0.0%  80.3ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────

=#

