
# TODO: MPI. Keep this module inside Trixi or move it to another repo as
#            external dependency with simple test suite and documentation?
module TrixiMPIArrays

using ArrayInterface: ArrayInterface
using MPI: MPI

import ..Trixi: mpi_comm, mpi_rank

export TrixiMPIArray


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

  function TrixiMPIArray{T, N, Parent}(u_local::Parent) where {T, N, Parent<:AbstractArray{T, N}}
    # TODO: MPI. Hard-coded to MPI.COMM_WORLD for now
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    return new{T, N, Parent}(u_local, mpi_comm, mpi_rank)
  end
end

function TrixiMPIArray(u_local::AbstractArray{T, N}) where {T, N}
  TrixiMPIArray{T, N, typeof(u_local)}(u_local)
end


# TODO: MPI. Adapt
# - wrap_array
# - wrap_array_native
# - return type of initialization stuff when setting an IC
# - dispatch on this array type instead of parallel trees etc. and use
#   `parent(u)` to get local versions instead of `invoke`


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


# TODO: MPI. Do we need customized broadcasting?
# See https://docs.julialang.org/en/v1/manual/interfaces/#man-interfaces-broadcasting


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


end # module


using .TrixiMPIArrays


#= TODO: MPI. Delete development code and notes

# Simple serial test

julia> trixi_include("examples/tree_2d_dgsem/elixir_euler_ec.jl", tspan=(0.0, 10.0))

julia> sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, ave_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              5.42s /  90.8%            864MiB /   1.9%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       4.24k    4.82s   97.8%  1.14ms   1.50MiB    8.9%     370B
   volume integral          4.24k    3.96s   80.3%   934μs     0.00B    0.0%    0.00B
   interface flux           4.24k    540ms   11.0%   127μs     0.00B    0.0%    0.00B
   surface integral         4.24k    111ms    2.2%  26.1μs     0.00B    0.0%    0.00B
   prolong2interfaces       4.24k    109ms    2.2%  25.7μs     0.00B    0.0%    0.00B
   Jacobian                 4.24k   45.6ms    0.9%  10.8μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              4.24k   45.6ms    0.9%  10.8μs     0.00B    0.0%    0.00B
   ~rhs!~                   4.24k   8.19ms    0.2%  1.93μs   1.50MiB    8.9%     370B
   prolong2mortars          4.24k    282μs    0.0%  66.7ns     0.00B    0.0%    0.00B
   prolong2boundaries       4.24k    279μs    0.0%  66.0ns     0.00B    0.0%    0.00B
   mortar flux              4.24k    174μs    0.0%  41.1ns     0.00B    0.0%    0.00B
   boundary flux            4.24k    105μs    0.0%  24.8ns     0.00B    0.0%    0.00B
   source terms             4.24k    102μs    0.0%  24.1ns     0.00B    0.0%    0.00B
 calculate dt                 848   51.6ms    1.0%  60.9μs     0.00B    0.0%    0.00B
 analyze solution              10   37.5ms    0.8%  3.75ms    160KiB    0.9%  16.0KiB
 I/O                           11   19.9ms    0.4%  1.81ms   15.1MiB   90.2%  1.38MiB
   save solution               10   19.8ms    0.4%  1.98ms   15.1MiB   90.0%  1.51MiB
   get element variables       10   83.5μs    0.0%  8.35μs   20.6KiB    0.1%  2.06KiB
   ~I/O~                       11   35.6μs    0.0%  3.24μs   7.20KiB    0.0%     671B
   save mesh                   10    912ns    0.0%  91.2ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────

julia> mpi_ode = remake(ode, u0=Trixi.TrixiMPIArray(copy(ode.u0)));

julia> mpi_sol = solve(mpi_ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, ave_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              5.42s /  90.6%            863MiB /   1.8%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       4.24k    4.81s   97.9%  1.13ms   9.33KiB    0.1%    2.25B
   volume integral          4.24k    3.96s   80.6%   935μs     0.00B    0.0%    0.00B
   interface flux           4.24k    542ms   11.0%   128μs     0.00B    0.0%    0.00B
   prolong2interfaces       4.24k    106ms    2.2%  25.0μs     0.00B    0.0%    0.00B
   surface integral         4.24k    104ms    2.1%  24.7μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              4.24k   44.6ms    0.9%  10.5μs     0.00B    0.0%    0.00B
   Jacobian                 4.24k   43.4ms    0.9%  10.2μs     0.00B    0.0%    0.00B
   ~rhs!~                   4.24k   7.10ms    0.1%  1.68μs   9.33KiB    0.1%    2.25B
   prolong2boundaries       4.24k    271μs    0.0%  64.0ns     0.00B    0.0%    0.00B
   prolong2mortars          4.24k    242μs    0.0%  57.1ns     0.00B    0.0%    0.00B
   mortar flux              4.24k    188μs    0.0%  44.4ns     0.00B    0.0%    0.00B
   source terms             4.24k   83.7μs    0.0%  19.8ns     0.00B    0.0%    0.00B
   boundary flux            4.24k   74.8μs    0.0%  17.7ns     0.00B    0.0%    0.00B
 calculate dt                 848   50.0ms    1.0%  58.9μs     0.00B    0.0%    0.00B
 analyze solution              10   36.7ms    0.7%  3.67ms    155KiB    1.0%  15.5KiB
 I/O                           11   18.6ms    0.4%  1.69ms   15.1MiB   98.9%  1.38MiB
   save solution               10   18.4ms    0.4%  1.84ms   15.1MiB   98.8%  1.51MiB
   get element variables       10    101μs    0.0%  10.1μs   22.2KiB    0.1%  2.22KiB
   ~I/O~                       11   22.6μs    0.0%  2.05μs   7.20KiB    0.0%     671B
   save mesh                   10    876ns    0.0%  87.6ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────

julia> sol = solve(ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              2.84s /  91.6%           12.1MiB /  82.8%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    2.34s   89.9%   995μs    853KiB    8.3%     372B
   volume integral          2.35k    1.90s   73.1%   808μs     0.00B    0.0%    0.00B
   interface flux           2.35k    268ms   10.3%   114μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   59.3ms    2.3%  25.3μs     0.00B    0.0%    0.00B
   surface integral         2.35k   57.7ms    2.2%  24.6μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   24.2ms    0.9%  10.3μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   23.5ms    0.9%  10.0μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   4.44ms    0.2%  1.89μs    853KiB    8.3%     372B
   prolong2mortars          2.35k    162μs    0.0%  69.0ns     0.00B    0.0%    0.00B
   prolong2boundaries       2.35k    159μs    0.0%  67.6ns     0.00B    0.0%    0.00B
   mortar flux              2.35k    102μs    0.0%  43.4ns     0.00B    0.0%    0.00B
   source terms             2.35k   46.2μs    0.0%  19.7ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   42.5μs    0.0%  18.1ns     0.00B    0.0%    0.00B
 I/O                            7    240ms    9.2%  34.3ms   9.08MiB   90.7%  1.30MiB
   save solution                6    240ms    9.2%  40.0ms   9.06MiB   90.6%  1.51MiB
   get element variables        6   61.1μs    0.0%  10.2μs   12.4KiB    0.1%  2.06KiB
   ~I/O~                        7   15.1μs    0.0%  2.15μs   5.20KiB    0.1%     761B
   save mesh                    6    499ns    0.0%  83.2ns     0.00B    0.0%    0.00B
 analyze solution               6   22.1ms    0.9%  3.68ms   96.3KiB    0.9%  16.0KiB
 ────────────────────────────────────────────────────────────────────────────────────

julia> mpi_sol = solve(mpi_ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              2.88s /  89.3%           11.4MiB /  80.7%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    2.54s   98.7%  1.08ms   9.33KiB    0.1%    4.07B
   volume integral          2.35k    2.06s   80.3%   878μs     0.00B    0.0%    0.00B
   interface flux           2.35k    289ms   11.3%   123μs     0.00B    0.0%    0.00B
   surface integral         2.35k   64.4ms    2.5%  27.4μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   63.1ms    2.5%  26.9μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   26.1ms    1.0%  11.1μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   25.6ms    1.0%  10.9μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   4.74ms    0.2%  2.02μs   9.33KiB    0.1%    4.07B
   prolong2mortars          2.35k    166μs    0.0%  70.6ns     0.00B    0.0%    0.00B
   prolong2boundaries       2.35k    151μs    0.0%  64.2ns     0.00B    0.0%    0.00B
   mortar flux              2.35k    101μs    0.0%  43.0ns     0.00B    0.0%    0.00B
   source terms             2.35k   49.5μs    0.0%  21.1ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   45.3μs    0.0%  19.3ns     0.00B    0.0%    0.00B
 analyze solution               6   21.7ms    0.8%  3.62ms   94.1KiB    1.0%  15.7KiB
 I/O                            7   11.1ms    0.4%  1.58ms   9.08MiB   98.9%  1.30MiB
   save solution                6   10.9ms    0.4%  1.82ms   9.06MiB   98.7%  1.51MiB
   get element variables        6    117μs    0.0%  19.6μs   13.3KiB    0.1%  2.22KiB
   ~I/O~                        7   17.8μs    0.0%  2.54μs   5.20KiB    0.1%     761B
   save mesh                    6    826ns    0.0%   138ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────


julia> trixi_include("examples/tree_2d_dgsem/elixir_euler_ec.jl", tspan=(0.0, 10.0), volume_flux=flux_ranocha_turbo)

julia> mpi_ode = remake(ode, u0=Trixi.TrixiMPIArray(copy(ode.u0)));

julia> sol = solve(ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              1.40s /  82.3%           12.1MiB /  82.8%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    1.12s   97.3%   478μs    853KiB    8.3%     372B
   volume integral          2.35k    665ms   57.6%   283μs     0.00B    0.0%    0.00B
   interface flux           2.35k    278ms   24.1%   119μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   63.3ms    5.5%  27.0μs     0.00B    0.0%    0.00B
   surface integral         2.35k   61.3ms    5.3%  26.1μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   24.8ms    2.1%  10.6μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   24.7ms    2.1%  10.5μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   4.52ms    0.4%  1.92μs    853KiB    8.3%     372B
   prolong2boundaries       2.35k    128μs    0.0%  54.6ns     0.00B    0.0%    0.00B
   prolong2mortars          2.35k    111μs    0.0%  47.5ns     0.00B    0.0%    0.00B
   mortar flux              2.35k   67.0μs    0.0%  28.5ns     0.00B    0.0%    0.00B
   source terms             2.35k   53.0μs    0.0%  22.6ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   47.3μs    0.0%  20.1ns     0.00B    0.0%    0.00B
 analyze solution               6   20.1ms    1.7%  3.35ms   96.3KiB    0.9%  16.0KiB
 I/O                            7   11.6ms    1.0%  1.65ms   9.08MiB   90.7%  1.30MiB
   save solution                6   11.4ms    1.0%  1.91ms   9.06MiB   90.6%  1.51MiB
   get element variables        6   95.8μs    0.0%  16.0μs   12.4KiB    0.1%  2.06KiB
   ~I/O~                        7   18.3μs    0.0%  2.61μs   5.20KiB    0.1%     761B
   save mesh                    6    816ns    0.0%   136ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────

julia> mpi_sol = solve(mpi_ode, RDPK3SpFSAL35(), abstol=1.0e-4, reltol=1.0e-4, save_everystep=false, callback=callbacks); summary_callback()

 ────────────────────────────────────────────────────────────────────────────────────
              Trixi.jl                      Time                    Allocations
                                   ───────────────────────   ────────────────────────
         Tot / % measured:              1.51s /  78.9%           11.4MiB /  80.7%

 Section                   ncalls     time    %tot     avg     alloc    %tot      avg
 ────────────────────────────────────────────────────────────────────────────────────
 rhs!                       2.35k    1.15s   96.5%   492μs   9.33KiB    0.1%    4.07B
   volume integral          2.35k    693ms   58.0%   295μs     0.00B    0.0%    0.00B
   interface flux           2.35k    280ms   23.4%   119μs     0.00B    0.0%    0.00B
   prolong2interfaces       2.35k   64.0ms    5.4%  27.2μs     0.00B    0.0%    0.00B
   surface integral         2.35k   62.3ms    5.2%  26.5μs     0.00B    0.0%    0.00B
   Jacobian                 2.35k   25.1ms    2.1%  10.7μs     0.00B    0.0%    0.00B
   reset ∂u/∂t              2.35k   24.6ms    2.1%  10.5μs     0.00B    0.0%    0.00B
   ~rhs!~                   2.35k   4.55ms    0.4%  1.94μs   9.33KiB    0.1%    4.07B
   prolong2mortars          2.35k    173μs    0.0%  73.7ns     0.00B    0.0%    0.00B
   prolong2boundaries       2.35k    142μs    0.0%  60.3ns     0.00B    0.0%    0.00B
   mortar flux              2.35k   73.5μs    0.0%  31.3ns     0.00B    0.0%    0.00B
   boundary flux            2.35k   46.4μs    0.0%  19.8ns     0.00B    0.0%    0.00B
   source terms             2.35k   43.8μs    0.0%  18.7ns     0.00B    0.0%    0.00B
 analyze solution               6   25.1ms    2.1%  4.19ms   94.1KiB    1.0%  15.7KiB
 I/O                            7   16.2ms    1.4%  2.32ms   9.08MiB   98.9%  1.30MiB
   save solution                6   15.7ms    1.3%  2.62ms   9.06MiB   98.7%  1.51MiB
   get element variables        6    444μs    0.0%  74.1μs   13.3KiB    0.1%  2.22KiB
   ~I/O~                        7   22.7μs    0.0%  3.25μs   5.20KiB    0.1%     761B
   save mesh                    6    536ns    0.0%  89.3ns     0.00B    0.0%    0.00B
 ────────────────────────────────────────────────────────────────────────────────────


=#
