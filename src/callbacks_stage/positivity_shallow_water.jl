# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO: TrixiShallowWater: generic wet/dry limiter

"""
    PositivityPreservingLimiterShallowWater(; variables)

The limiter is specifically designed for the shallow water equations.
It is applied to all scalar `variables` in their given order
using the defined `threshold_limiter` from the [`ShallowWaterEquations1D`](@ref) struct
or the [`ShallowWaterEquations2D`](@ref) struct to determine the minimal acceptable values.
The order of the `variables` is important and might have a strong influence
on the robustness.

As opposed to the standard version of the [`PositivityPreservingLimiterZhangShu`](@ref),
nodes with a water height below the `threshold_limiter` are treated in a special way.
To avoid numerical problems caused by velocities close to zero,
the velocity is cut off, such that the node can be identified as "dry". The special feature of the
`ShallowWaterEquations` used here is that the bottom topography is stored as an additional
quantity in the solution vector `u`. However, the value of the bottom topography
should not be changed. That is why, it is not limited.

After the limiting process is applied to all degrees of freedom, for safety reasons,
the `threshold_limiter` is applied again on all the DG nodes in order to avoid water height below.
In the case where the cell mean value is below the threshold before applying the limiter,
there could still be dry nodes afterwards due to the logic of the limiter.

This fully-discrete positivity-preserving limiter is based on the work of
- Zhang, Shu (2011)
  Maximum-principle-satisfying and positivity-preserving high-order schemes
  for conservation laws: survey and new developments
  [doi: 10.1098/rspa.2011.0153](https://doi.org/10.1098/rspa.2011.0153)
"""
struct PositivityPreservingLimiterShallowWater{N, Variables <: NTuple{N, Any}}
    variables::Variables
end

function PositivityPreservingLimiterShallowWater(; variables)
    PositivityPreservingLimiterShallowWater(variables)
end

function (limiter!::PositivityPreservingLimiterShallowWater)(u_ode, integrator,
                                                             semi::AbstractSemidiscretization,
                                                             t)
    u = wrap_array(u_ode, semi)
    @trixi_timeit timer() "positivity-preserving limiter" limiter_shallow_water!(u,
                                                                                 limiter!.variables,
                                                                                 mesh_equations_solver_cache(semi)...)
end

# Iterate over tuples in a type-stable way using "lispy tuple programming",
# similar to https://stackoverflow.com/a/55849398:
# Iterating over tuples of different functions isn't type-stable in general
# but accessing the first element of a tuple is type-stable. Hence, it's good
# to process one element at a time and replace iteration by recursion here.
# Note that you shouldn't use this with too many elements per tuple since the
# compile times can increase otherwise - but a handful of elements per tuple
# is definitely fine.
function limiter_shallow_water!(u, variables::NTuple{N, Any},
                                mesh,
                                equations::Union{ShallowWaterEquations1D,
                                                 ShallowWaterEquations2D},
                                solver, cache) where {N}
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    limiter_shallow_water!(u, equations.threshold_limiter, variable, mesh, equations,
                           solver, cache)
    limiter_shallow_water!(u, remaining_variables, mesh, equations, solver, cache)
    return nothing
end

# terminate the type-stable iteration over tuples
function limiter_shallow_water!(u, variables::Tuple{},
                                mesh,
                                equations::Union{ShallowWaterEquations1D,
                                                 ShallowWaterEquations2D},
                                solver, cache)
    nothing
end

include("positivity_shallow_water_dg1d.jl")
include("positivity_shallow_water_dg2d.jl")
end # @muladd
