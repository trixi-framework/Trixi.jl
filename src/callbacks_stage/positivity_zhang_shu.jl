# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PositivityPreservingLimiterZhangShu(; threshold, variables)

The fully-discrete positivity-preserving limiter of
- Zhang, Shu (2011)
  Maximum-principle-satisfying and positivity-preserving high-order schemes
  for conservation laws: survey and new developments
  [doi: 10.1098/rspa.2011.0153](https://doi.org/10.1098/rspa.2011.0153)
The limiter is applied to all scalar `variables` in their given order
using the associated `thresholds` to determine the minimal acceptable values.
The order of the `variables` is important and might have a strong influence
on the robustness.
"""
struct PositivityPreservingLimiterZhangShu{N, Thresholds <: NTuple{N, <:Real},
                                           Variables <: NTuple{N, Any}}
    thresholds::Thresholds
    variables::Variables
end

function PositivityPreservingLimiterZhangShu(; thresholds, variables)
    PositivityPreservingLimiterZhangShu(thresholds, variables)
end

function (limiter!::PositivityPreservingLimiterZhangShu)(u_ode, integrator,
                                                         semi::AbstractSemidiscretization,
                                                         t)
    u = wrap_array(u_ode, semi)
    @trixi_timeit timer() "positivity-preserving limiter" begin
        limiter_zhang_shu!(u, limiter!.thresholds, limiter!.variables,
                           mesh_equations_solver_cache(semi)...)
    end
end

# Iterate over tuples in a type-stable way using "lispy tuple programming",
# similar to https://stackoverflow.com/a/55849398:
# Iterating over tuples of different functions isn't type-stable in general
# but accessing the first element of a tuple is type-stable. Hence, it's good
# to process one element at a time and replace iteration by recursion here.
# Note that you shouldn't use this with too many elements per tuple since the
# compile times can increase otherwise - but a handful of elements per tuple
# is definitely fine.
function limiter_zhang_shu!(u, thresholds::NTuple{N, <:Real}, variables::NTuple{N, Any},
                            mesh, equations, solver, cache) where {N}
    threshold = first(thresholds)
    remaining_thresholds = Base.tail(thresholds)
    variable = first(variables)
    remaining_variables = Base.tail(variables)

    limiter_zhang_shu!(u, threshold, variable, mesh, equations, solver, cache)
    limiter_zhang_shu!(u, remaining_thresholds, remaining_variables, mesh, equations,
                       solver, cache)
    return nothing
end

# terminate the type-stable iteration over tuples
function limiter_zhang_shu!(u, thresholds::Tuple{}, variables::Tuple{},
                            mesh, equations, solver, cache)
    nothing
end

include("positivity_zhang_shu_dg1d.jl")
include("positivity_zhang_shu_dg2d.jl")
include("positivity_zhang_shu_dg3d.jl")
end # @muladd
