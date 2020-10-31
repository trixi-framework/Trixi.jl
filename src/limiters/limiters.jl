
"""
    PositivityPreservingLimiterZhangShu(; threshold, variables)

The fully-discrete positivity-preserving limiter of
- Zhang, Shu (2011)
  Maximum-principle-satisfying and positivity-preserving high-order schemes
  for conservation laws: survey and new developments
  [doi: 10.1098/rspa.2011.0153](https://doi.org/10.1098/rspa.2011.0153)
The limiter is applied to all scalar `variables` in their given order
using the associated `thresholds` to determine the minimal acceptable values.
"""
struct PositivityPreservingLimiterZhangShu{N, Thresholds<:NTuple{N,<:Real}, Variables<:NTuple{N,Any}}
  thresholds::Thresholds
  variables::Variables
end

function PositivityPreservingLimiterZhangShu(; thresholds, variables)
  PositivityPreservingLimiterZhangShu(thresholds, variables)
end


function (limiter!::PositivityPreservingLimiterZhangShu)(
    u_ode::AbstractVector, f, semi::AbstractSemidiscretization, t)
  u = wrap_array(u_ode, semi)
  @timeit_debug timer() "positivity-preserving limiter" limiter_zhang_shu!(
    u, limiter!.thresholds, limiter!.variables, mesh_equations_solver_cache(semi)...)
end


# iterate over tuples in a type-stable way
function limiter_zhang_shu!(u::AbstractArray{<:Any},
                            thresholds::NTuple{N,<:Real}, variables::NTuple{N,Any},
                            mesh, equations, solver, cache) where {N}
  threshold = first(thresholds)
  remaining_thresholds = Base.tail(thresholds)
  variable = first(variables)
  remaining_variables = Base.tail(variables)

  limiter_zhang_shu!(u, threshold, variable, mesh, equations, solver, cache)
  limiter_zhang_shu!(u, remaining_thresholds, remaining_variables, mesh, equations, solver, cache)
  return nothing
end

# terminate the type-stable iteration over tuples
function limiter_zhang_shu!(u::AbstractArray{<:Any},
                            thresholds::Tuple{}, variables::Tuple{},
                            mesh, equations, solver, cache)
  nothing
end

include("limiters_dg3d.jl")
