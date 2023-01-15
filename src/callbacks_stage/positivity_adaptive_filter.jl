# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct FilterParam{FactorType<:Real}
  relaxation_factor_entropy_var::FactorType
  relaxation_factor_cons_var   ::FactorType
end

abstract type AdaptiveFilter end
struct SecondOrderExponentialAdaptiveFilter{FactorType<:Real} <: AdaptiveFilter
  param::FilterParam{FactorType}
end
struct ZhangShuScalingAdaptiveFilter{FactorType<:Real} <: AdaptiveFilter
  param::FilterParam{FactorType}
end

function SecondOrderExponentialAdaptiveFilter(; relaxation_factor_entropy_var,
                                                relaxation_factor_cons_var)
  SecondOrderExponentialAdaptiveFilter(FilterParam(relaxation_factor_entropy_var,
                                                   relaxation_factor_cons_var))
end

function ZhangShuScalingAdaptiveFilter(; relaxation_factor_entropy_var,
                                         relaxation_factor_cons_var)
  ZhangShuScalingAdaptiveFilter(FilterParam(relaxation_factor_entropy_var,
                                            relaxation_factor_cons_var))
end

function get_relaxation_factor_cons_var(filter::AdaptiveFilter)
  return filter.param.relaxation_factor_cons_var
end

function get_relaxation_factor_entropy_var(filter::AdaptiveFilter)
  return filter.param.relaxation_factor_entropy_var
end

struct CompressibleFlowBound{RhoMin, RhoMax, RhoeMin, RhoeMax, VlastMax}
  vlastmax::VlastMax
  ρmin    ::RhoMin
  ρmax    ::RhoMax
  ρemin   ::RhoeMin
  ρemax   ::RhoeMax
end

function AdaptiveFilter(; relaxation_factor_entropy_var, relaxation_factor_cons_var)
  AdaptiveFilter(relaxation_factor_entropy_var, relaxation_factor_cons_var)
end

function (limiter!::AdaptiveFilter)(
    u_ode, integrator, semi::AbstractSemidiscretization, t)
  u = wrap_array(u_ode, semi)
  @trixi_timeit timer() "adaptive filter" adaptive_filter!(u, limiter!, mesh_equations_solver_cache(semi)...)
end

include("positivity_adaptive_filter_dg1d.jl")

end # @muladd
