# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

struct FilterParam{FactorType<:Real}
  relaxation_factor_entropy_var::FactorType
  relaxation_factor_cons_var   ::FactorType
end

struct FilterOperators{InvVDM}
  invVDM::InvVDM
end

struct FilterCache{UModalCoeffs<:AbstractArray,LocalUModalCoeff<:AbstractArray}
  u_modal_coeffs               ::UModalCoeffs
  local_u_modal_coeffs_threaded::LocalUModalCoeff
end

abstract type AdaptiveFilter end
struct SecondOrderExponentialAdaptiveFilter{FactorType<:Real,InvVDM,
                                            UModalCoeffs<:AbstractArray,
                                            LocalUModalCoeff<:AbstractArray} <: AdaptiveFilter
  param::FilterParam{FactorType}
  ops  ::FilterOperators{InvVDM}
  cache::FilterCache{UModalCoeffs,LocalUModalCoeff}
end

function SecondOrderExponentialAdaptiveFilter(; relaxation_factor_entropy_var,
                                                relaxation_factor_cons_var,
                                                mesh::DGMultiMesh{1},
                                                equations,
                                                dg::DGMulti{1})

  rd = dg.basis
  md = mesh.md
  nvars = nvariables(equations)
  uEltype = real(dg)

  # storage for 'unfiltered' modal coefficients of u
  u_modal_coeffs = allocate_nested_array(uEltype, nvars, (rd.Np, md.num_elements), dg) 

  # local storage for 'filtered' modal coefficients of u
  local_u_modal_coeffs_threaded = [allocate_nested_array(uEltype, nvars, (rd.Np,), dg) for _ in 1:Threads.nthreads()]

  SecondOrderExponentialAdaptiveFilter(FilterParam(relaxation_factor_entropy_var,
                                                   relaxation_factor_cons_var),
                                       FilterOperators(inv(rd.VDM)),
                                       FilterCache(u_modal_coeffs,
                                                   local_u_modal_coeffs_threaded))
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
