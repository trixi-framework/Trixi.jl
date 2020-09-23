
mutable struct AMRCallback{RealT<:Real, Indicator, Adaptor, Cache}
  interval::Int
  threshold_refinement::RealT
  threshold_coarsening::RealT
  indicator::Indicator
  adaptor::Adaptor
  cache::Cache
end


# TODO: Taal refactor, decide where to move this and implement everything
"""
    IndicatorHennemannGassner

Indicator used for shock-capturing or AMR used by
- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
struct IndicatorHennemannGassner{RealT<:Real}
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
end


struct IndicatorTwoLevel{RealT<:Real, Indicator}
  base_level::Int
  base_threshold::RealT
  max_level::Int
  max_threshold::RealT
  indicator::Indicator
end


"""
    IndicatorLöhner (equivalent to IndicatorLoehner)

AMR indicator adapted from a FEM indicator by Löhner (1987),
also used in the FLASH code as standard AMR indicator.
The indicator estimates a weighted second derivative of a
specified variable locally.
- Löhner (1987)
  "An adaptive finite element scheme for transient problems in CFD"
  [doi: 10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825(87)90098-3)
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node59.html#SECTION05163100000000000000
"""
struct IndicatorLöhner{RealT<:Real, Variable}
  f_wave::RealT
  variable::Variable
end

const IndicatorLoehner = IndicatorLöhner


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   print(io, "AMRCallback")
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  @unpack interval, indicator, threshold_refinement, threshold_coarsening = amr_callback
  println(io, "AMRCallback with")
  println(io, "- interval:  ", interval)
  println(io, "- indicator: ", indicator)
  println(io, "- threshold_refinement: ", indicator)
  print(io,   "- threshold_coarsening: ", indicator)
end


function AMRCallback(indicator; interval=5,
                                alpha_max=1.0, alpha_min=0.0, alpha_smooth=false)
  condition = (u, t, integrator) -> interval > 0 && (integrator.iter % interval == 0)

  # TODO: Taal, implement
  error("TODO")
  amr_callback = AMRCallback(0.0)

  DiscreteCallback(condition, amr_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AMRCallback}
  reset_timer!(timer())
  amr_callback = cb.affect!
  # TODO: Taal, implement
  return nothing
end


function (amr_callback::AMRCallback)(integrator)
  # TODO: Taal, implement

  return nothing
end
