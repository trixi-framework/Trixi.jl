
struct AMRCallback{Indicator, Adaptor}
  indicator::Indicator
  interval::Int
  adapt_initial_conditions_only_refine::Bool
  adaptor::Adaptor
end


function AMRCallback(semi, indicator, adaptor; interval=5,
                                               adapt_initial_conditions_only_refine=true)
  # AMR every `interval` time steps
  condition = (u, t, integrator) -> interval > 0 && (integrator.iter % interval == 0)

  amr_callback = AMRCallback{typeof(indicator), typeof(adaptor)}(
                  indicator, interval, adapt_initial_conditions_only_refine, adaptor)

  DiscreteCallback(condition, amr_callback,
                   save_positions=(false,false))
end

function AMRCallback(semi, indicator; kwargs...)
  adaptor = AdaptorAMR(semi)
  AMRCallback(semi, indicator, adaptor; kwargs...)
end

function AdaptorAMR(semi; kwargs...)
  mesh, _, solver, _ = mesh_equations_solver_cache(semi)
  AdaptorAMR(mesh, solver; kwargs...)
end


# TODO: Taal bikeshedding, implement a method with less information and the signature
# function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   print(io, "AMRCallback")
# end
function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  @unpack interval, indicator, adapt_initial_conditions_only_refine = amr_callback
  println(io, "AMRCallback with")
  println(io, "- indicator: ", indicator)
  println(io, "- interval: ", interval)
  print(io,   "- adapt_initial_conditions_only_refine: ", adapt_initial_conditions_only_refine)
end


function (cb::DiscreteCallback{Condition,Affect!})(ode::ODEProblem) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  semi = ode.p

  @timeit_debug timer() "initial condition AMR" begin
    # iterate until mesh does not change anymore
    has_changed = true
    while has_changed
      has_changed = amr_callback(ode.u0, semi,
                                 only_refine=amr_callback.adapt_initial_conditions_only_refine)
      ode.u0 .= compute_coefficients(ode.tspan[1], semi)
    end
  end

  return nothing
end


function (amr_callback::AMRCallback)(integrator; kwargs...)
  @unpack u = integrator
  semi = integrator.p

  println("AMRCallback: active") # TODO: Taal debug
  has_changed = amr_callback(u, semi; kwargs...)
  println("AMRCallback: has_changed=", has_changed) # TODO: Taal debug
  resize!(integrator, length(u))

  u_modified!(integrator, has_changed)
  return has_changed
end


@inline function (amr_callback::AMRCallback)(u::AbstractVector, semi::SemidiscretizationHyperbolic; kwargs...)
  amr_callback(u, mesh_equations_solver_cache(semi)...; kwargs...)
end


# TODO: Taal refactor, decide where to move this and implement everything
# """
#     IndicatorHennemannGassner

# Indicator used for shock-capturing or AMR used by
# - Hennemann, Gassner (2020)
#   "A provably entropy stable subcell shock capturing approach for high order split form DG"
#   [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
# """
# struct IndicatorHennemannGassner{RealT<:Real}
#   alpha_max::RealT
#   alpha_min::RealT
#   alpha_smooth::Bool
# end


# TODO: Taal document
struct IndicatorTwoLevel{RealT<:Real, Indicator, Cache}
  base_level::Int
  base_threshold::RealT
  max_level::Int
  max_threshold::RealT
  indicator::Indicator
  cache::Cache
end

function IndicatorTwoLevel(semi, indicator; base_level=1, base_threshold=0.0,
                                            max_level =1, max_threshold =1.0)
  base_threshold, max_threshold = promote(base_threshold, max_threshold)
  cache = indicator_cache(semi)
  IndicatorTwoLevel{typeof(base_threshold), typeof(indicator), typeof(cache)}(
    base_level, base_threshold, max_level, max_threshold, indicator, cache
  )
end

indicator_cache(semi) = indicator_cache(mesh_equations_solver_cache(semi)...)


function Base.show(io::IO, indicator::IndicatorTwoLevel)
  print(io, "IndicatorTwoLevel(")
  print(io, indicator.indicator)
  print(io, ", base_level=", indicator.base_level)
  print(io, ", base_threshold=", indicator.base_threshold)
  print(io, ", max_level=", indicator.max_level)
  print(io, ", max_threshold=", indicator.max_threshold)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorTwoLevel)
  println(io, "IndicatorTwoLevel with")
  println(io, "- ", indicator.indicator)
  println(io, "- base_level:     ", indicator.base_level)
  println(io, "- base_threshold: ", indicator.base_threshold)
  println(io, "- max_level:      ", indicator.max_level)
  print(io,   "- max_threshold:  ", indicator.max_threshold)
end


"""
    IndicatorLöhner (equivalent to IndicatorLoehner)

AMR indicator adapted from a FEM indicator by Löhner (1987), also used in the
FLASH code as standard AMR indicator.
The indicator estimates a weighted second derivative of a specified variable locally.
- Löhner (1987)
  "An adaptive finite element scheme for transient problems in CFD"
  [doi: 10.1016/0045-7825(87)90098-3](https://doi.org/10.1016/0045-7825(87)90098-3)
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node59.html#SECTION05163100000000000000
"""
struct IndicatorLöhner{RealT<:Real, Variable, Cache}
  f_wave::RealT # TODO: Taal, better name and documentation
  variable::Variable
  cache::Cache
end

function IndicatorLöhner(semi; f_wave=0.2, variable=first)
  cache = löhner_cache(semi)
  return IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end

löhner_cache(semi) = löhner_cache(mesh_equations_solver_cache(semi)...)

function Base.show(io::IO, indicator::IndicatorLöhner)
  print(io, "IndicatorLöhner(")
  print(io, "f_wave=", indicator.f_wave, ", variable=", indicator.variable, ")")
end
# TODO: Taal bikeshedding, implement a method with extended information and the signature
# function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorLöhner)
#   println(io, "IndicatorLöhner with")
#   println(io, "- indicator: ", indicator.indicator)
# end

const IndicatorLoehner = IndicatorLöhner


Base.first(u, equations::AbstractEquations) = first(u)


include("amr_dg2d.jl")
