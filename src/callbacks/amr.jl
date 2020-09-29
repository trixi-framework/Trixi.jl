
struct AMRCallback{Indicator, Adaptor}
  indicator::Indicator
  interval::Int
  adapt_initial_conditions::Bool
  adapt_initial_conditions_only_refine::Bool
  adaptor::Adaptor
end


function AMRCallback(semi, indicator, adaptor; interval=5,
                                               adapt_initial_conditions=true,
                                               adapt_initial_conditions_only_refine=true)
  # AMR every `interval` time steps
  condition = (u, t, integrator) -> interval > 0 && (integrator.iter % interval == 0)

  amr_callback = AMRCallback{typeof(indicator), typeof(adaptor)}(
                  indicator, interval, adapt_initial_conditions,
                  adapt_initial_conditions_only_refine, adaptor)

  DiscreteCallback(condition, amr_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
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
  println(io, "AMRCallback with")
  println(io, "- indicator: ", amr_callback.indicator)
  println(io, "- interval: ", amr_callback.interval)
  println(io, "- adapt_initial_conditions: ", amr_callback.adapt_initial_conditions)
  print(io,   "- adapt_initial_conditions_only_refine: ", amr_callback.adapt_initial_conditions_only_refine)
end


function get_element_variables!(element_variables, u, mesh, equations, solver, cache, amr_callback::AMRCallback)
  get_element_variables!(element_variables, u, mesh, equations, solver, cache, amr_callback.indicator, amr_callback)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:AMRCallback}
  amr_callback = cb.affect!
  semi = integrator.p

  @timeit_debug timer() "initial condition AMR" begin
    # iterate until mesh does not change anymore
    has_changed = true
    while has_changed
      has_changed = amr_callback(integrator,
                                 only_refine=amr_callback.adapt_initial_conditions_only_refine)
      integrator.u .= compute_coefficients(t, semi)
    end
  end

  return nothing
end


# TODO: Taal remove?
# function (cb::DiscreteCallback{Condition,Affect!})(ode::ODEProblem) where {Condition, Affect!<:AMRCallback}
#   amr_callback = cb.affect!
#   semi = ode.p

#   @timeit_debug timer() "strange" sleep(0.5)

#   @timeit_debug timer() "initial condition AMR" begin
#     # iterate until mesh does not change anymore
#     has_changed = true
#     while has_changed
#       has_changed = amr_callback(ode.u0, semi,
#                                  only_refine=amr_callback.adapt_initial_conditions_only_refine)
#       ode.u0 .= compute_coefficients(ode.tspan[1], semi)
#     end
#   end

#   return nothing
# end


function (amr_callback::AMRCallback)(integrator; kwargs...)
  u_ode = integrator.u
  semi = integrator.p

  @timeit_debug timer() "AMR" begin
    has_changed = amr_callback(u_ode, semi; kwargs...)
    resize!(integrator, length(u_ode))
  end

  u_modified!(integrator, has_changed)
  return has_changed
end


@inline function (amr_callback::AMRCallback)(u_ode::AbstractVector, semi::SemidiscretizationHyperbolic; kwargs...)
  amr_callback(u_ode, mesh_equations_solver_cache(semi)...; kwargs...)
end


# TODO: Taal document
struct IndicatorThreeLevel{RealT<:Real, Indicator, Cache}
  base_level::Int
  med_level ::Int
  max_level ::Int
  med_threshold::RealT
  max_threshold::RealT
  indicator::Indicator
  cache::Cache
end

function IndicatorThreeLevel(semi, indicator; base_level=1,
                                              med_level=base_level, med_threshold=0.0,
                                              max_level=base_level, max_threshold=1.0)
  med_threshold, max_threshold = promote(med_threshold, max_threshold)
  cache = indicator_cache(semi)
  IndicatorThreeLevel{typeof(max_threshold), typeof(indicator), typeof(cache)}(
    base_level, med_level, max_level, med_threshold, max_threshold, indicator, cache)
end

indicator_cache(semi) = indicator_cache(mesh_equations_solver_cache(semi)...)


function Base.show(io::IO, indicator::IndicatorThreeLevel)
  print(io, "IndicatorThreeLevel(")
  print(io, indicator.indicator)
  print(io, ", base_level=", indicator.base_level)
  print(io, ", med_level=",  indicator.med_level)
  print(io, ", max_level=",  indicator.max_level)
  print(io, ", med_threshold=", indicator.med_threshold)
  print(io, ", max_threshold=", indicator.max_threshold)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorThreeLevel)
  println(io, "IndicatorThreeLevel with")
  println(io, "- ", indicator.indicator)
  println(io, "- base_level: ", indicator.base_level)
  println(io, "- med_level:  ", indicator.med_level)
  println(io, "- max_level:  ", indicator.max_level)
  println(io, "- med_threshold: ", indicator.med_threshold)
  print(io,   "- max_threshold: ", indicator.max_threshold)
end


function get_element_variables!(element_variables, u, mesh, equations, solver, cache, indicator::IndicatorThreeLevel, amr_callback::AMRCallback)
  # call the indicator to get up-to-date values for IO
  indicator.indicator(u, equations, solver, cache)
  get_element_variables!(element_variables, indicator.indicator, amr_callback)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::AMRCallback)
  element_variables[:indicator_amr] = indicator.cache.alpha
  return nothing
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
struct IndicatorLöhner{RealT<:Real, Variable, Cache} <: AbstractIndicator
  f_wave::RealT # TODO: Taal, better name and documentation
  variable::Variable
  cache::Cache
end

function IndicatorLöhner(basis, equations; f_wave=0.2, variable=first)
  cache = create_cache(IndicatorLöhner, equations, basis)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end

function IndicatorLöhner(semi::AbstractSemidiscretization; f_wave=0.2, variable=first)
  cache = create_cache(IndicatorLöhner, semi)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end


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


# TODO: Taal decide, shall we keep this?
struct IndicatorMax{Variable, Cache<:NamedTuple} <: AbstractIndicator
  variable::Variable
  cache::Cache
end

function IndicatorMax(basis, equations::AbstractEquations; variable=first)
  cache = create_cache(IndicatorMax, equations, basis)
  IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

function IndicatorMax(semi; variable=first)
  cache = create_cache(IndicatorMax, semi)
  return IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

function Base.show(io::IO, indicator::IndicatorMax)
  print(io, "IndicatorMax(")
  print(io, "variable=", indicator.variable, ")")
end
# TODO: Taal bikeshedding, implement a method with extended information and the signature
# function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorMax)
#   println(io, "IndicatorMax with")
#   println(io, "- indicator: ", indicator.indicator)
# end


Base.first(u, equations::AbstractEquations) = first(u)


include("amr_dg2d.jl")
