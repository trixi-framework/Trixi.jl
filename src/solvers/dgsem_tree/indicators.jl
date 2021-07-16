# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


abstract type AbstractIndicator end

function create_cache(typ::Type{IndicatorType}, semi) where {IndicatorType<:AbstractIndicator}
  create_cache(typ, mesh_equations_solver_cache(semi)...)
end

function get_element_variables!(element_variables, indicator::AbstractIndicator, ::VolumeIntegralShockCapturingHG)
  element_variables[:indicator_shock_capturing] = indicator.cache.alpha
  return nothing
end



"""
    IndicatorHennemannGassner

Indicator used for shock-capturing or AMR used by
- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
struct IndicatorHennemannGassner{RealT<:Real, Variable, Cache} <: AbstractIndicator
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorHennemannGassner(equations::AbstractEquations, basis;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   variable)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorHennemannGassner, equations, basis)
  IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorHennemannGassner(semi::AbstractSemidiscretization;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   variable)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorHennemannGassner, semi)
  IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, variable, cache)
end


function Base.show(io::IO, indicator::IndicatorHennemannGassner)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorHennemannGassner(")
  print(io, indicator.variable)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorHennemannGassner)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "max. α" => indicator.alpha_max,
             "min. α" => indicator.alpha_min,
             "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
            ]
    summary_box(io, "IndicatorHennemannGassner", setup)
  end
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
  f_wave::RealT # TODO: Taal documentation
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorLöhner(equations::AbstractEquations, basis;
                         f_wave=0.2, variable)
  cache = create_cache(IndicatorLöhner, equations, basis)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorLöhner(semi::AbstractSemidiscretization;
                         f_wave=0.2, variable)
  cache = create_cache(IndicatorLöhner, semi)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end


function Base.show(io::IO, indicator::IndicatorLöhner)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorLöhner(")
  print(io, "f_wave=", indicator.f_wave, ", variable=", indicator.variable, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorLöhner)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "f_wave" => indicator.f_wave,
            ]
    summary_box(io, "IndicatorLöhner", setup)
  end
end

const IndicatorLoehner = IndicatorLöhner

# dirty Löhner estimate, direction by direction, assuming constant nodes
@inline function local_löhner_estimate(um::Real, u0::Real, up::Real, löhner::IndicatorLöhner)
  num = abs(up - 2 * u0 + um)
  den = abs(up - u0) + abs(u0-um) + löhner.f_wave * (abs(up) + 2 * abs(u0) + abs(um))
  return num / den
end



struct IndicatorMax{Variable, Cache<:NamedTuple} <: AbstractIndicator
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorMax(equations::AbstractEquations, basis;
                      variable)
  cache = create_cache(IndicatorMax, equations, basis)
  IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorMax(semi::AbstractSemidiscretization;
                      variable)
  cache = create_cache(IndicatorMax, semi)
  return IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end


function Base.show(io::IO, indicator::IndicatorMax)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorMax(")
  print(io, "variable=", indicator.variable, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorMax)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
            ]
    summary_box(io, "IndicatorMax", setup)
  end
end

"""
    IndicatorNNPP

Artificial neural network based indicator used for shock-capturing or AMR.

!!! warning "Experimental implementation"                                                           
This is an experimental feature and may change in future releases.

"""
struct IndicatorNNPP{RealT<:Real, Variable, Chain, Cache} <: AbstractIndicator
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  alpha_continuous::Bool
  alpha_amr::Bool
  variable::Variable
  network::Chain
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorNNPP(equations::AbstractEquations, basis;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   alpha_continuous=true,
                                   alpha_amr=false,
                                   variable,
                                   network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorNNPP, equations, basis)
  IndicatorNNPP{typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorNNPP(semi::AbstractSemidiscretization;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   alpha_continuous=true,
                                   alpha_amr=true,
                                   variable,
                                   network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorNNPP, semi)
  IndicatorNNPP{typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network, cache)
end


function Base.show(io::IO, indicator::IndicatorNNPP)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorNNPP(")
  print(io, indicator.variable)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ", alpha_continuous=", indicator.alpha_continuous)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorNNPP)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "max. α" => indicator.alpha_max,
             "min. α" => indicator.alpha_min,
             "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
             "continuous α" => (indicator.alpha_continuous ? "yes" : "no"),
            ]
    summary_box(io, "IndicatorNNPP", setup)
  end
end


"""
    IndicatorNNRH

Artificial neural network based indicator used for shock-capturing or AMR
  - Ray, Hesthaven (2018)
    "An artificial neural network as a troubled-cell indicator"
    [doi:10.1016/j.jcp.2018.04.029](https://doi.org/10.1016/j.jcp.2018.04.029)
  - Ray, Hesthaven (2019)
    "Detecting troubled-cells on two-dimensional unstructured grids using a neural network"
    [doi:10.1016/j.jcp.2019.07.043](https://doi.org/10.1016/j.jcp.2019.07.043)
"""
struct IndicatorNNRH{RealT<:Real, Variable, Chain, Cache} <: AbstractIndicator
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  alpha_continuous::Bool
  alpha_amr::Bool
  variable::Variable
  network::Chain
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorNNRH(equations::AbstractEquations, basis, mesh;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   alpha_continuous=true,
                                   alpha_amr=false,
                                   variable,
                                   network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorNNRH, equations, basis, mesh)
  IndicatorNNRH{typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorNNRH(semi::AbstractSemidiscretization;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   alpha_continuous=true,
                                   alpha_amr=true,
                                   variable,
                                   network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorNNRH, semi)
  IndicatorNNRH{typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network, cache)
end


function Base.show(io::IO, indicator::IndicatorNNRH)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorNNRH(")
  print(io, indicator.variable)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ", alpha_continuous=", indicator.alpha_continuous)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorNNRH)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "max. α" => indicator.alpha_max,
             "min. α" => indicator.alpha_min,
             "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
             "continuous α" => (indicator.alpha_continuous ? "yes" : "no"),
            ]
    summary_box(io, "IndicatorNNRH", setup)
  end
end

"""
    IndicatorCNN

Convolutional neural network based indicator used for shock-capturing or AMR
"""
struct IndicatorCNN{RealT<:Real, Variable, Chain, Cache} <: AbstractIndicator
  alpha_max::RealT
  alpha_min::RealT
  alpha_smooth::Bool
  alpha_continuous::Bool
  alpha_amr::Bool
  variable::Variable
  network::Chain
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorCNN(equations::AbstractEquations, basis;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   alpha_continuous=true,
                                   alpha_amr=false,
                                   variable,
                                   network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorNNPP, equations, basis)
  IndicatorCNN{typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorCNN(semi::AbstractSemidiscretization;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   alpha_continuous=true,
                                   alpha_amr=true,
                                   variable,
                                   network)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorCNN, semi)
  IndicatorCNN{typeof(alpha_max), typeof(variable), typeof(network), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, alpha_continuous, alpha_amr, variable, network, cache)
end


function Base.show(io::IO, indicator::IndicatorCNN)
  @nospecialize indicator # reduce precompilation time

  print(io, "IndicatorCNN(")
  print(io, indicator.variable)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ", alpha_continuous=", indicator.alpha_continuous)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorCNN)
  @nospecialize indicator # reduce precompilation time

  if get(io, :compact, false)
    show(io, indicator)
  else
    setup = [
             "indicator variable" => indicator.variable,
             "max. α" => indicator.alpha_max,
             "min. α" => indicator.alpha_min,
             "smooth α" => (indicator.alpha_smooth ? "yes" : "no"),
             "continuous α" => (indicator.alpha_continuous ? "yes" : "no"),
            ]
    summary_box(io, "IndicatorCNN", setup)
  end
end

end # @muladd
