# Abstract supertype for DG-type solvers
# `POLYDEG` corresponds to `N` in the school of Kopriva
abstract type AbstractDg{NDIMS, POLYDEG} <: AbstractSolver{NDIMS} end

@inline Base.ndims(dg::AbstractDg) = ndims(equations(dg))

# Return polynomial degree for a DG solver
@inline polydeg(::AbstractDg{NDIMS, POLYDEG}) where {NDIMS, POLYDEG} = POLYDEG

# Return number of nodes in one direction
@inline nnodes(::AbstractDg{NDIMS, POLYDEG}) where {NDIMS, POLYDEG} = POLYDEG + 1

# Return system of equations instance for a DG solver
@inline equations(dg::AbstractDg) = dg.equations

# Return number of variables for the system of equations in use
@inline nvariables(dg::AbstractDg) = nvariables(equations(dg))

# Return number of degrees of freedom
@inline ndofs(dg::AbstractDg) = dg.n_elements * nnodes(dg)^ndims(dg)

"""
    get_node_coords(x, dg::AbstractDg, indices...)

Return an `ndims(dg)`-dimensional `SVector` for the DG node specified via the `i, j, k, element_id` indices (3D) or `i, j, element_id` indices (2D).
"""
@inline get_node_coords(x, dg::AbstractDg, indices...) = SVector(ntuple(idx -> x[idx, indices...], ndims(dg)))

"""
    get_node_vars(u, dg::AbstractDg, indices...)

Return an `nvariables(dg)`-dimensional `SVector` of the conservative variables for the DG node specified via the `i, j, k, element_id` indices (3D) or `i, j, element_id` indices (2D).
"""
@inline get_node_vars(u, dg::AbstractDg, indices...) = SVector(ntuple(v -> u[v, indices...], nvariables(dg)))

@inline function get_surface_node_vars(u, dg::AbstractDg, indices...)
  u_ll = SVector(ntuple(v -> u[1, v, indices...], nvariables(dg)))
  u_rr = SVector(ntuple(v -> u[2, v, indices...], nvariables(dg)))
  return u_ll, u_rr
end

@inline function set_node_vars!(u, u_node, ::AbstractDg, indices...)
  for v in eachindex(u_node)
    u[v, indices...] = u_node[v]
  end
  return nothing
end

@inline function add_to_node_vars!(u, u_node, ::AbstractDg, indices...)
  for v in eachindex(u_node)
    u[v, indices...] += u_node[v]
  end
  return nothing
end




abstract type AbstractVolumeIntegral end

get_element_variables!(element_variables, u, mesh, equations, volume_integral::AbstractVolumeIntegral, dg, cache) = nothing

"""
    VolumeIntegralWeakForm

The classical weak form volume integral type for DG methods as explained in standard
textbooks such as
- Kopriva (2009)
  Implementing Spectral Methods for Partial Differential Equations:
  Algorithms for Scientists and Engineers
  [doi: 10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
"""
struct VolumeIntegralWeakForm <: AbstractVolumeIntegral end

create_cache(mesh, equations, ::VolumeIntegralWeakForm, dg) = NamedTuple()

"""
    VolumeIntegralFluxDifferencing

Volume integral type for DG methods based on SBP operators and flux differencing using
symmetric two-point volume fluxes. Based upon the theory developed by
- LeFloch, Mercier, Rohde (2002)
  Fully Discrete, Entropy Conservative Schemes of Arbitrary Order
  [doi: 10.1137/S003614290240069X](https://doi.org/10.1137/S003614290240069X)
- Fisher, Carpenter (2013)
  High-order entropy stable finite difference schemes for nonlinear
  conservation laws: Finite domains
  [doi: 10.1016/j.jcp.2013.06.014](https://doi.org/10.1016/j.jcp.2013.06.014)
- Ranocha (2017)
  Comparison of Some Entropy Conservative Numerical Fluxes for the Euler Equations
  [arXiv: 1701.02264](https://arxiv.org/abs/1701.02264)
  [doi: 10.1007/s10915-017-0618-1](https://doi.org/10.1007/s10915-017-0618-1)
- Chen, Shu (2017)
  Entropy stable high order discontinuous Galerkin methods with suitable
  quadrature rules for hyperbolic conservation laws
  [doi: 10.1016/j.jcp.2017.05.025](https://doi.org/10.1016/j.jcp.2017.05.025)
"""
struct VolumeIntegralFluxDifferencing{VolumeFlux} <: AbstractVolumeIntegral
  volume_flux::VolumeFlux
end

"""
    VolumeIntegralShockCapturingHG

Shock-capturing volume integral type for DG methods proposed by
- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
struct VolumeIntegralShockCapturingHG{VolumeFluxDG, VolumeFluxFV, Indicator} <: AbstractVolumeIntegral
  volume_flux_dg::VolumeFluxDG # symmetric, e.g. split-form or entropy-conservative
  volume_flux_fv::VolumeFluxFV # non-symmetric in general, e.g. entropy-dissipative
  indicator::Indicator
end

function VolumeIntegralShockCapturingHG(indicator; volume_flux_dg=flux_central,
                                                   volume_flux_fv=flux_lax_friedrichs)
  VolumeIntegralShockCapturingHG{typeof(volume_flux_dg), typeof(volume_flux_fv), typeof(indicator)}(
    volume_flux_dg, volume_flux_fv, indicator)
end


function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralShockCapturingHG, dg, cache)
  # call the indicator to get up-to-date values for IO
  volume_integral.indicator(u, equations, dg, cache)
  get_element_variables!(element_variables, volume_integral.indicator, volume_integral)
end



abstract type AbstractBasisSBP{RealT<:Real} end

abstract type AbstractMortar{RealT<:Real} end

abstract type MortarL2{RealT<:Real} <: AbstractMortar{RealT} end


struct DG{RealT, Basis<:AbstractBasisSBP{RealT}, Mortar, SurfaceFlux, VolumeIntegral}
  basis::Basis
  mortar::Mortar
  surface_flux::SurfaceFlux
  volume_integral::VolumeIntegral
end

function Base.show(io::IO, dg::DG{RealT}) where {RealT}
  print(io, "DG{", RealT, "}(")
  print(io,       dg.basis)
  print(io, ", ", dg.mortar)
  print(io, ", ", dg.surface_flux)
  print(io, ", ", dg.volume_integral)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", dg::DG{RealT}) where {RealT}
  println(io, "DG{", RealT, "} using")
  println(io, "- ", dg.basis)
  println(io, "- ", dg.mortar)
  println(io, "- ", dg.surface_flux)
  print(io,   "- ", dg.volume_integral)
end

@inline Base.real(dg::DG{RealT}) where {RealT} = RealT

# TODO: Taal refactor, use case?
# Deprecate in favor of nnodes or order_of_accuracy?
@inline polydeg(dg::DG) = polydeg(dg.basis)

@inline ndofs(mesh::TreeMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


function get_element_variables!(element_variables, u, mesh, equations, dg::DG, cache)
  get_element_variables!(element_variables, u, mesh, equations, dg.volume_integral, dg, cache)
end


# TODO: Taal performance, 1:nnodes(dg) vs. Base.OneTo(nnodes(dg)) vs. SOneTo(nnodes(dg))
@inline eachnode(dg::DG)             = Base.OneTo(nnodes(dg))
@inline eachelement(dg::DG, cache)   = Base.OneTo(nelements(dg, cache))
@inline eachinterface(dg::DG, cache) = Base.OneTo(ninterfaces(dg, cache))
@inline eachboundary(dg::DG, cache)  = Base.OneTo(nboundaries(dg, cache))
@inline eachmortar(dg::DG, cache)    = Base.OneTo(nmortars(dg, cache))

@inline nnodes(dg::DG)             = nnodes(dg.basis)
@inline nelements(dg::DG, cache)   = nelements(cache.elements)
@inline ninterfaces(dg::DG, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(dg::DG, cache) = nboundaries(cache.boundaries)
@inline nmortars(dg::DG, cache)    = nmortars(cache.mortars)


@inline function get_node_coords(x, equations, solver::DG, indices...)
  SVector(ntuple(idx -> x[idx, indices...], ndims(equations)))
end

@inline function get_node_vars(u, equations, solver::DG, indices...)
  SVector(ntuple(v -> u[v, indices...], nvariables(equations)))
end

@inline function get_surface_node_vars(u, equations, solver::DG, indices...)
  u_ll = SVector(ntuple(v -> u[1, v, indices...], nvariables(equations)))
  u_rr = SVector(ntuple(v -> u[2, v, indices...], nvariables(equations)))
  return u_ll, u_rr
end

@inline function set_node_vars!(u, u_node, equations, solver::DG, indices...)
  for v in eachvariable(equations)
    u[v, indices...] = u_node[v]
  end
  return nothing
end

@inline function add_to_node_vars!(u, u_node, equations, solver::DG, indices...)
  for v in eachvariable(equations)
    u[v, indices...] += u_node[v]
  end
  return nothing
end


function allocate_coefficients(mesh::TreeMesh, equations, dg::DG, cache)
  # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
  # cf. wrap_array
  zeros(real(dg), nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
end



# Used for analyze_solution
abstract type SolutionAnalyzer{RealT<:Real} end

abstract type AdaptorAMR{RealT<:Real} end

abstract type AdaptorL2{RealT<:Real} <: AdaptorAMR{RealT} end

SolutionAnalyzer(dg::DG; kwargs...) = SolutionAnalyzer(dg.basis; kwargs...)

AdaptorAMR(mesh, dg::DG) = AdaptorL2(dg.basis)



# Include utilities
include("interpolation.jl")
include("l2projection.jl")
include("lobatto_legendre.jl")

"""
    DGSEM([RealT=Float64,] polydeg::Integer,
          surface_flux=flux_central,
          volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm())

Create a discontinuous Galerkin spectral element method (DGSEM) using a
[`LobattoLegendreBasis`](@ref) with polynomials of degree `polydeg`.
"""
const DGSEM = DG{RealT, Basis, Mortar, SurfaceFlux, VolumeIntegral} where {RealT<:Real, Basis<:LobattoLegendreBasis{RealT}, Mortar, SurfaceFlux, VolumeIntegral}

function DGSEM(basis::LobattoLegendreBasis,
               surface_flux=flux_central,
               volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm())
  mortar = MortarL2(basis)

  return DG{real(basis), typeof(basis), typeof(mortar), typeof(surface_flux), typeof(volume_integral)}(
    basis, mortar, surface_flux, volume_integral)
end

function DGSEM(RealT, polydeg::Integer,
               surface_flux=flux_central,
               volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm())
  basis = LobattoLegendreBasis(RealT, polydeg)

  return DGSEM(basis, surface_flux, volume_integral)
end

DGSEM(polydeg, surface_flux=flux_central, volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm()) = DGSEM(Float64, polydeg, surface_flux, volume_integral)



"""
    pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg, cache)

Given blending factors `alpha` and the solver `dg`, fill
`element_ids_dg` with the IDs of elements using a pure DG scheme and
`element_ids_dgfv` with the IDs of elements using a blended DG-FV scheme.
"""
function pure_and_blended_element_ids!(element_ids_dg, element_ids_dgfv, alpha, dg::DG, cache)
  empty!(element_ids_dg)
  empty!(element_ids_dgfv)

  for element in eachelement(dg, cache)
    # Clip blending factor for values close to zero (-> pure DG)
    dg_only = isapprox(alpha[element], 0, atol=1e-12)
    if dg_only
      push!(element_ids_dg, element)
    else
      push!(element_ids_dgfv, element)
    end
  end

  return nothing
end



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
                                   variable=first)
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
                                   variable=first)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorHennemannGassner, semi)
  IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, variable, cache)
end


function Base.show(io::IO, indicator::IndicatorHennemannGassner)
  print(io, "IndicatorHennemannGassner(")
  print(io, indicator.variable)
  print(io, ", alpha_max=", indicator.alpha_max)
  print(io, ", alpha_min=", indicator.alpha_min)
  print(io, ", alpha_smooth=", indicator.alpha_smooth)
  print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", indicator::IndicatorHennemannGassner)
  println(io, "IndicatorHennemannGassner(")
  println(io, "- ", indicator.variable)
  println(io, "- alpha_max:    ", indicator.alpha_max)
  println(io, "- alpha_min:    ", indicator.alpha_min)
  println(io, "- alpha_smooth: ", indicator.alpha_smooth)
  print(io,   "- cache with fields:")
  for key in keys(indicator.cache)
    print(io, " ", key)
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
  f_wave::RealT # TODO: Taal, better name and documentation
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorLöhner(equations::AbstractEquations, basis;
                         f_wave=0.2, variable=first)
  cache = create_cache(IndicatorLöhner, equations, basis)
  IndicatorLöhner{typeof(f_wave), typeof(variable), typeof(cache)}(f_wave, variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorLöhner(semi::AbstractSemidiscretization;
                         f_wave=0.2, variable=first)
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

# TODO: Taal dimension agnostic
# dirty Löhner estimate, direction by direction, assuming constant nodes
@inline function (löhner::IndicatorLöhner)(um::Real, u0::Real, up::Real)
  num = abs(up - 2 * u0 + um)
  den = abs(up - u0) + abs(u0-um) + löhner.f_wave * (abs(up) + 2 * abs(u0) + abs(um))
  return num / den
end



# TODO: Taal decide, shall we keep this?
struct IndicatorMax{Variable, Cache<:NamedTuple} <: AbstractIndicator
  variable::Variable
  cache::Cache
end

# this method is used when the indicator is constructed as for shock-capturing volume integrals
function IndicatorMax(equations::AbstractEquations, basis;
                      variable=first)
  cache = create_cache(IndicatorMax, equations, basis)
  IndicatorMax{typeof(variable), typeof(cache)}(variable, cache)
end

# this method is used when the indicator is constructed as for AMR
function IndicatorMax(semi::AbstractSemidiscretization;
                      variable=first)
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



# Include 1D implementation
include("1d/containers.jl")
include("1d/dg.jl")
include("1d/amr.jl")
include("dg_1d.jl")
include("dg_1d_indicators.jl")

# Include 2D implementation
include("2d/containers.jl")
include("2d/dg.jl")
include("2d/amr.jl")
include("dg_2d.jl")
include("dg_2d_indicators.jl")

# Include 3D implementation
include("3d/containers.jl")
include("3d/dg.jl")
include("3d/amr.jl")
