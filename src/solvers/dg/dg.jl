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



abstract type AbstractIndicator end

function create_cache!(element_variables, typ::Type{IndicatorType}, semi) where {IndicatorType<:AbstractIndicator}
  create_cache!(element_variables, typ, mesh_equations_solver_cache(semi)...)
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

function IndicatorHennemannGassner(equations, basis;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   variable=first)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache(IndicatorHennemannGassner, equations, basis)
  IndicatorHennemannGassner{typeof(alpha_max), typeof(variable), typeof(cache)}(
    alpha_max, alpha_min, alpha_smooth, variable, cache)
end

function IndicatorHennemannGassner(semi;
                                   alpha_max=0.5,
                                   alpha_min=0.001,
                                   alpha_smooth=true,
                                   variable=first)
  alpha_max, alpha_min = promote(alpha_max, alpha_min)
  cache = create_cache!(semi.cache.element_variables, IndicatorHennemannGassner, semi)
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


abstract type AbstractBasisSBP{RealT<:Real} end

abstract type AbstractMortar{RealT<:Real} end

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



# TODO: Taal refactor, where to put this?
@inline eachvariable(equations::AbstractEquations) = Base.OneTo(nvariables(equations))

# TODO: Taal performance, 1:nnodes(dg) vs. Base.OneTo(nnodes(dg)) vs. SOneTo(nnodes(dg))
@inline eachnode(dg::DG)             = Base.OneTo(nnodes(dg))
@inline eachelement(dg::DG, cache)   = Base.OneTo(nelements(dg, cache))
@inline eachinterface(dg::DG, cache) = Base.OneTo(ninterfaces(dg, cache))
@inline eachboundary(dg::DG, cache)  = Base.OneTo(nboundaries(dg, cache))
@inline eachmortar(dg::DG, cache)    = Base.OneTo(nmortars(dg, cache))

@inline nnodes(dg::DG) = nnodes(dg.basis)
@inline nelements(dg::DG, cache)   = nelements(cache.elements)
@inline ninterfaces(dg::DG, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(dg::DG, cache) = nboundaries(cache.boundaries)
@inline nmortars(dg::DG, cache)    = nmortars(cache.mortars)


@inline get_node_coords(x, equations, solver::DG, indices...) = SVector(ntuple(idx -> x[idx, indices...], ndims(equations)))

@inline get_node_vars(u, equations, solver::DG, indices...) = SVector(ntuple(v -> u[v, indices...], nvariables(equations)))

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


# Include utilities
include("interpolation.jl")
include("l2projection.jl")
include("lobatto_legendre.jl") # TODO: Taal new

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


SolutionAnalyzer(dg::DG; kwargs...) = SolutionAnalyzer(dg.basis; kwargs...)

AdaptorAMR(mesh, dg::DG) = AdaptorL2(dg.basis)


# Include 2D implementation
include("2d/containers.jl")
include("2d/dg.jl")
include("2d/amr.jl")
include("dg_2d.jl") # TODO: Taal new

# Include 3D implementation
include("3d/containers.jl")
include("3d/dg.jl")
include("3d/amr.jl")
