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

create_cache(mesh, equations, ::VolumeIntegralWeakForm) = NamedTuple()

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
struct VolumeIntegralShockCapturingHG{VolumeFluxDG, VolumeFluxFV, ShockIndicatorVariable, RealT<:Real} <: AbstractVolumeIntegral
  volume_flux_dg::VolumeFluxDG # symmetric, e.g. split-form or entropy-conservative
  volume_flux_fv::VolumeFluxFV # non-symmetric in general, e.g.entropy-dissipative
  shock_indicator_variable::ShockIndicatorVariable
  shock_alpha_max::RealT
  shock_alpha_min::RealT
  shock_alpha_smooth::Bool
end


abstract type AbstractBasisSBP{RealT<:Real} end

abstract type AbstractMortar{RealT<:Real} end

# TODO: Taal refactor, weird inheritance to use get_node_vars etc.
struct DG{RealT, Basis<:AbstractBasisSBP{RealT}, Mortar, SurfaceFlux, VolumeIntegral} <: AbstractDg{0,0}
  basis::Basis
  mortar::Mortar
  surface_flux::SurfaceFlux
  volume_integral::VolumeIntegral
end

# TODO: Taal bikeshedding, implement a method with reduced information and the signature
# function Base.show(io::IO, dg::DG{RealT}) where {RealT}
function Base.show(io::IO, ::MIME"text/plain", dg::DG{RealT}) where {RealT}
  println(io, "DG{", RealT, "} using")
  println(io, "- ", dg.basis)
  println(io, "- ", dg.mortar)
  println(io, "- ", dg.surface_flux)
  println(io, "- ", dg.volume_integral)
end

@inline Base.real(dg::DG{RealT}) where {RealT} = RealT

@inline nnodes(dg::DG) = nnodes(dg.basis)

# TODO: Taal refactor, use case?
# Deprecate in favor of nnodes or order_of_accuracy?
@inline polydeg(dg::DG) = polydeg(dg.basis)

@inline ndofs(mesh::TreeMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)



# TODO: Taal refactor, where to put this?
@inline eachvariable(equations::AbstractEquations) = Base.OneTo(nvariables(equations))

# TODO: Taal performance, 1:nnodes(dg) vs. Base.OneTo(nnodes(dg)) vs. SOneTo(nnodes(dg))
@inline eachnode(dg::DG)             = Base.OneTo(nnodes(dg))
@inline eachelement(dg::DG, cache)   = Base.OneTo(nelements(cache.elements))
@inline eachinterface(dg::DG, cache) = Base.OneTo(ninterfaces(cache.interfaces))
@inline eachboundary(dg::DG, cache)  = Base.OneTo(nboundaries(cache.boundaries))
@inline eachmortar(dg::DG, cache)    = Base.OneTo(nmortars(cache.mortars))


# Include utilities
include("interpolation.jl")
include("l2projection.jl")
include("lobatto_legendre.jl") # TODO: Taal new

const DGSEM = DG{RealT, Basis, Mortar, SurfaceFlux, VolumeIntegral} where {RealT<:Real, Basis<:LobattoLegendreBasis{RealT}, Mortar, SurfaceFlux, VolumeIntegral}

function DGSEM(RealT, polydeg::Integer, surface_flux=flux_central, volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm())
  basis = LobattoLegendreBasis(RealT, polydeg)
  mortar = MortarL2(basis)

  return DG{RealT, typeof(basis), typeof(mortar), typeof(surface_flux), typeof(volume_integral)}(
    basis, mortar, surface_flux, volume_integral
  )
end

DGSEM(polydeg, surface_flux=flux_central, volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm()) = DGSEM(Float64, polydeg, surface_flux, volume_integral)

# Include 2D implementation
include("2d/containers.jl")
include("2d/dg.jl")
include("2d/amr.jl")
include("dg_2d.jl") # TODO: Taal new

# Include 3D implementation
include("3d/containers.jl")
include("3d/dg.jl")
include("3d/amr.jl")
