
abstract type AbstractVolumeIntegral end

get_element_variables!(element_variables, u, mesh, equations,
                       volume_integral::AbstractVolumeIntegral, dg, cache) = nothing

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

function Base.show(io::IO, ::MIME"text/plain", integral::VolumeIntegralFluxDifferencing)
  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "volume flux" => integral.volume_flux
            ]
    summary_box(io, "VolumeIntegralFluxDifferencing", setup)
  end
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

function Base.show(io::IO, mime::MIME"text/plain", integral::VolumeIntegralShockCapturingHG)
  if get(io, :compact, false)
    show(io, integral)
  else
    summary_header(io, "VolumeIntegralFluxDifferencing")
    summary_line(io, "volume flux DG", integral.volume_flux_dg)
    summary_line(io, "volume flux FV", integral.volume_flux_dg)
    summary_line(io, "indicator", typeof(integral.indicator).name)
    show(increment_indent(io), mime, integral.indicator)
    summary_footer(io)
  end
end

function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralShockCapturingHG, dg, cache)
  # call the indicator to get up-to-date values for IO
  volume_integral.indicator(u, equations, dg, cache)
  get_element_variables!(element_variables, volume_integral.indicator, volume_integral)
end



"""
    DG(basis, basis, surface_flux, volume_integral)

Create a discontinuous Galerkin method.
If [`basis isa LobattoLegendreBasis`](@ref LobattoLegendreBasis),
this creates a [`DGSEM`](@ref).
"""
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

function Base.show(io::IO, mime::MIME"text/plain", dg::DG{RealT}) where {RealT}
  if get(io, :compact, false)
    show(io, dg)
  else
    summary_header(io, "DG{" * string(RealT) * "}")
    summary_line(io, "polynomial degree", polydeg(dg))
    summary_line(io, "basis", dg.basis)
    summary_line(io, "mortar", dg.mortar)
    summary_line(io, "surface flux", dg.surface_flux)
    summary_line(io, "volume integral", typeof(dg.volume_integral).name)
    if !(dg.volume_integral isa VolumeIntegralWeakForm)
      show(increment_indent(io), mime, dg.volume_integral)
    end
    summary_footer(io)
  end
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
@inline eachmpiinterface(dg::DG, cache) = Base.OneTo(nmpiinterfaces(dg, cache))

@inline nnodes(dg::DG)             = nnodes(dg.basis)
@inline nelements(dg::DG, cache)   = nelements(cache.elements)
@inline nelementsglobal(dg::DG, cache) = mpi_isparallel() ? cache.mpi_cache.n_elements_global : nelements(dg, cache)
@inline ninterfaces(dg::DG, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(dg::DG, cache) = nboundaries(cache.boundaries)
@inline nmortars(dg::DG, cache)    = nmortars(cache.mortars)
@inline nmpiinterfaces(dg::DG, cache) = nmpiinterfaces(cache.mpi_interfaces)


@inline function get_node_coords(x, equations, solver::DG, indices...)
  SVector(ntuple(idx -> x[idx, indices...], ndims(equations)))
end

@inline function get_node_vars(u, equations, solver::DG, indices...)
  # There is a cut-off at `n == 10` inside of the method
  # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
  # in Julia `v1.5`, leading to type instabilities if
  # more than ten variables are used. That's why we use
  # `Val(...)` below.
  SVector(ntuple(v -> u[v, indices...], Val(nvariables(equations))))
end

@inline function get_surface_node_vars(u, equations, solver::DG, indices...)
  # There is a cut-off at `n == 10` inside of the method
  # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
  # in Julia `v1.5`, leading to type instabilities if
  # more than ten variables are used. That's why we use
  # `Val(...)` below.
  u_ll = SVector(ntuple(v -> u[1, v, indices...], Val(nvariables(equations))))
  u_rr = SVector(ntuple(v -> u[2, v, indices...], Val(nvariables(equations))))
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
SolutionAnalyzer(dg::DG; kwargs...) = SolutionAnalyzer(dg.basis; kwargs...)

AdaptorAMR(mesh, dg::DG) = AdaptorL2(dg.basis)



# Include utilities
include("interpolation.jl")
include("l2projection.jl")
include("basis_lobatto_legendre.jl")

"""
    DGSEM([RealT=Float64,] polydeg::Integer,
          surface_flux=flux_central,
          volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm(),
          mortar=MortarL2(basis))

Create a discontinuous Galerkin spectral element method (DGSEM) using a
[`LobattoLegendreBasis`](@ref) with polynomials of degree `polydeg`.
"""
const DGSEM = DG{RealT, Basis, Mortar, SurfaceFlux, VolumeIntegral} where {RealT<:Real, Basis<:LobattoLegendreBasis{RealT}, Mortar, SurfaceFlux, VolumeIntegral}

function DGSEM(basis::LobattoLegendreBasis,
               surface_flux=flux_central,
               volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm(),
               mortar=MortarL2(basis))

  return DG{real(basis), typeof(basis), typeof(mortar), typeof(surface_flux), typeof(volume_integral)}(
    basis, mortar, surface_flux, volume_integral)
end

function DGSEM(RealT, polydeg::Integer,
               surface_flux=flux_central,
               volume_integral::AbstractVolumeIntegral=VolumeIntegralWeakForm(),
               mortar=MortarL2(LobattoLegendreBasis(RealT, polydeg)))
  basis = LobattoLegendreBasis(RealT, polydeg)

  return DGSEM(basis, surface_flux, volume_integral, mortar)
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



# indicators used for shock-capturing and AMR
include("indicators.jl")
include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")

# 1D DG implementation
include("containers_1d.jl")
include("dg_1d.jl")

# 2D DG implementation
include("containers_2d.jl")
include("dg_2d.jl")
include("dg_2d_parallel.jl")

# 3D DG implementation
include("containers_3d.jl")
include("dg_3d.jl")
