# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


abstract type AbstractVolumeIntegral end

get_element_variables!(element_variables, u, mesh, equations,
                       volume_integral::AbstractVolumeIntegral, dg, cache) = nothing

"""
    VolumeIntegralStrongForm

The classical strong form volume integral type for FD/DG methods.
"""
struct VolumeIntegralStrongForm <: AbstractVolumeIntegral end

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

create_cache(mesh, equations, ::VolumeIntegralWeakForm, dg, uEltype) = NamedTuple()

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
  @nospecialize integral # reduce precompilation time

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
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    summary_header(io, "VolumeIntegralShockCapturingHG")
    summary_line(io, "volume flux DG", integral.volume_flux_dg)
    summary_line(io, "volume flux FV", integral.volume_flux_fv)
    summary_line(io, "indicator", integral.indicator |> typeof |> nameof)
    show(increment_indent(io), mime, integral.indicator)
    summary_footer(io)
  end
end


"""
    VolumeIntegralPureLGLFiniteVolume

A volume integral that only uses the subcell finite volume scheme from the paper
- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
This gives a formally O(1)-accurate finite volume scheme on an LGL-type subcell mesh (LGL = Legendre-Gauss-Lobatto).
!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct VolumeIntegralPureLGLFiniteVolume{VolumeFluxFV} <: AbstractVolumeIntegral
  volume_flux_fv::VolumeFluxFV # non-symmetric in general, e.g. entropy-dissipative
end
# TODO: Figure out if this can also be used for Gauss nodes, not just LGL, and adjust the name accordingly

function Base.show(io::IO, ::MIME"text/plain", integral::VolumeIntegralPureLGLFiniteVolume)
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "FV flux" => integral.volume_flux_fv
            ]
    summary_box(io, "VolumeIntegralPureLGLFiniteVolume", setup)
  end
end


function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralShockCapturingHG, dg, cache)
  # call the indicator to get up-to-date values for IO
  volume_integral.indicator(u, equations, dg, cache)
  get_element_variables!(element_variables, volume_integral.indicator, volume_integral)
end



abstract type AbstractSurfaceIntegral end

"""
    SurfaceIntegralWeakForm(surface_flux=flux_central)

The classical weak form surface integral type for DG methods as explained in standard
textbooks such as
- Kopriva (2009)
  Implementing Spectral Methods for Partial Differential Equations:
  Algorithms for Scientists and Engineers
  [doi: 10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)

See also [`VolumeIntegralWeakForm`](@ref).
"""
struct SurfaceIntegralWeakForm{SurfaceFlux} <: AbstractSurfaceIntegral
  surface_flux::SurfaceFlux
end

SurfaceIntegralWeakForm() = SurfaceIntegralWeakForm(flux_central)

function Base.show(io::IO, ::MIME"text/plain", integral::SurfaceIntegralWeakForm)
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "surface flux" => integral.surface_flux
            ]
    summary_box(io, "SurfaceIntegralWeakForm", setup)
  end
end

"""
    SurfaceIntegralStrongForm(surface_flux=flux_central)

The classical strong form surface integral type for FD/DG methods.

See also [`VolumeIntegralStrongForm`](@ref).
"""
struct SurfaceIntegralStrongForm{SurfaceFlux} <: AbstractSurfaceIntegral
  surface_flux::SurfaceFlux
end

SurfaceIntegralStrongForm() = SurfaceIntegralStrongForm(flux_central)

function Base.show(io::IO, ::MIME"text/plain", integral::SurfaceIntegralStrongForm)
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "surface flux" => integral.surface_flux
            ]
    summary_box(io, "SurfaceIntegralStrongForm", setup)
  end
end



"""
    DG(; basis, mortar, surface_integral, volume_integral)

Create a discontinuous Galerkin method.
If [`basis isa LobattoLegendreBasis`](@ref LobattoLegendreBasis),
this creates a [`DGSEM`](@ref).
"""
struct DG{Basis, Mortar, SurfaceIntegral, VolumeIntegral}
  basis::Basis
  mortar::Mortar
  surface_integral::SurfaceIntegral
  volume_integral::VolumeIntegral
end

function Base.show(io::IO, dg::DG)
  @nospecialize dg # reduce precompilation time

  print(io, "DG{", real(dg), "}(")
  print(io,       dg.basis)
  print(io, ", ", dg.mortar)
  print(io, ", ", dg.surface_integral)
  print(io, ", ", dg.volume_integral)
  print(io, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", dg::DG)
  @nospecialize dg # reduce precompilation time

  if get(io, :compact, false)
    show(io, dg)
  else
    summary_header(io, "DG{" * string(real(dg)) * "}")
    summary_line(io, "basis", dg.basis)
    summary_line(io, "mortar", dg.mortar)
    summary_line(io, "surface integral", dg.surface_integral |> typeof |> nameof)
    show(increment_indent(io), mime, dg.surface_integral)
    summary_line(io, "volume integral", dg.volume_integral |> typeof |> nameof)
    if !(dg.volume_integral isa VolumeIntegralWeakForm)
      show(increment_indent(io), mime, dg.volume_integral)
    end
    summary_footer(io)
  end
end

Base.summary(io::IO, dg::DG) = print(io, "DG(" * summary(dg.basis) * ")")

@inline Base.real(dg::DG) = real(dg.basis)

@inline ndofs(mesh::TreeMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


function get_element_variables!(element_variables, u, mesh, equations, dg::DG, cache)
  get_element_variables!(element_variables, u, mesh, equations, dg.volume_integral, dg, cache)
end


# TODO: Taal performance, 1:nnodes(dg) vs. Base.OneTo(nnodes(dg)) vs. SOneTo(nnodes(dg)) for DGSEM
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
  SVector(ntuple(@inline(idx -> x[idx, indices...]), Val(ndims(equations))))
end

@inline function get_node_vars(u, equations, solver::DG, indices...)
  # There is a cut-off at `n == 10` inside of the method
  # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
  # in Julia `v1.5`, leading to type instabilities if
  # more than ten variables are used. That's why we use
  # `Val(...)` below.
  # We use `@inline` to make sure that the `getindex` calls are
  # really inlined, which might be the default choice of the Julia
  # compiler for standard `Array`s but not necessarily for more
  # advanced array types such as `PtrArray`s, cf.
  # https://github.com/JuliaSIMD/VectorizationBase.jl/issues/55
  SVector(ntuple(@inline(v -> u[v, indices...]), Val(nvariables(equations))))
end

@inline function get_surface_node_vars(u, equations, solver::DG, indices...)
  # There is a cut-off at `n == 10` inside of the method
  # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
  # in Julia `v1.5`, leading to type instabilities if
  # more than ten variables are used. That's why we use
  # `Val(...)` below.
  u_ll = SVector(ntuple(@inline(v -> u[1, v, indices...]), Val(nvariables(equations))))
  u_rr = SVector(ntuple(@inline(v -> u[2, v, indices...]), Val(nvariables(equations))))
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

# Use this function instead of `add_to_node_vars` to speed up
# multiply-and-add-to-node-vars operations
# See https://github.com/trixi-framework/Trixi.jl/pull/643
@inline function multiply_add_to_node_vars!(u, factor, u_node, equations, solver::DG, indices...)
  for v in eachvariable(equations)
    u[v, indices...] = u[v, indices...] + factor * u_node[v]
  end
  return nothing
end


# Used for analyze_solution
SolutionAnalyzer(dg::DG; kwargs...) = SolutionAnalyzer(dg.basis; kwargs...)

AdaptorAMR(mesh, dg::DG) = AdaptorL2(dg.basis)



# Include utilities
include("interpolation.jl")
include("l2projection.jl")
include("basis_lobatto_legendre.jl")

"""
    DGSEM(; RealT=Float64, polydeg::Integer,
            surface_flux=flux_central,
            surface_integral=SurfaceIntegralWeakForm(surface_flux),
            volume_integral=VolumeIntegralWeakForm(),
            mortar=MortarL2(basis))

Create a discontinuous Galerkin spectral element method (DGSEM) using a
[`LobattoLegendreBasis`](@ref) with polynomials of degree `polydeg`.
"""
const DGSEM = DG{Basis} where {Basis<:LobattoLegendreBasis}

# TODO: Deprecated in v0.3 (no longer documented)
function DGSEM(basis::LobattoLegendreBasis,
               surface_flux=flux_central,
               volume_integral=VolumeIntegralWeakForm(),
               mortar=MortarL2(basis))

  surface_integral = SurfaceIntegralWeakForm(surface_flux)
  return DG{typeof(basis), typeof(mortar), typeof(surface_integral), typeof(volume_integral)}(
    basis, mortar, surface_integral, volume_integral)
end

# TODO: Deprecated in v0.3 (no longer documented)
function DGSEM(basis::LobattoLegendreBasis,
               surface_integral::AbstractSurfaceIntegral,
               volume_integral=VolumeIntegralWeakForm(),
               mortar=MortarL2(basis))

  return DG{typeof(basis), typeof(mortar), typeof(surface_integral), typeof(volume_integral)}(
    basis, mortar, surface_integral, volume_integral)
end

# TODO: Deprecated in v0.3 (no longer documented)
function DGSEM(RealT, polydeg::Integer,
               surface_flux=flux_central,
               volume_integral=VolumeIntegralWeakForm(),
               mortar=MortarL2(LobattoLegendreBasis(RealT, polydeg)))
  basis = LobattoLegendreBasis(RealT, polydeg)

  return DGSEM(basis, surface_flux, volume_integral, mortar)
end

DGSEM(polydeg, surface_flux=flux_central, volume_integral=VolumeIntegralWeakForm()) = DGSEM(Float64, polydeg, surface_flux, volume_integral)

# The constructor using only keyword arguments is convenient for elixirs since
# it allows to modify the polynomial degree and other parameters via
# `trixi_include`.
function DGSEM(; RealT=Float64,
                 polydeg::Integer,
                 surface_flux=flux_central,
                 surface_integral=SurfaceIntegralWeakForm(surface_flux),
                 volume_integral=VolumeIntegralWeakForm())
  basis = LobattoLegendreBasis(RealT, polydeg)
  return DGSEM(basis, surface_integral, volume_integral)
end

@inline polydeg(dg::DGSEM) = polydeg(dg.basis)

Base.summary(io::IO, dg::DGSEM) = print(io, "DGSEM(polydeg=$(polydeg(dg)))")



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


function volume_jacobian(element, mesh::TreeMesh, cache)
  return inv(cache.elements.inverse_jacobian[element])^ndims(mesh)
end



# indicators used for shock-capturing and AMR
include("indicators.jl")
include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")

# Container data structures
include("containers.jl")

# 1D DG implementation
include("dg_1d.jl")

# 2D DG implementation
include("dg_2d.jl")
include("dg_2d_parallel.jl")

# 3D DG implementation
include("dg_3d.jl")


end # @muladd
