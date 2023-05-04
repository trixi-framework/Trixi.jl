# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


abstract type AbstractVolumeIntegral end

get_element_variables!(element_variables, u, mesh, equations,
                       volume_integral::AbstractVolumeIntegral, dg, cache) = nothing


"""
    VolumeIntegralStrongForm()

The classical strong form volume integral type for FD/DG methods.
"""
struct VolumeIntegralStrongForm <: AbstractVolumeIntegral end


"""
    VolumeIntegralWeakForm()

The classical weak form volume integral type for DG methods as explained in
standard textbooks.

## References

- Kopriva (2009)
  Implementing Spectral Methods for Partial Differential Equations:
  Algorithms for Scientists and Engineers
  [doi: 10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
- Hesthaven, Warburton (2007)
  Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and
  Applications
  [doi: 10.1007/978-0-387-72067-8](https://doi.org/10.1007/978-0-387-72067-8)
"""
struct VolumeIntegralWeakForm <: AbstractVolumeIntegral end

create_cache(mesh, equations, ::VolumeIntegralWeakForm, dg, uEltype) = NamedTuple()


"""
    VolumeIntegralFluxDifferencing(volume_flux)

Volume integral type for DG methods based on SBP operators and flux differencing
using a symmetric two-point `volume_flux`. This `volume_flux` needs to satisfy
the interface of numerical fluxes in Trixi.jl.

## References

- LeFloch, Mercier, Rohde (2002)
  Fully Discrete, Entropy Conservative Schemes of Arbitrary Order
  [doi: 10.1137/S003614290240069X](https://doi.org/10.1137/S003614290240069X)
- Fisher, Carpenter (2013)
  High-order entropy stable finite difference schemes for nonlinear
  conservation laws: Finite domains
  [doi: 10.1016/j.jcp.2013.06.014](https://doi.org/10.1016/j.jcp.2013.06.014)
- Hendrik Ranocha (2017)
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
    VolumeIntegralShockCapturingHG(indicator; volume_flux_dg=flux_central,
                                              volume_flux_fv=flux_lax_friedrichs)

Shock-capturing volume integral type for DG methods using a convex blending of
the finite volume method with numerical flux `volume_flux_fv` and the
[`VolumeIntegralFluxDifferencing`](@ref) with volume flux `volume_flux_dg`.
The amount of blending is determined by the `indicator`, e.g.,
[`IndicatorHennemannGassner`](@ref).

## References

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

function get_element_variables!(element_variables, u, mesh, equations,
                                volume_integral::VolumeIntegralShockCapturingHG, dg, cache)
  # call the indicator to get up-to-date values for IO
  volume_integral.indicator(u, mesh, equations, dg, cache)
  get_element_variables!(element_variables, volume_integral.indicator, volume_integral)
end


"""
    VolumeIntegralPureLGLFiniteVolume(volume_flux_fv)

A volume integral that only uses the subcell finite volume schemes of the
[`VolumeIntegralShockCapturingHG`](@ref).

This gives a formally O(1)-accurate finite volume scheme on an LGL-type subcell
mesh (LGL = Legendre-Gauss-Lobatto).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

## References

- Hennemann, Gassner (2020)
  "A provably entropy stable subcell shock capturing approach for high order split form DG"
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
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


# TODO: FD. Should this definition live in a different file because it is
# not strictly a DG method?
"""
    VolumeIntegralUpwind(splitting)

Specialized volume integral for finite difference summation-by-parts (FDSBP)
solvers. Can be used together with the upwind SBP operators of Mattsson (2017)
implemented in SummationByPartsOperators.jl. The `splitting` controls the
discretization.

See also [`splitting_steger_warming`](@ref), [`splitting_lax_friedrichs`](@ref),
[`splitting_vanleer_haenel`](@ref).

## References

- Mattsson (2017)
  Diagonal-norm upwind SBP operators
  [doi: 10.1016/j.jcp.2017.01.042](https://doi.org/10.1016/j.jcp.2017.01.042)

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.
"""
struct VolumeIntegralUpwind{FluxSplitting} <: AbstractVolumeIntegral
  splitting::FluxSplitting
end

function Base.show(io::IO, ::MIME"text/plain", integral::VolumeIntegralUpwind)
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "flux splitting" => integral.splitting
            ]
    summary_box(io, "VolumeIntegralUpwind", setup)
  end
end


abstract type AbstractSurfaceIntegral end

"""
    SurfaceIntegralWeakForm(surface_flux=flux_central)

The classical weak form surface integral type for DG methods as explained in standard
textbooks.

See also [`VolumeIntegralWeakForm`](@ref).

## References

- Kopriva (2009)
  Implementing Spectral Methods for Partial Differential Equations:
  Algorithms for Scientists and Engineers
  [doi: 10.1007/978-90-481-2261-5](https://doi.org/10.1007/978-90-481-2261-5)
- Hesthaven, Warburton (2007)
  Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and
  Applications
  [doi: 10.1007/978-0-387-72067-8](https://doi.org/10.1007/978-0-387-72067-8)
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


# TODO: FD. Should this definition live in a different file because it is
# not strictly a DG method?
"""
    SurfaceIntegralUpwind(splitting)

Couple elements with upwind simultaneous approximation terms (SATs)
that use a particular flux `splitting`, e.g.,
[`splitting_steger_warming`](@ref).

See also [`VolumeIntegralUpwind`](@ref).

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.
"""
struct SurfaceIntegralUpwind{FluxSplitting} <: AbstractSurfaceIntegral
  splitting::FluxSplitting
end

function Base.show(io::IO, ::MIME"text/plain", integral::SurfaceIntegralUpwind)
  @nospecialize integral # reduce precompilation time

  if get(io, :compact, false)
    show(io, integral)
  else
    setup = [
            "flux splitting" => integral.splitting
            ]
    summary_box(io, "SurfaceIntegralUpwind", setup)
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


function get_element_variables!(element_variables, u, mesh, equations, dg::DG, cache)
  get_element_variables!(element_variables, u, mesh, equations, dg.volume_integral, dg, cache)
end


const MeshesDGSEM = Union{TreeMesh, StructuredMesh, UnstructuredMesh2D, P4estMesh}

@inline ndofs(mesh::MeshesDGSEM, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)

# TODO: Taal performance, 1:nnodes(dg) vs. Base.OneTo(nnodes(dg)) vs. SOneTo(nnodes(dg)) for DGSEM
"""
    eachnode(dg::DG)

Return an iterator over the indices that specify the location in relevant data structures
for the nodes in `dg`. 
In particular, not the nodes themselves are returned.
"""
@inline eachnode(dg::DG) = Base.OneTo(nnodes(dg))
@inline nnodes(dg::DG)   = nnodes(dg.basis)

# This is used in some more general analysis code and needs to dispatch on the
# `mesh` for some combinations of mesh/solver.
@inline nelements(mesh, dg::DG, cache) = nelements(dg, cache)
@inline ndofsglobal(mesh, dg::DG, cache) = nelementsglobal(dg, cache) * nnodes(dg)^ndims(mesh)

"""
    eachelement(dg::DG, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the elements in `cache`. 
In particular, not the elements themselves are returned.
"""
@inline eachelement(dg::DG, cache)   = Base.OneTo(nelements(dg, cache))

"""
    eachinterface(dg::DG, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the interfaces in `cache`. 
In particular, not the interfaces themselves are returned.
"""
@inline eachinterface(dg::DG, cache) = Base.OneTo(ninterfaces(dg, cache))

"""
    eachboundary(dg::DG, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the boundaries in `cache`. 
In particular, not the boundaries themselves are returned.
"""
@inline eachboundary(dg::DG, cache)  = Base.OneTo(nboundaries(dg, cache))

"""
    eachmortar(dg::DG, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the mortars in `cache`. 
In particular, not the mortars themselves are returned.
"""
@inline eachmortar(dg::DG, cache)    = Base.OneTo(nmortars(dg, cache))

"""
    eachmpiinterface(dg::DG, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the MPI interfaces in `cache`. 
In particular, not the interfaces themselves are returned.
"""
@inline eachmpiinterface(dg::DG, cache) = Base.OneTo(nmpiinterfaces(dg, cache))

"""
    eachmpimortar(dg::DG, cache)

Return an iterator over the indices that specify the location in relevant data structures
for the MPI mortars in `cache`. 
In particular, not the mortars themselves are returned.
"""
@inline eachmpimortar(dg::DG, cache) = Base.OneTo(nmpimortars(dg, cache))

@inline nelements(dg::DG, cache)   = nelements(cache.elements)
@inline nelementsglobal(dg::DG, cache) = mpi_isparallel() ? cache.mpi_cache.n_elements_global : nelements(dg, cache)
@inline ninterfaces(dg::DG, cache) = ninterfaces(cache.interfaces)
@inline nboundaries(dg::DG, cache) = nboundaries(cache.boundaries)
@inline nmortars(dg::DG, cache)    = nmortars(cache.mortars)
@inline nmpiinterfaces(dg::DG, cache) = nmpiinterfaces(cache.mpi_interfaces)
@inline nmpimortars(dg::DG, cache) = nmpimortars(cache.mpi_mortars)


# The following functions assume an array-of-structs memory layout
# We would like to experiment with different memory layout choices
# in the future, see
# - https://github.com/trixi-framework/Trixi.jl/issues/88
# - https://github.com/trixi-framework/Trixi.jl/issues/87
# - https://github.com/trixi-framework/Trixi.jl/issues/86
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

@inline function set_node_vars!(u, u_node, variables::NTuple{N, Integer}, solver::DG, indices...) where N
  for v in variables
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



# General structs for discretizations based on the basic principle of
# DGSEM (discontinuous Galerkin spectral element method)
include("dgsem/dgsem.jl")



function allocate_coefficients(mesh::AbstractMesh, equations, dg::DG, cache)
  # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
  # cf. wrap_array
  zeros(eltype(cache.elements), nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
end

@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations, dg::DGSEM, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  # We would like to use
  #     reshape(u_ode, (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
  # but that results in
  #     ERROR: LoadError: cannot resize array with shared data
  # when we resize! `u_ode` during AMR.
  #
  # !!! danger "Segfaults"
  #     Remember to `GC.@preserve` temporaries such as copies of `u_ode`
  #     and other stuff that is only used indirectly via `wrap_array` afterwards!

  # Currently, there are problems when AD is used with `PtrArray`s in broadcasts
  # since LoopVectorization does not support `ForwardDiff.Dual`s. Hence, we use
  # optimized `PtrArray`s whenever possible and fall back to plain `Array`s
  # otherwise.
  if LoopVectorization.check_args(u_ode)
    # This version using `PtrArray`s from StrideArrays.jl is very fast and
    # does not result in allocations.
    #
    # !!! danger "Heisenbug"
    #     Do not use this code when `@threaded` uses `Threads.@threads`. There is
    #     a very strange Heisenbug that makes some parts very slow *sometimes*.
    #     In fact, everything can be fast and fine for many cases but some parts
    #     of the RHS evaluation can take *exactly* (!) five seconds randomly...
    #     Hence, this version should only be used when `@threaded` is based on
    #     `@batch` from Polyester.jl or something similar. Using Polyester.jl
    #     is probably the best option since everything will be handed over to
    #     Chris Elrod, one of the best performance software engineers for Julia.
    PtrArray(pointer(u_ode),
             (StaticInt(nvariables(equations)), ntuple(_ -> StaticInt(nnodes(dg)), ndims(mesh))..., nelements(dg, cache)))
            #  (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
  else
    # The following version is reasonably fast and allows us to `resize!(u_ode, ...)`.
    unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
                (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
  end
end

# General fallback
@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations, dg::DG, cache)
  wrap_array_native(u_ode, mesh, equations, dg, cache)
end

# Like `wrap_array`, but guarantees to return a plain `Array`, which can be better
# for interfacing with external C libraries (MPI, HDF5, visualization),
# writing solution files etc.
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations, dg::DG, cache)
  @boundscheck begin
    @assert length(u_ode) == nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache)
  end
  unsafe_wrap(Array{eltype(u_ode), ndims(mesh)+2}, pointer(u_ode),
              (nvariables(equations), ntuple(_ -> nnodes(dg), ndims(mesh))..., nelements(dg, cache)))
end


function compute_coefficients!(u, func, t, mesh::AbstractMesh{1}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, element)
    end
  end
end

function compute_coefficients!(u, func, t, mesh::AbstractMesh{2}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, element)
    end
  end
end

function compute_coefficients!(u, func, t, mesh::AbstractMesh{3}, equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      x_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, k, element)
      u_node = func(x_node, t, equations)
      set_node_vars!(u, u_node, equations, dg, i, j, k, element)
    end
  end
end


# Discretizations specific to each mesh type of Trixi.jl
# If some functionality is shared by multiple combinations of meshes/solvers,
# it is defined in the directory of the most basic mesh and solver type.
# The most basic solver type in Trixi.jl is DGSEM (historic reasons and background
# of the main contributors).
# We consider the `TreeMesh` to be the most basic mesh type since it is Cartesian
# and was the first mesh in Trixi.jl. The order of the other mesh types is the same
# as the include order below.
include("dgsem_tree/dg.jl")
include("dgsem_structured/dg.jl")
include("dgsem_unstructured/dg.jl")
include("dgsem_p4est/dg.jl")


# Finite difference methods using summation by parts (SBP) operators
# These methods are very similar to DG methods since they also impose interface
# and boundary conditions weakly. Thus, these methods can re-use a lot of
# functionality implemented for DGSEM.
include("fdsbp_tree/fdsbp.jl")


end # @muladd
