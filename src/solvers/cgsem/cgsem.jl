# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CGSEM(; RealT = Float64, polydeg::Integer,
            surface_flux = flux_lax_friedrichs,
            volume_integral = VolumeIntegralWeakForm())
    CGSEM(dg::DGSEM)

Continuous Galerkin spectral element method (CGSEM) with polynomials of degree
`polydeg` on a [`LobattoLegendreBasis`](@ref), i.e., the continuous (C0) counterpart of
[`DGSEM`](@ref), on conforming [`P4estMesh`](@ref)es in 2D.

CGSEM reuses the element-local storage and the kernels of the `DGSEM` solver `dg`,
which is built by the keyword constructor or passed directly. The DGSEM volume
integrals already contain the element-boundary term ``-M^{-1} B f(u)`` of the weak
form, so summing the element-local residuals of all copies of a shared node yields the
continuous Galerkin scheme without interface fluxes. The right-hand side therefore
consists of the volume integral, the weakly imposed boundary fluxes, the source terms,
and the gather-scatter step [`assemble_and_apply_jacobian!`](@ref), which replaces the
DGSEM `apply_jacobian!`: at every node, the residuals of all copies are summed and
divided by the summed Jacobian (the assembled lumped mass matrix), and all copies
receive the identical value. The initial condition is evaluated as in DGSEM and then
projected onto the continuous space with [`l2_projection!`](@ref); see
[`AssemblyCGSEM`](@ref) for the data structure behind both kernels.

Supported are
- conforming (no mortars, no AMR), serial [`P4estMesh`](@ref)es in 2D, including
  curved and periodic meshes,
- equations without nonconservative terms,
- the volume integrals [`VolumeIntegralWeakForm`](@ref),
  [`VolumeIntegralFluxDifferencing`](@ref) (e.g., with an entropy-conservative
  two-point flux), [`VolumeIntegralPureLGLFiniteVolume`](@ref), and
  [`VolumeIntegralShockCapturingHG`](@ref) blending them; a shared node acts as a
  merged finite volume subcell in the latter two,
- weakly imposed boundary conditions as in DGSEM, which are the only place where
  `surface_flux` enters,
- source terms, which are added before the assembly,
- parabolic terms of a [`SemidiscretizationHyperbolicParabolic`](@ref) with
  [`ParabolicFormulationBassiRebay1`](@ref) and parabolic source terms (see below),
- the [`AnalysisCallback`](@ref), the [`StepsizeCallback`](@ref) (hyperbolic time step
  restriction only, using the DGSEM estimate), the [`SaveSolutionCallback`](@ref), and
  the plotting functions.

Everything else, e.g., nonconforming meshes and AMR, MPI-parallel meshes, GPU
backends, restart files, and the stage limiters of DGSEM, is not supported and fails
with an error instead of silently running the DGSEM code.

The gradients of the parabolic terms are computed element-locally from the continuous
solution with the strong form derivative, i.e., they are discontinuous across element
faces and are not assembled. Data of Dirichlet-type boundary conditions enters the
gradients via the lift ``M^{-1} B (u_{bc} - u) n`` on the boundary faces, as in the BR1
scheme of DGSEM, and Neumann data enters via the boundary flux of the divergence. The
divergence of the parabolic fluxes uses the weak form of DGSEM and is assembled by
[`assemble_and_apply_jacobian_parabolic!`](@ref). The parabolic formulation of DGSEM
only determines interface fluxes, which CGSEM does not compute, so only the default
[`ParabolicFormulationBassiRebay1`](@ref) is accepted.

Since no interface dissipation is present, CGSEM is a central scheme in the interior
of the domain; [`VolumeIntegralFluxDifferencing`](@ref) with an entropy-conservative
flux is recommended for nonlinear problems.

!!! warning "Experimental code"
    This code is experimental and may change in any future release.
"""
struct CGSEM{Basis, SurfaceIntegral, VolumeIntegral, DGType <: DGSEM}
    basis::Basis
    surface_integral::SurfaceIntegral
    volume_integral::VolumeIntegral
    # The DGSEM solver providing the element-local kernels and containers. All
    # element-local work is delegated to it; the fields above are its components so
    # that code reading `solver.basis` etc. keeps working.
    dg::DGType
end

# @eval due to @muladd
@eval Adapt.@adapt_structure(CGSEM)

function CGSEM(; RealT = Float64, polydeg::Integer,
               surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())
    dg = DGSEM(; RealT, polydeg, surface_flux, volume_integral)
    return CGSEM(dg)
end

function CGSEM(dg::DGSEM)
    if !(dg.basis isa LobattoLegendreBasis)
        throw(ArgumentError("CGSEM requires a `LobattoLegendreBasis` since the nodes " *
                            "must lie on the element faces, " *
                            "got `$(nameof(typeof(dg.basis)))`"))
    end
    if !(dg.surface_integral isa SurfaceIntegralWeakForm)
        throw(ArgumentError("CGSEM requires a `SurfaceIntegralWeakForm`, " *
                            "got `$(nameof(typeof(dg.surface_integral)))`"))
    end
    if !is_admissible_volume_integral_cgsem(dg.volume_integral)
        throw(ArgumentError("CGSEM does not support " *
                            "`$(nameof(typeof(dg.volume_integral)))`; supported are " *
                            "`VolumeIntegralWeakForm`, `VolumeIntegralFluxDifferencing`, " *
                            "`VolumeIntegralPureLGLFiniteVolume`, and " *
                            "`VolumeIntegralShockCapturingHG` blending them"))
    end
    return CGSEM(dg.basis, dg.surface_integral, dg.volume_integral, dg)
end

# A volume integral is admissible if DGSEM with its kernel and the weak-form surface
# integral with the own-state flux f*(u^-, u^+) = f(u^-) reduces to the volume kernel
# alone. Then, summing the element-local residuals yields the continuous Galerkin
# scheme. This holds for the weak form and flux differencing (summation-by-parts
# property) and for the subcell finite volume kernel (its faces at the element
# boundary carry the own-state flux), and thus also for convex blends of them.
is_admissible_volume_integral_cgsem(::VolumeIntegralWeakForm) = true
is_admissible_volume_integral_cgsem(::VolumeIntegralFluxDifferencing) = true
is_admissible_volume_integral_cgsem(::VolumeIntegralPureLGLFiniteVolume) = true
function is_admissible_volume_integral_cgsem(integral::VolumeIntegralShockCapturingHGType)
    return all(is_admissible_volume_integral_cgsem, volume_integral_children(integral))
end
is_admissible_volume_integral_cgsem(::AbstractVolumeIntegral) = false

function Base.show(io::IO, solver::CGSEM)
    @nospecialize solver # reduce precompilation time

    print(io, "CGSEM{", real(solver), "}(")
    print(io, solver.basis)
    print(io, ", ", solver.surface_integral)
    print(io, ", ", solver.volume_integral)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", solver::CGSEM)
    @nospecialize solver # reduce precompilation time

    if get(io, :compact, false)
        show(io, solver)
    else
        summary_header(io, "CGSEM{" * string(real(solver)) * "}")
        summary_line(io, "basis", solver.basis)
        summary_line(io, "surface integral",
                     solver.surface_integral |> typeof |> nameof)
        show(increment_indent(io), mime, solver.surface_integral)
        summary_line(io, "volume integral", solver.volume_integral |> typeof |> nameof)
        if !(solver.volume_integral isa VolumeIntegralWeakForm)
            show(increment_indent(io), mime, solver.volume_integral)
        end
        summary_footer(io)
    end
    return nothing
end

Base.summary(io::IO, solver::CGSEM) = print(io, "CGSEM(polydeg=$(polydeg(solver)))")

# The cache is the DGSEM cache of `solver.dg` plus the gather-scatter data. Only serial
# meshes are accepted since `AssemblyCGSEM` numbers the nodes via the local interfaces.
function create_cache_hyperbolic(mesh::P4estMeshSerial{2}, equations::AbstractEquations,
                                 solver::CGSEM, RealT,
                                 ::Type{uEltype}) where {uEltype <: Real}
    if have_nonconservative_terms(equations) isa True
        throw(ArgumentError("CGSEM does not support equations with nonconservative " *
                            "terms, got `$(nameof(typeof(equations)))`"))
    end
    (; dg) = solver

    # Containers and volume integral temporaries of DGSEM. `AssemblyCGSEM` checks that
    # the mesh is conforming, i.e., that the mortar container is empty.
    cache = create_cache_hyperbolic(mesh, equations, dg, RealT, uEltype)

    # CGSEM does not compute interface fluxes and never writes the interior interface
    # entries of `surface_flux_values`. They are zeroed once here so that the surface
    # integral adds only the boundary fluxes.
    fill!(cache.elements.surface_flux_values, zero(uEltype))

    assembly = AssemblyCGSEM(mesh, equations, solver, cache)
    return (; cache..., assembly)
end

# The parabolic cache is the one of DGSEM (element-local gradient variables, gradients,
# and parabolic fluxes). The parabolic right-hand side shares `surface_flux_values` and
# the boundary container with the hyperbolic one in `cache`, whose interior interface
# entries stay zero (see `create_cache`).
function create_cache_parabolic(mesh::P4estMesh{2},
                                equations_hyperbolic::AbstractEquations,
                                solver::CGSEM, n_elements, uEltype)
    return create_cache_parabolic(mesh, equations_hyperbolic, solver.dg, n_elements,
                                  uEltype)
end

function compute_coefficients!(backend::Nothing, u, func, t, mesh::P4estMesh{2},
                               equations, solver::CGSEM, cache)
    (; dg) = solver
    compute_coefficients!(backend, u, func, t, mesh, equations, dg, cache)
    # The pointwise evaluation of `func` is single-valued at shared nodes only up to
    # roundoff (the node coordinates of the copies differ by roundoff); the projection
    # makes the copies bitwise identical.
    l2_projection!(u, cache.assembly, mesh, equations, solver, cache)
    return nothing
end

###############################################################################
# Forwarding to the DGSEM solver
#
# CGSEM uses the element-local storage of DGSEM, so everything that only reads the
# element-local data (array layout, quadrature, analysis, time step estimate, solution
# files) is delegated to `solver.dg`. Functionality that is not listed here (e.g., AMR,
# stage limiters, parabolic time step estimates, MPI, GPU backends, restart files) fails
# with a `MethodError` instead of silently running the DGSEM code. The plotting entry
# points are forwarded in `src/visualization/types.jl`, where the plot data types are
# defined.

# Core
@inline Base.real(solver::CGSEM) = real(solver.dg)
@inline polydeg(solver::CGSEM) = polydeg(solver.dg)
@inline nnodes(solver::CGSEM) = nnodes(solver.dg)
@inline eachnode(solver::CGSEM) = eachnode(solver.dg)
@inline nelements(solver::CGSEM, cache) = nelements(solver.dg, cache)
@inline nelements(mesh, solver::CGSEM, cache) = nelements(mesh, solver.dg, cache)
@inline function nelementsglobal(mesh, solver::CGSEM, cache)
    return nelementsglobal(mesh, solver.dg, cache)
end
@inline eachelement(solver::CGSEM, cache) = eachelement(solver.dg, cache)
# Number of element-local degrees of freedom, i.e., the size of the solution arrays.
# The number of distinct nodes is `nnodes_unique(cache.assembly)`.
@inline ndofs(mesh, solver::CGSEM, cache) = ndofs(mesh, solver.dg, cache)
@inline ndofsglobal(mesh, solver::CGSEM, cache) = ndofsglobal(mesh, solver.dg, cache)

@inline function wrap_array(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                            solver::CGSEM, cache)
    return wrap_array(u_ode, mesh, equations, solver.dg, cache)
end
@inline function wrap_array_native(u_ode::AbstractVector, mesh::AbstractMesh, equations,
                                   solver::CGSEM, cache)
    return wrap_array_native(u_ode, mesh, equations, solver.dg, cache)
end
function allocate_coefficients(mesh::AbstractMesh, equations, solver::CGSEM, cache)
    return allocate_coefficients(mesh, equations, solver.dg, cache)
end
function get_element_variables!(element_variables, u, mesh, equations, solver::CGSEM,
                                cache)
    return get_element_variables!(element_variables, u, mesh, equations, solver.dg,
                                  cache)
end

# AnalysisCallback: errors and integrals are element-local quadratures
SolutionAnalyzer(solver::CGSEM; kwargs...) = SolutionAnalyzer(solver.dg; kwargs...)
function create_cache_analysis(analyzer, mesh::P4estMesh{2}, equations, solver::CGSEM,
                               cache, RealT, uEltype)
    return create_cache_analysis(analyzer, mesh, equations, solver.dg, cache, RealT,
                                 uEltype)
end
function calc_error_norms(func, u, t, analyzer, mesh::P4estMesh{2}, equations,
                          initial_condition, solver::CGSEM, cache, cache_analysis)
    return calc_error_norms(func, u, t, analyzer, mesh, equations, initial_condition,
                            solver.dg, cache, cache_analysis)
end
function integrate_via_indices(func::Func, u, mesh::P4estMesh{2}, equations,
                               solver::CGSEM, cache, args...;
                               normalize = true) where {Func}
    return integrate_via_indices(func, u, mesh, equations, solver.dg, cache, args...;
                                 normalize)
end
function integrate(func::Func, u, mesh::P4estMesh{2}, equations, solver::CGSEM, cache;
                   normalize = true) where {Func}
    return integrate(func, u, mesh, equations, solver.dg, cache; normalize)
end
# All analysis quantities, including `entropy_timederivative`
function analyze(quantity, du, u, t, mesh::P4estMesh{2}, equations, solver::CGSEM,
                 cache)
    return analyze(quantity, du, u, t, mesh, equations, solver.dg, cache)
end

# StepsizeCallback: the DGSEM estimate is used. CGSEM admits a somewhat larger time
# step since the assembled lumped mass matrix at a shared node is the sum of the
# element contributions; a specialized estimate can follow in the future.
function max_dt(u, t, mesh::P4estMesh{2}, constant_speed, equations, solver::CGSEM,
                cache)
    return max_dt(u, t, mesh, constant_speed, equations, solver.dg, cache)
end

# SaveSolutionCallback: the element-local arrays are written as for DGSEM
function save_solution_file(u, time, dt, timestep, mesh::P4estMesh{2}, equations,
                            solver::CGSEM, cache, solution_callback,
                            element_variables = Dict{Symbol, Any}(),
                            node_variables = Dict{Symbol, Any}(); system = "")
    return save_solution_file(u, time, dt, timestep, mesh, equations, solver.dg, cache,
                              solution_callback, element_variables, node_variables;
                              system)
end
end # @muladd
