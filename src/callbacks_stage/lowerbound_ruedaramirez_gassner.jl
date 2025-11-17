# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LowerBoundPreservingLimiterRuedaRamirezGassner(semi::AbstractSemidiscretization;
                                                   beta_rho = 0.1, beta_p = 0.1,
                                                   near_zero_tol = 1e-9,
                                                   max_iterations = 10,
                                                   root_tol = 1e-13, damping = 0.8,
                                                   use_density_init = true,
                                                   surface_flux_fv = semi.solver.surface_integral.surface_flux,
                                                   volume_flux_fv = semi.solver.volume_integral.volume_flux_fv,
                                                   alpha_max = semi.solver.volume_integral.indicator.alpha_max + 0.1)

Lower-bound (and thus also positivity-preserving) limiter for density and pressure for
finite volume (FV) subcell stabilized DGSEM.
Applicable to every set of equations where density and pressure are the only variables
required to be bounded from below (i.e., positive)
These are for instance the [`CompressibleEulerEquations1D`](@ref) or the [`IdealGlmMhdEquations1D`](@ref)
in any spatial dimension.

!!! note
    This limiter requires the usage of the [`VolumeIntegralShockCapturingHG`](@ref)
    volume integral.
    Furthermore, this limiter is currently only implemented as a `stage_callback` for
    [`Trixi.SimpleSSPRK33`](@ref).

This limiter is a generalization of the [`PositivityPreservingLimiterZhangShu`](@ref) with a
more sophisticated safe solution state.
In particular, positivity at the DG nodes is not just enforced by limiting towards 
the average/mean of the entire DG element, but instead by limiting towards a
first-order finite volume solution on the subcells defined by the DG nodes.
In particular, if positivity (or more general a relative lower bound) is violated at a DG node ``j``,
the solution is limited towards the FV solution at that node by increasing the blending coefficient ``\alpha`` 
in the Hennemann-Gassner shock capturing approach:
```math
u_j^\mathrm{DG-FV} = [1 - (\alpha^\mathrm{SC-HG} + \alpha^\mathrm{PP-RRG})] u_j^\mathrm{DG} + 
(\alpha^\mathrm{SC-HG} + \alpha^\mathrm{PP-RRG}) u_j^\mathrm{FV}
```
In addition to the increased accuracy of this limiter, it also comes with the option to
require limiting if the pure DG solution goes below a certain fraction of the FV solution.
This is controlled by the parameters ``\beta_\rho, \beta_p`` via:
```math
\rho^\mathrm{DG} \overset{!}{\geq} \beta_\rho \rho^\mathrm{FV}, \quad 
p^\mathrm{DG} \overset{!}{\geq} \beta_p p^\mathrm{FV}
```
The major computational cost of this limiter is the computation of the pure first-order FV solution
for every stage of the time integrator.

# Arguments
- `semi::AbstractSemidiscretization`: The semidiscretization to which this limiter is applied.
- `beta_rho::RealT = 0.1`: Factor that quantifies the permitted DG lower relative density deviation from the FV solution,
                           see formula above. Must be in (0, 1].
- `beta_p::RealT = 0.1`: Factor that quantifies the permitted DG lower relative deviation from the FV solution,
                         see formula above. Must be in (0, 1].
- `near_zero_tol::RealT = 1e-9`: Tolerance to avoid division by close-to-zero denominators.
- `max_iterations::Int = 10`: Maximum number of Newton iterations to compute the required increase in ``\alpha``.
- `root_tol::RealT = 1e-13`: Tolerance to determine if correction of the blending parameter ``\alpha``
                             is needed at all & convergence of Newton iteration.
- `damping::RealT = 0.8`: Damping factor for the Newton iteration.
- `use_density_init::Bool = true`: Whether to use the density-based ``\delta \alpha`` as initial guess
                                   for the pressure correction Newton iteration.
- `surface_flux_fv`: Surface flux function for the finite volume solver.
                     By default the same flux as for the DGSEM solver.
- `volume_flux_fv`: Volume flux function for the finite volume solver.
                    By default the same flux as used for the finite volume subcells in
                    [`VolumeIntegralShockCapturingHG`](@ref).
- `alpha_max::RealT`: Maximum allowed value for the blending coefficient ``\alpha``.
                      By default 0.1 larger than the value as used in the [`VolumeIntegralShockCapturingHG`](@ref).
                      This is required to ensure that the limiter has any room to work in.
                      May at most be 1.0 to maintain the convex combination property.

# Reference
- A. M. Rueda-Ramírez, G. J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations
  [DOI: 10.48550/arXiv.2102.06017](https://doi.org/10.48550/arXiv.2102.06017)
"""
mutable struct LowerBoundPreservingLimiterRuedaRamirezGassner{RealT <: Real,
                                                              SolverFV <: DGSEM,
                                                              uType <: AbstractVector,
                                                              vType <: AbstractArray}
    ### Limiter parameters ###
    const beta_rho::RealT # Factor that quantifies the permitted DG lower density deviation from the FV value
    const beta_p::RealT # Factor that quantifies the permitted DG lower pressure deviation from the FV value
    const alpha_max::RealT # Maximum FV-DG blending coefficient
    const near_zero_tol::RealT # Tolerance to avoid division by close-to-zero denominators

    # Newton parameters for pressure correction
    const max_iterations::Int # Maximum number iterations
    const root_tol::RealT # Determines if correction is needed at all & convergence of Newton iteration
    const damping::RealT # Damping factor
    const use_density_init::Bool # Whether to use the density-based `delta_alpha` as initial guess for pressure correction

    const solver_fv::SolverFV # Finite Volume solver

    ### Additional storage ###
    u_fv_ode::uType # Finite-Volume update
    # These are thread-local storage for temporary variables at a node
    u_dg_node_threaded::vType      # Pure DG solution
    du_dalpha_node_threaded::vType # Derivative of solution w.r.t. alpha
    dp_du_node_threaded::vType     # Derivative of pressure w.r.t. conserved variables
    u_newton_node_threaded::vType  # Temporary storage for Newton update
end

function LowerBoundPreservingLimiterRuedaRamirezGassner(semi::AbstractSemidiscretization;
                                                        beta_rho = 0.1, beta_p = 0.1,
                                                        near_zero_tol = 1e-9,
                                                        max_iterations = 10,
                                                        root_tol = 1e-13, damping = 0.8,
                                                        use_density_init = true,
                                                        surface_flux_fv = semi.solver.surface_integral.surface_flux,
                                                        volume_flux_fv = semi.solver.volume_integral.volume_flux_fv,
                                                        alpha_max = semi.solver.volume_integral.indicator.alpha_max +
                                                                    0.1)
    if beta_rho <= 0 || beta_rho > 1
        throw(ArgumentError("`beta_rho` must be in (0, 1] to make the limiter work!"))
    end
    if beta_p <= 0 || beta_p > 1
        throw(ArgumentError("`beta_p` must be in (0, 1] to make the limiter work!"))
    end
    if alpha_max > 1
        throw(ArgumentError("`alpha_max` must be less than or equal to 1 to maintain convex combination property!"))
    end

    @unpack basis = semi.solver
    @unpack equations = semi

    volume_integral = VolumeIntegralPureLGLFiniteVolume(volume_flux_fv)
    solver_fv = DGSEM(basis, surface_flux_fv, volume_integral)

    # Array for FV solution
    u_fv_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)

    n_vars = nvariables(equations)
    RealT = real(solver_fv)
    MV = MVector{n_vars, RealT}

    # Short thread-local vectors for storing temporary solutions and derivatives at a node
    u_dg_node_threaded = MV[MV(undef) for _ in 1:Threads.maxthreadid()]
    du_dalpha_node_threaded = MV[MV(undef) for _ in 1:Threads.maxthreadid()]
    dp_du_node_threaded = MV[MV(undef) for _ in 1:Threads.maxthreadid()]
    u_newton_node_threaded = MV[MV(undef) for _ in 1:Threads.maxthreadid()]

    return LowerBoundPreservingLimiterRuedaRamirezGassner{RealT,
                                                          typeof(solver_fv),
                                                          typeof(u_fv_ode),
                                                          typeof(u_dg_node_threaded)}(beta_rho,
                                                                                      beta_p,
                                                                                      alpha_max,
                                                                                      near_zero_tol,
                                                                                      max_iterations,
                                                                                      root_tol,
                                                                                      damping,
                                                                                      use_density_init,
                                                                                      solver_fv,
                                                                                      u_fv_ode,
                                                                                      u_dg_node_threaded,
                                                                                      du_dalpha_node_threaded,
                                                                                      dp_du_node_threaded,
                                                                                      u_newton_node_threaded)
end

# Required to be used as `stage_callback` in `Trixi.SimpleIntegratorSSP`
init_callback(limiter!::LowerBoundPreservingLimiterRuedaRamirezGassner, semi) = nothing
# Required to be used as `stage_callback` in `Trixi.SimpleIntegratorSSP`
finalize_callback(limiter!::LowerBoundPreservingLimiterRuedaRamirezGassner, semi) = nothing

# Get pure FV solution for the current stage
function compute_u_fv!(limiter::LowerBoundPreservingLimiterRuedaRamirezGassner,
                       integrator::Trixi.SimpleIntegratorSSP, stage)
    @unpack alg, t, dt, u, du, f = integrator

    semi = integrator.p
    @unpack mesh, equations, boundary_conditions, source_terms, cache = semi

    @unpack solver_fv, u_fv_ode = limiter

    # Since AMRCallback is a StepCallback, it suffices to resize at stage 1 only
    if stage == 1
        resize!(u_fv_ode, length(u))
    end

    # Revert the DG/DGFV update
    @threaded for i in eachindex(u_fv_ode)
        u_fv_ode[i] = u[i] - dt * du[i]
    end

    # Compute first-order FV update, overwrite `integrator.du` with FV RHS
    @trixi_timeit timer() "rhs!" rhs!(wrap_array(du, semi), wrap_array(u_fv_ode, semi),
                                      t + dt * alg.c[stage],
                                      mesh, equations,
                                      boundary_conditions, source_terms,
                                      solver_fv, cache)

    # Apply the first-order FV update
    @threaded for i in eachindex(u_fv_ode)
        u_fv_ode[i] += dt * du[i]
    end

    return nothing
end

function (limiter!::LowerBoundPreservingLimiterRuedaRamirezGassner)(u_ode,
                                                                    integrator::Trixi.SimpleIntegratorSSP,
                                                                    stage)
    @trixi_timeit timer() "positivity-preserving limiter RRG" begin
        semi = integrator.p
        @unpack mesh = semi

        # pure FV solution for current `stage`
        compute_u_fv!(limiter!, integrator, stage)

        u_dgfv = wrap_array(u_ode, semi)
        limiter_rueda_gassner!(u_dgfv, mesh, semi, limiter!)
    end

    return nothing
end

# Compute DG-FV solution update given Hennemann-Gassner blending:
#    u_dgfv_old = (1 - α) u_dg + α * u_fv
#    u_dgfv_new = (1 - [α + Δα]) u_dg + [α + Δα] * u_fv
# => u_dgfv_new = u_dgfv_old + Δα * (u_fv - u_dg)
@inline function compute_dgfv_update(ui_dgfv, ui_fv, ui_dg, delta_alpha)
    return ui_dgfv + delta_alpha * (ui_fv - ui_dg)
end

# Compute pure DG solution given Hennemann-Gassner blending:
#     u_dgfv = (1 - α) u_dg + α * u_fv
# <=>   u_dg = (u_dgfv - α * u_fv) / (1 - α)
@inline function compute_pure_dg(ui_dgfv, ui_fv, alpha)
    return (ui_dgfv - alpha * ui_fv) / (1 - alpha)
end

include("lowerbound_ruedaramirez_gassner_1d.jl")
end # @muladd
