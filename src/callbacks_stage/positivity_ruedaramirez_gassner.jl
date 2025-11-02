# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PositivityPreservingLimiterRuedaRamirezGassner

Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations based on
Finite Volume subcells as proposed in

- A. M. Rueda-Ramirez, G. J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations
  [DOI: 10.48550/arXiv.2102.06017](https://doi.org/10.48550/arXiv.2102.06017)

Can be seen as a generalization of the [`PositivityPreservingLimiterZhangShu`](@ref) with a
more sophisticated safe solution state.
# TODO: explain math and keywords
"""
mutable struct PositivityPreservingLimiterRuedaRamirezGassner{RealT <: Real,
                                                              SolverFV <: DGSEM,
                                                              uType <: AbstractVector,
                                                              vType <: AbstractVector}
    ### Limiter parameters ###
    beta::RealT # Factor that quantifies the permitted DG lower deviation from the FV solution
    alpha_max::RealT # Maximum FV-DG blending coefficient
    near_zero_tol::RealT # Tolerance to avoid division by close-to-zero denominators

    # Newton parameters for pressure correction
    max_iterations::Int # Maximum number iterations
    root_tol::RealT # Determines if correction is needed at all & convergence of Newton iteration
    damping::RealT # Damping factor

    solver_fv::SolverFV # Finite Volume solver

    ### Additional storage ###
    u_fv_ode::uType       # Finite-Volume update
    u_dg_node::vType      # Pure DG solution at a node
    du_dalpha_node::vType # Derivative of solution w.r.t. alpha at a node
    dp_du_node::vType     # Derivative of pressure w.r.t. conserved variables at a node
    u_newton_node::vType  # Temporary storage for Newton update at a node
end

function PositivityPreservingLimiterRuedaRamirezGassner(semi::AbstractSemidiscretization;
                                                        beta = 0.1,
                                                        near_zero_tol = 1e-9,
                                                        max_iterations = 10,
                                                        root_tol = 1e-15, damping = 0.8,
                                                        surface_flux_fv = semi.solver.surface_integral.surface_flux,
                                                        # Note: These default values assume `VolumeIntegralShockCapturingHG`
                                                        volume_flux_fv = semi.solver.volume_integral.volume_flux_fv,
                                                        alpha_max = semi.solver.volume_integral.indicator.alpha_max)
    @unpack basis = semi.solver
    @unpack equations = semi

    volume_integral = VolumeIntegralPureLGLFiniteVolume(volume_flux_fv)
    solver_fv = DGSEM(basis, surface_flux_fv, volume_integral)

    # Array for FV solution
    u_fv_ode = allocate_coefficients(mesh_equations_solver_cache(semi)...)

    n_vars = nvariables(equations)
    RealT = real(solver_fv)

    # Short vectors for storing temporary solutions and derivatives at a node
    u_dg_node = MVector{n_vars, RealT}(undef)
    du_dalpha_node = MVector{n_vars, RealT}(undef)
    dp_du_node = MVector{n_vars, RealT}(undef)
    u_newton_node = MVector{n_vars, RealT}(undef)

    return PositivityPreservingLimiterRuedaRamirezGassner{RealT,
                                                          typeof(solver_fv),
                                                          typeof(u_fv_ode),
                                                          typeof(u_dg_node)}(beta,
                                                                             alpha_max,
                                                                             near_zero_tol,
                                                                             max_iterations,
                                                                             root_tol,
                                                                             damping,
                                                                             solver_fv,
                                                                             u_fv_ode,
                                                                             u_dg_node,
                                                                             du_dalpha_node,
                                                                             dp_du_node,
                                                                             u_newton_node)
end

# Required to be used as `stage_callback` in `Trixi.SimpleIntegratorSSP`
init_callback(limiter!::PositivityPreservingLimiterRuedaRamirezGassner, semi) = nothing
# Required to be used as `stage_callback` in `Trixi.SimpleIntegratorSSP`
finalize_callback(limiter!::PositivityPreservingLimiterRuedaRamirezGassner, semi) = nothing

# Get pure FV solution for the current stage
function compute_u_fv!(limiter::PositivityPreservingLimiterRuedaRamirezGassner,
                       integrator::Trixi.SimpleIntegratorSSP, stage)
    @unpack alg, t, dt, u, du, f = integrator

    semi = integrator.p
    @unpack mesh, equations, boundary_conditions, source_terms, cache = semi

    @unpack solver_fv, u_fv_ode = limiter

    # Since AMRCallback is a stepcallback, it suffices to resize at stage 1 only
    if stage == 1
        resize!(u_fv_ode, length(u))
    end

    # Revert the DG/DGFV update
    @threaded for i in eachindex(u_fv_ode)
        u_fv_ode[i] = u[i] - dt * du[i]
    end

    # Compute first-order FV update, overwrite `integrator.du` with FV RHS
    rhs!(wrap_array(du, semi), wrap_array(u_fv_ode, semi),
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

function (limiter!::PositivityPreservingLimiterRuedaRamirezGassner)(u_ode,
                                                                    integrator::Trixi.SimpleIntegratorSSP,
                                                                    stage)
    @trixi_timeit timer() "positivity-preserving limiter RRG" begin
        semi = integrator.p
        @unpack mesh = semi

        # pure FV solution for stage s
        compute_u_fv!(limiter!, integrator, stage)

        u_dgfv = wrap_array(u_ode, semi)
        limiter_rueda_gassner!(u_dgfv, mesh, semi, limiter!)
    end

    return nothing
end

include("positivity_ruedaramirez_gassner_1d.jl")
end # @muladd
