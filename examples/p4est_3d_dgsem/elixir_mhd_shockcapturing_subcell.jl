using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations3D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations3D)

Weak magnetic blast wave setup taken from Section 6.1 of the paper:
- A. M. Rueda-Ramírez, S. Hennemann, F. J. Hindenlang, A. R. Winters, G. J. Gassner (2021)
  An entropy stable nodal discontinuous Galerkin method for the resistive MHD
  equations. Part II: Subcell finite volume shock capturing
  [doi: 10.1016/j.jcp.2021.110580](https://doi.org/10.1016/j.jcp.2021.110580)
"""
function initial_condition_blast_wave(x, t, equations::IdealGlmMhdEquations3D)
    # Center of the blast wave is selected for the domain [0, 3]^3
    inicenter = (1.5, 1.5, 1.5)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    z_norm = x[3] - inicenter[3]
    r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)

    delta_0 = 0.1
    r_0 = 0.3
    lambda = exp(5.0 / delta_0 * (r - r_0))

    prim_inner = SVector(1.2, 0.1, 0.0, 0.1, 0.9, 1.0, 1.0, 1.0, 0.0)
    prim_outer = SVector(1.2, 0.2, -0.4, 0.2, 0.3, 1.0, 1.0, 1.0, 0.0)
    prim_vars = (prim_inner + lambda * prim_outer) / (1.0 + lambda)

    return prim2cons(prim_vars, equations)
end
initial_condition = initial_condition_blast_wave

# Up to version 0.13.0, `max_abs_speed_naive` was used as the default wave speed estimate of
# `const flux_lax_friedrichs = FluxLaxFriedrichs(), i.e., `FluxLaxFriedrichs(max_abs_speed = max_abs_speed_naive)`.
# In the `StepsizeCallback`, though, the less diffusive `max_abs_speeds` is employed which is consistent with `max_abs_speed`.
# Thus, we exchanged in PR#2458 the default wave speed used in the LLF flux to `max_abs_speed`.
# To ensure that every example still runs we specify explicitly `FluxLaxFriedrichs(max_abs_speed_naive)`.
# We remark, however, that the now default `max_abs_speed` is in general recommended due to compliance with the
# `StepsizeCallback` (CFL-Condition) and less diffusion.
surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive),
                flux_nonconservative_powell_local_symmetric)
# volume_flux = (flux_derigs_etal, flux_nonconservative_powell_local_symmetric)
volume_flux = (flux_central, flux_nonconservative_powell_local_symmetric)

# surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
# volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

surface_flux = (FluxLaxFriedrichs(max_abs_speed_naive), flux_nonconservative_powell)
volume_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)

# TODO: Test with working fluxes

basis = LobattoLegendreBasis(3)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                positivity_variables_nonlinear = [pressure],
                                positivity_correction_factor = 0.1)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5, -0.5)
coordinates_max = (0.5, 0.5, 0.5)
trees_per_dimension = (2, 2, 2)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 3,
                 #  mapping = mapping,
                 coordinates_min = coordinates_min,
                 coordinates_max = coordinates_max,
                 initial_refinement_level = 2,
                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

cfl = 0.4
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation
stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback())

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks);
