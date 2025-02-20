using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal multi-ion MHD equations
equations = IdealGlmMhdMultiIonEquations2D(gammas = (1.4, 1.667),
                                           charge_to_mass = (1.0, 2.0))

initial_condition = initial_condition_weak_blast_wave

# Entropy conservative numerical fluxes
volume_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
surface_flux = (flux_ruedaramirez_etal, flux_nonconservative_ruedaramirez_etal)
# For provably entropy-stable surface fluxes, use
# surface_flux = (FluxPlusDissipation(flux_ruedaramirez_etal, DissipationLaxFriedrichsEntropyVariables()),
#                 flux_nonconservative_ruedaramirez_etal)
# For a standard local Lax-Friedrichs surface flux, use
# surface_flux = (flux_lax_friedrichs, flux_nonconservative_central)

solver = DGSEM(polydeg = 3, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-2.0, -2.0)
coordinates_max = (2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_lorentz)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 10
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1, # interval=100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 0.5

stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
