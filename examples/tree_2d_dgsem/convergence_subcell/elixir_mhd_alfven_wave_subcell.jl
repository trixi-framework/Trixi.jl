
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations
gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

initial_condition = initial_condition_convergence_test

volume_flux = (flux_central, flux_nonconservative_powell2)
surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell2)

basis = LobattoLegendreBasis(3)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons=[1],
                                positivity_correction_factor=0.1)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg=volume_flux,
                                                volume_flux_fv=surface_flux)

solver = DGSEM(polydeg = 3,
               surface_flux = surface_flux,
               volume_integral = volume_integral)

coordinates_min = (0.0, 0.0)
coordinates_max = (sqrt(2.0), sqrt(2.0))
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 100_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = true,
                                     extra_analysis_integrals = (entropy, energy_total,
                                                                 energy_kinetic,
                                                                 energy_internal,
                                                                 energy_magnetic,
                                                                 cross_helicity))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

cfl = 0.5
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
stage_callbacks = (SubcellLimiterIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks); #
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
