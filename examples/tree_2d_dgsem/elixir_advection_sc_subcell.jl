using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

surface_flux = flux_lax_friedrichs
volume_flux = flux_godunov
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["first"],
                                )
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

mortar = Trixi.MortarIDP(basis; alternative = false, local_factor = true,
                         first_order = true)
solver = DGSEM(basis, surface_flux, volume_integral, mortar)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
refinement_patches = ((type = "box", coordinates_min = (0.0, -0.5),
                       coordinates_max = (0.5, 0.5)),)
initial_refinement_level = 4
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 10_000,
                refinement_patches = refinement_patches,
                periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_errors = (:conservation_error,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback(save_errors = true))

sol = Trixi.solve(ode,
                  # Trixi.SimpleEuler(stage_callbacks = stage_callbacks);
                  Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);
