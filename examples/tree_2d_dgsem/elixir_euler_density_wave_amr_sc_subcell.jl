using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

# initial_condition = Trixi.initial_condition_density_wave_highdensity
initial_condition = Trixi.initial_condition_density_wave

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["rho"],
                                # positivity_variables_nonlinear = [pressure],
                                )
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)

mortar = MortarIDP(basis; alternative = false, local_factor = true,
                   basis_function = :piecewise_constant,
                   positivity_variables_cons = [1],
                   positivity_variables_nonlinear = [pressure])
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

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient,))

# amr_indicator = IndicatorMax(semi, variable = first)

# amr_controller = ControllerThreeLevel(semi, amr_indicator,
#                                       base_level = initial_refinement_level,
#                                       med_level = initial_refinement_level + 1, med_threshold = 2.0,
#                                       max_level = initial_refinement_level + 2, max_threshold = 2.05)

# amr_callback = AMRCallback(semi, amr_controller,
#                            interval = 1,
#                            adapt_initial_condition = true,
#                            adapt_initial_condition_only_refine = false)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        # amr_callback,
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
