
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

initial_condition = Trixi.initial_condition_double_mach_reflection

boundary_condition_inflow_outflow = BoundaryConditionCharacteristic(initial_condition)

boundary_conditions = (y_neg = Trixi.boundary_condition_mixed_dirichlet_wall,
                       y_pos = boundary_condition_inflow_outflow,
                       x_pos = boundary_condition_inflow_outflow,
                       x_neg = boundary_condition_inflow_outflow)

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 4
basis = LobattoLegendreBasis(polydeg)

limiter_idp = SubcellLimiterIDP(equations, basis;
                                local_minmax_variables_cons = [1],
                                spec_entropy = true,
                                positivity_correction_factor = 0.1,
                                max_iterations_newton = 100,
                                bar_states = true)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

initial_refinement_level = 6
cells_per_dimension = (4 * 2^initial_refinement_level, 2^initial_refinement_level)
coordinates_min = (0.0, 0.0)
coordinates_max = (4.0, 1.0)
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                      periodicity = false)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(), BoundsCheckCallback(save_errors = false))

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  callback = callbacks);
summary_callback() # print the timer summary
