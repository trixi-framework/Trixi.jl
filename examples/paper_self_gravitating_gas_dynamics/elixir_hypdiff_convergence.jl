
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

equations = HyperbolicDiffusionEquations2D()

initial_condition = initial_condition_poisson_nonperiodic
# 1 => -x, 2 => +x, 3 => -y, 4 => +y as usual for orientations
boundary_conditions = (x_neg = boundary_condition_poisson_nonperiodic,
                       x_pos = boundary_condition_poisson_nonperiodic,
                       y_neg = boundary_condition_periodic,
                       y_pos = boundary_condition_periodic)

polydeg = 3
surface_flux = flux_lax_friedrichs
solver = DGSEM(polydeg, surface_flux)

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 2,
                n_cells_max = 30_000,
                periodicity = (false, true))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_poisson_nonperiodic,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

resid_tol = 1.0e-10
steady_state_callback = SteadyStateCallback(abstol = resid_tol, reltol = 0.0)

stepsize_callback = StepsizeCallback(cfl = 1.0)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

analysis_interval = 500
alive_callback = AliveCallback(analysis_interval = analysis_interval)
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy, energy_total))

callbacks = CallbackSet(summary_callback, steady_state_callback, stepsize_callback,
                        save_solution,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
