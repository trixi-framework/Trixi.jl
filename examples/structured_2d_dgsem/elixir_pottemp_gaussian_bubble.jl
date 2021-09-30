
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleDryEulerEquations2D()

initial_condition = initial_condition_gaussian_bubble

boundary_conditions = (x_neg=boundary_condition_periodic,
                       x_pos=boundary_condition_periodic,
                       y_neg=boundary_condition_slip_wall,
                       y_pos=boundary_condition_slip_wall)

source_terms = source_terms_warm_bubble

###############################################################################
# Get the DG approximation space


solver = DGSEM(polydeg=3, surface_flux=flux_LMARS)


coordinates_min = (0.0, 0.0)
coordinates_max = (1500.0, 1500.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=30_000)

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.006)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1
solution_variables = cons2pot

analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=solution_variables)

stepsize_callback = StepsizeCallback(cfl=0.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
