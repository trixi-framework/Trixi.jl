
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear hyperbolic advection diffusion equation

advectionvelocity = 1.0
equations = HyperbolicAdvectionDiffusionEquations1D(advectionvelocity, nu=1.0)
# equations = HyperbolicAdvectionDiffusionEquations1D(advectionvelocity, nu=inv(2.0*pi))

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(3, flux_lax_friedrichs)

coordinates_min = 0 # minimum coordinate
coordinates_max = 1 # maximum coordinate

# Create a uniformely refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=5,
                n_cells_max=30_000,
                periodicity=false) # set maximum capacity of tree data structure


initial_condition = initial_condition_sin_nonperiodic
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_sin_nonperiodic)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 5.0
ode = semidiscretize(semi, (0.0, 5.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

resid_tol = 5.0e-12
steady_state_callback = SteadyStateCallback(abstol=resid_tol, reltol=0.0)

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.9)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, steady_state_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve(ode, Trixi.HypDiffN3Erk3Sstar52(),
                  dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep=false, callback=callbacks);

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
