
using Convex, ECOS
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = -1.0 # minimum coordinate
coordinates_max = 1.0 # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 20.0
tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 2.5)

alive_callback = AliveCallback(alive_interval = analysis_interval)

save_solution = SaveSolutionCallback(dt = 0.1,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback,
                        alive_callback,
                        save_solution,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# Construct second order paired explicit Runge-Kutta method with 6 stages for given simulation setup.
# Pass `tspan` to calculate maximum time step allowed for the bisection algorithm used 
# in calculating the polynomial coefficients in the ODE algorithm.
ode_algorithm = Trixi.PairedExplicitRK2(6, tspan, semi)

sol = Trixi.solve(ode, ode_algorithm,
                  dt = 1.0, # Manual time step value, will be overwritten by the stepsize_callback when it is specified.
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
