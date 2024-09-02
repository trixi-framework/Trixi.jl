using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the linear advection equation

a = 0.1
b = 1.0
equationsfull = LinearAcousticAdvectionEquation1D(a, b)
equationsslow = LinearAcousticAdvectionSlowEquation1D(a, b)
equationsfast = LinearAcousticAdvectionFastEquation1D(a, b)
# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_rusanov)

coordinates_min = (0.0,) # minimum coordinate
coordinates_max = (1.0,) # maximum coordinate
cells_per_dimension = (128,)

# Create curved mesh with 16 cells
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)
boundary_conditions1 = boundary_condition_periodic
boundary_conditions2 = boundary_condition_periodic
# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicSplit(mesh, (equationsslow, equationsfast),
                                         initial_condition_fast_slow, solver, solver;
                                         boundary_conditions = (boundary_conditions1,
                                                                boundary_conditions2))
###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
# ode = semidiscretizesplit(semifull,semislow, semifast, (0.0, 0.001));
ode = semidiscretize(semi, (0.0, 1.0));
# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.1)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = Trixi.solve(ode, Trixi.SimpleIMEX(),
                  dt = 0.0195, # solve needs some value here but it will be overwritten by the stepsize_callback
                  save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
