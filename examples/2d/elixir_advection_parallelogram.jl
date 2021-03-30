
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (2.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(3, flux_lax_friedrichs)

# Define faces for a parallelogram that looks like this
#
#             (0,1) __________ (2, 1)
#                ⟋         ⟋
#             ⟋         ⟋
#          ⟋         ⟋
# (-2,-1) ‾‾‾‾‾‾‾‾‾‾ (0,-1)
f1(s) = SVector(s-1,  s)
f2(s) = SVector(s+1,  s)
f3(s) = SVector(s-1, -1)
f4(s) = SVector(s+1,  1)

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements
mesh = CurvedMesh(cells_per_dimension, (f1, f2, f3, f4), Float64)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_parallelogram, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
