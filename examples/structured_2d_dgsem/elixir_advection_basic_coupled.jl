
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# First mesh is the left half of a [-1,1]^2 square
coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.0,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (8, 16)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                     boundary_conditions=(
                                       # Connect left boundary with right boundary of right mesh
                                       x_neg=BoundaryConditionCoupled(2, (:end, :i_forward), Float64),
                                       # Connect right boundary with left boundary of right mesh
                                       x_pos=BoundaryConditionCoupled(2, (:begin, :i_forward),  Float64),
                                       y_neg=boundary_condition_periodic,
                                       y_pos=boundary_condition_periodic))


# Second mesh is the right half of a [-1,1]^2 square
coordinates_min = (0.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (8, 16)

mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                     boundary_conditions=(
                                       # Connect left boundary with right boundary of left mesh
                                       x_neg=BoundaryConditionCoupled(1, (:end, :i_forward), Float64),
                                       # Connect right boundary with left boundary of left mesh
                                       x_pos=BoundaryConditionCoupled(1, (:begin, :i_forward),  Float64),
                                       y_neg=boundary_condition_periodic,
                                       y_pos=boundary_condition_periodic))

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupled((semi1, semi2))

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
callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback, save_solution)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
