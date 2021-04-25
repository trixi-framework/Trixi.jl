
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0, 1.0)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

indices_left  = (1,     :i, :j)
indices_right = (:end,  :i, :j)

boundary_1_left   = Trixi.BoundaryConditionCoupled(2, 1, indices_right, 3, Float64)
boundary_1_right  = Trixi.BoundaryConditionCoupled(2, 1, indices_left,  3, Float64)
boundary_2_left   = Trixi.BoundaryConditionCoupled(1, 1, indices_right, 3, Float64)
boundary_2_right  = Trixi.BoundaryConditionCoupled(1, 1, indices_left,  3, Float64)

coordinates_min = (-1.0, -1.0, -2.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = ( 0.0,  1.0, 2.0) # maximum coordinates (max(x), max(y), max(z))
cells_per_dimension = (4, 10, 16)

mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                     boundary_conditions=(boundary_1_left, boundary_1_right,
                                                          boundary_condition_periodic, boundary_condition_periodic,
                                                          boundary_condition_periodic, boundary_condition_periodic))

coordinates_min = ( 0.0, -1.0, -2.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = ( 1.0,  1.0, 2.0) # maximum coordinates (max(x), max(y), max(z))
cells_per_dimension = (4, 10, 16)

mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                     boundary_conditions=(boundary_2_left, boundary_2_right,
                                                          boundary_condition_periodic, boundary_condition_periodic,
                                                          boundary_condition_periodic, boundary_condition_periodic))

semi = SemidiscretizationHyperbolicCoupled((semi1, semi2))
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
stepsize_callback = StepsizeCallback(cfl=1.2)

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
