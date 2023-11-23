# The same setup as tree_2d_dgsem/elixir_advection_basic.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 2 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # maximum coordinates (max(x), max(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (16, 16)
# cells_per_dimension = (32, 16)

# Create curved mesh with 16 x 16 elements
parent = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

#mesh = StructuredMeshView(parent; index_min = (1, 1), index_max = (16, 16))
#semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

mesh1 = StructuredMeshView(parent; index_min = (1, 1), index_max = (8, 16))
mesh2 = StructuredMeshView(parent; index_min = (9, 1), index_max = (16, 16))

# Define the coupled boundary conditions
boundary_conditions1 = (
                        # Connect left boundary with right boundary of left mesh
                        x_neg = BoundaryConditionCoupled(2, (:end, :i_forward), Float64),
                        x_pos = BoundaryConditionCoupled(2, (:begin, :i_forward), Float64),
                        #                         x_neg = boundary_condition_periodic,
                        #                         x_pos = boundary_condition_periodic,
                        y_neg = boundary_condition_periodic,
                        y_pos = boundary_condition_periodic)
boundary_conditions2 = (
                        # Connect left boundary with right boundary of left mesh
                        x_neg = BoundaryConditionCoupled(1, (:end, :i_forward), Float64),
                        x_pos = BoundaryConditionCoupled(1, (:begin, :i_forward), Float64),
                        #                         x_neg = boundary_condition_periodic,
                        #                         x_pos = boundary_condition_periodic,
                        y_neg = boundary_condition_periodic,
                        y_pos = boundary_condition_periodic)

# A semidiscretization collects data structures and functions for the spatial discretization
# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)
semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = boundary_conditions1)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = boundary_conditions2)
semi = SemidiscretizationCoupled(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
#analysis_callback = AnalysisCallback(semi, interval=10)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
# save_solution = SaveSolutionCallback(interval=100,
#                                      solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
# stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
# callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)
callbacks = CallbackSet(summary_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 5.0e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

errors_ref = (l2 = [8.312427642603623e-6], linf = [6.626865824577166e-5])

# Print the timer summary
summary_callback()
