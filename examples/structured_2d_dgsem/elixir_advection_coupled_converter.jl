using OrdinaryDiffEq
using Trixi

###############################################################################
# Coupled semidiscretization of two linear advection systems using converter functions such that
# the upper half of the domain is coupled periodically, while the lower half is not coupled
# and any incoming wave is completely absorbed.
#
# In this elixir, we have a square domain that is divided into a left half and a right half. On each
# half of the domain, a completely independent SemidiscretizationHyperbolic is created for the
# linear advection equations. The two systems are coupled in the x-direction and have periodic
# boundaries in the y-direction. For a high-level overview, see also the figure below:
#
# (-1,  1)                                   ( 1,  1)
#     ┌────────────────────┬────────────────────┐
#     │    ↑ periodic ↑    │    ↑ periodic ↑    │
#     │                    │                    │
#     │                    │                    │
#     │     =========      │     =========      │
#     │     system #1      │     system #2      │
#     │     =========      │     =========      │
#     │                    │                    │
#     │                    │                    │
#     │                    │                    │
#     │                    │                    │
#     │         coupled -->│<-- coupled         │
#     │                    │                    │
#     │<-- coupled         │         coupled -->│
#     │                    │                    │
#     │                    │                    │
#     │    ↓ periodic ↓    │    ↓ periodic ↓    │
#     └────────────────────┴────────────────────┘
# (-1, -1)                                   ( 1, -1)

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# First mesh is the left half of a [-1,1]^2 square
coordinates_min1 = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max1 = (0.0, 1.0) # maximum coordinates (max(x), max(y))

# Define identical resolution as a variable such that it is easier to change from `trixi_include`
cells_per_dimension = (8, 16)

cells_per_dimension1 = cells_per_dimension

mesh1 = StructuredMesh(cells_per_dimension1, coordinates_min1, coordinates_max1)

coupling_function1 = coupling_converter_heaviside_2d(-0.5, 1.0, 1.0, equations)
# The user can define their own coupling functions.
# coupling_function1 = (x, u) -> (sign(x[2] - 0.0)*0.1 + 1.0)/1.1 * u

boundary_conditions_x_neg1 = BoundaryConditionCoupled(2, (:end, :i_forward), Float64,
                                                      coupling_function1)
boundary_conditions_x_pos1 = BoundaryConditionCoupled(2, (:begin, :i_forward), Float64,
                                                      coupling_function1)

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = (
                                                            # Connect left boundary with right boundary of right mesh
                                                            x_neg = boundary_conditions_x_neg1,
                                                            # Connect right boundary with left boundary of right mesh
                                                            x_pos = boundary_conditions_x_pos1,
                                                            y_neg = boundary_condition_periodic,
                                                            y_pos = boundary_condition_periodic))

# Second mesh is the right half of a [-1,1]^2 square
coordinates_min2 = (0.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max2 = (1.0, 1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension2 = cells_per_dimension

mesh2 = StructuredMesh(cells_per_dimension2, coordinates_min2, coordinates_max2)

coupling_function2 = coupling_converter_heaviside_2d(-0.5, 1.0, 1.0, equations)
# coupling_function2 = (x, u) -> (sign(x[2] - 0.0)*0.1 + 1.0)/1.1 * u

boundary_conditions_x_neg2 = BoundaryConditionCoupled(1, (:end, :i_forward), Float64,
                                                      coupling_function2)
boundary_conditions_x_pos2 = BoundaryConditionCoupled(1, (:begin, :i_forward), Float64,
                                                      coupling_function2)

semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = (
                                                            # Connect left boundary with right boundary of left mesh
                                                            x_neg = boundary_conditions_x_neg2,
                                                            # Connect right boundary with left boundary of left mesh
                                                            x_pos = boundary_conditions_x_pos2,
                                                            y_neg = boundary_condition_periodic,
                                                            y_pos = boundary_condition_periodic))

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupled(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 2.0
ode = semidiscretize(semi, (0.0, 20.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 1,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
