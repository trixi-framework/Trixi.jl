using OrdinaryDiffEq
using Trixi

###############################################################################
# Coupled semidiscretization of four linear advection systems using converter functions such that
# they are also coupled across the domain boundaries to generate a periodic system.
#
# In this elixir, we have a square domain that is divided into a upper-left, lower-left,
# upper-right and lower-right quarter. On each quarter
# of the domain, a completely independent SemidiscretizationHyperbolic is created for the
# linear advection equations. The four systems are coupled in the x and y-direction.
# For a high-level overview, see also the figure below:
#
# (-1,  1)                                   ( 1,  1)
#     ┌────────────────────┬────────────────────┐
#     │    ↑ coupled ↑     │    ↑ coupled ↑     │
#     │                    │                    │
#     │     =========      │     =========      │
#     │     system #1      │     system #2      │
#     │     =========      │     =========      │
#     │                    │                    │
#     │<-- coupled         │<-- coupled         │
#     │         coupled -->│         coupled -->│
#     │                    │                    │
#     │    ↓ coupled ↓     │    ↓ coupled ↓     │
#     ├────────────────────┼────────────────────┤
#     │    ↑ coupled ↑     │    ↑ coupled ↑     │
#     │                    │                    │
#     │     =========      │     =========      │
#     │     system #3      │     system #4      │
#     │     =========      │     =========      │
#     │                    │                    │
#     │<-- coupled         │<-- coupled         │
#     │         coupled -->│         coupled -->│
#     │                    │                    │
#     │    ↓ coupled  ↓    │    ↓ coupled  ↓    │
#     └────────────────────┴────────────────────┘
# (-1, -1)                                   ( 1, -1)

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# This will be the number of elements for each quarter/semidiscretization.
cells_per_dimension = (8, 8)

###########
# system #1
###########

coordinates_min1 = (-1.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max1 = (0.0, 1.0) # maximum coordinates (max(x), max(y))

mesh1 = StructuredMesh(cells_per_dimension, coordinates_min1, coordinates_max1,
                       periodicity = false)

# Define the coupling functions
coupling_function12 = (x, u, equations_other, equations_own) -> u
coupling_function13 = (x, u, equations_other, equations_own) -> u

# Define the coupling boundary conditions and the system it is coupled to.
boundary_conditions_x_neg1 = BoundaryConditionCoupled(2, (:end, :i_forward), Float64,
                                                      coupling_function12)
boundary_conditions_x_pos1 = BoundaryConditionCoupled(2, (:begin, :i_forward), Float64,
                                                      coupling_function12)
boundary_conditions_y_neg1 = BoundaryConditionCoupled(3, (:i_forward, :end), Float64,
                                                      coupling_function13)
boundary_conditions_y_pos1 = BoundaryConditionCoupled(3, (:i_forward, :begin), Float64,
                                                      coupling_function13)

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = (x_neg = boundary_conditions_x_neg1,
                                                            x_pos = boundary_conditions_x_pos1,
                                                            y_neg = boundary_conditions_y_neg1,
                                                            y_pos = boundary_conditions_y_pos1))

###########
# system #2
###########

coordinates_min2 = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max2 = (1.0, 1.0) # maximum coordinates (max(x), max(y))

mesh2 = StructuredMesh(cells_per_dimension, coordinates_min2, coordinates_max2,
                       periodicity = false)

# Define the coupling functions
coupling_function21 = (x, u, equations_other, equations_own) -> u
coupling_function24 = (x, u, equations_other, equations_own) -> u

# Define the coupling boundary conditions and the system it is coupled to.
boundary_conditions_x_neg2 = BoundaryConditionCoupled(1, (:end, :i_forward), Float64,
                                                      coupling_function21)
boundary_conditions_x_pos2 = BoundaryConditionCoupled(1, (:begin, :i_forward), Float64,
                                                      coupling_function21)
boundary_conditions_y_neg2 = BoundaryConditionCoupled(4, (:i_forward, :end), Float64,
                                                      coupling_function24)
boundary_conditions_y_pos2 = BoundaryConditionCoupled(4, (:i_forward, :begin), Float64,
                                                      coupling_function24)

# A semidiscretization collects data structures and functions for the spatial discretization
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = (x_neg = boundary_conditions_x_neg2,
                                                            x_pos = boundary_conditions_x_pos2,
                                                            y_neg = boundary_conditions_y_neg2,
                                                            y_pos = boundary_conditions_y_pos2))

###########
# system #3
###########

coordinates_min3 = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max3 = (0.0, 0.0) # maximum coordinates (max(x), max(y))

mesh3 = StructuredMesh(cells_per_dimension, coordinates_min3, coordinates_max3,
                       periodicity = false)

# Define the coupling functions
coupling_function34 = (x, u, equations_other, equations_own) -> u
coupling_function31 = (x, u, equations_other, equations_own) -> u

# Define the coupling boundary conditions and the system it is coupled to.
boundary_conditions_x_neg3 = BoundaryConditionCoupled(4, (:end, :i_forward), Float64,
                                                      coupling_function34)
boundary_conditions_x_pos3 = BoundaryConditionCoupled(4, (:begin, :i_forward), Float64,
                                                      coupling_function34)
boundary_conditions_y_neg3 = BoundaryConditionCoupled(1, (:i_forward, :end), Float64,
                                                      coupling_function31)
boundary_conditions_y_pos3 = BoundaryConditionCoupled(1, (:i_forward, :begin), Float64,
                                                      coupling_function31)

# A semidiscretization collects data structures and functions for the spatial discretization
semi3 = SemidiscretizationHyperbolic(mesh3, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = (x_neg = boundary_conditions_x_neg3,
                                                            x_pos = boundary_conditions_x_pos3,
                                                            y_neg = boundary_conditions_y_neg3,
                                                            y_pos = boundary_conditions_y_pos3))

###########
# system #4
###########

coordinates_min4 = (0.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max4 = (1.0, 0.0) # maximum coordinates (max(x), max(y))

mesh4 = StructuredMesh(cells_per_dimension, coordinates_min4, coordinates_max4,
                       periodicity = false)

# Define the coupling functions
coupling_function43 = (x, u, equations_other, equations_own) -> u
coupling_function42 = (x, u, equations_other, equations_own) -> u

# Define the coupling boundary conditions and the system it is coupled to.
boundary_conditions_x_neg4 = BoundaryConditionCoupled(3, (:end, :i_forward), Float64,
                                                      coupling_function43)
boundary_conditions_x_pos4 = BoundaryConditionCoupled(3, (:begin, :i_forward), Float64,
                                                      coupling_function43)
boundary_conditions_y_neg4 = BoundaryConditionCoupled(2, (:i_forward, :end), Float64,
                                                      coupling_function42)
boundary_conditions_y_pos4 = BoundaryConditionCoupled(2, (:i_forward, :begin), Float64,
                                                      coupling_function42)

# A semidiscretization collects data structures and functions for the spatial discretization
semi4 = SemidiscretizationHyperbolic(mesh4, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = (x_neg = boundary_conditions_x_neg4,
                                                            x_pos = boundary_conditions_x_pos4,
                                                            y_neg = boundary_conditions_y_neg4,
                                                            y_pos = boundary_conditions_y_pos4))

# Create a semidiscretization that bundles all the semidiscretizations.
semi = SemidiscretizationCoupled(semi1, semi2, semi3, semi4)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 2.0
ode = semidiscretize(semi, (0.0, 2.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback3 = AnalysisCallback(semi3, interval = 100)
analysis_callback4 = AnalysisCallback(semi4, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2,
                                            analysis_callback3, analysis_callback4)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
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
