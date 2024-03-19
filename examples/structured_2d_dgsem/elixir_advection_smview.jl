using OrdinaryDiffEq
using Trixi

###############################################################################
# Coupled semidiscretization of two linear advection systems using converter functions
# and mesh views for the semidiscretizations. First we define a parent mesh
# for the entire physical domain, then we define the two mesh views on this parent.
#
# In this elixir, we have a square domain that is divided into left and right subdomains.
# On each half of the domain, a completely independent `SemidiscretizationHyperbolic`
# is created for the linear advection equations. The two systems are coupled in the
# x-direction.
# For a high-level overview, see also the figure below:
#
# (-1,  1)                                   ( 1,  1)
#     ┌────────────────────┬────────────────────┐
#     │    ↑ periodic ↑    │    ↑ periodic ↑    │
#     │                    │                    │
#     │     =========      │     =========      │
#     │     system #1      │     system #2      │
#     │     =========      │     =========      │
#     │                    │                    │
#     │<-- coupled         │<-- coupled         │
#     │         coupled -->│         coupled -->│
#     │                    │                    │
#     │    ↓ periodic ↓    │    ↓ periodic ↓    │
#     └────────────────────┴────────────────────┘
# (-1, -1)                                   ( 1, -1)

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Domain size of the parent mesh.
coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

# Cell dimensions of the parent mesh.
cells_per_dimension = (16, 16)

# Create parent mesh with 16 x 16 elements
parent_mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

# Create the two mesh views, each of which takes half of the parent mesh.
mesh1 = StructuredMeshView(parent_mesh; indices_min = (1, 1), indices_max = (8, 16))
mesh2 = StructuredMeshView(parent_mesh; indices_min = (9, 1), indices_max = (16, 16))

# The coupling function is simply the identity, as we are dealing with two identical systems.
coupling_function = (x, u, equations_other, equations_own) -> u

# Define the coupled boundary conditions
# The indices (:end, :i_forward) and (:begin, :i_forward) denote the interface indexing.
# For a system with coupling in x and y see examples/structured_2d_dgsem/elixir_advection_coupled.jl.
boundary_conditions1 = (
                        # Connect left boundary with right boundary of left mesh
                        x_neg = BoundaryConditionCoupled(2, (:end, :i_forward), Float64,
                                                         coupling_function),
                        x_pos = BoundaryConditionCoupled(2, (:begin, :i_forward), Float64,
                                                         coupling_function),
                        y_neg = boundary_condition_periodic,
                        y_pos = boundary_condition_periodic)
boundary_conditions2 = (
                        # Connect left boundary with right boundary of left mesh
                        x_neg = BoundaryConditionCoupled(1, (:end, :i_forward), Float64,
                                                         coupling_function),
                        x_pos = BoundaryConditionCoupled(1, (:begin, :i_forward), Float64,
                                                         coupling_function),
                        y_neg = boundary_condition_periodic,
                        y_pos = boundary_condition_periodic)

# A semidiscretization collects data structures and functions for the spatial discretization
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
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval = analysis_interval)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
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
            dt = 5.0e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

# Print the timer summary
summary_callback()
