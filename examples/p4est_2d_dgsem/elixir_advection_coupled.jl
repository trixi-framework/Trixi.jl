using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Simplest coupled setup consisting of two non-trivial mesh views.

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (8, 8)

# Create parent P4estMesh with 8 x 8 trees and 8 x 8 elements
# Since we couple through the boundaries, the periodicity does not matter here,
# but it is to trigger parts of the code for the test.
parent_mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                        coordinates_min = coordinates_min,
                        coordinates_max = coordinates_max,
                        initial_refinement_level = 0,
                        periodicity = false)

# Define the mesh views.
cell_ids1 = vcat((1:18), (23:26), (31:34), (39:42), (47:64))
mesh1 = P4estMeshView(parent_mesh, cell_ids1)
cell_ids2 = vcat((19:22), (27:30), (35:38), (43:46))
mesh2 = P4estMeshView(parent_mesh, cell_ids2)

# Define a trivial coupling function.
coupling_function = (x, u, equations_other, equations_own) -> u

boundary_conditions = Dict(:x_neg => BoundaryConditionCoupledP4est(coupling_function),
                           :y_neg => BoundaryConditionCoupledP4est(coupling_function),
                           :y_pos => BoundaryConditionCoupledP4est(coupling_function),
                           :x_pos => BoundaryConditionCoupledP4est(coupling_function))

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = boundary_conditions)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = boundary_conditions)

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupledP4est(semi1, semi2)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 2.0
ode = semidiscretize(semi, (0.0, 2.0))

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
# We require this definition for the test, even though we don't use it in the CallbackSet.
analysis_callback1 = AnalysisCallback(semi1, interval = 100)
analysis_callback2 = AnalysisCallback(semi2, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback1, analysis_callback2)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, save_solution, stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
