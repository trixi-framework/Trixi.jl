using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# Simplest coupled setup consisting of two non-trivial mesh views.

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Define the physical domain for the parent mesh.
coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (8, 8)

# Create parent P4estMesh with 8 x 8 trees and 8 x 8 elements.
# The mesh is periodic so that the outer faces of the ring view connect to the
# opposite ring faces as regular internal interfaces, rather than appearing as
# physical domain boundaries. This makes the combined coupled system truly
# double-periodic: the solution travels across the center square into the ring
# and wraps around.
parent_mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                        coordinates_min = coordinates_min,
                        coordinates_max = coordinates_max,
                        initial_refinement_level = 0,
                        periodicity = true)

# Split elements into a center square (|x|<0.5 and |y|<0.5) and an outer ring.
# Use element center coordinates so the split works at any initial_refinement_level.
semi_parent = SemidiscretizationHyperbolic(parent_mesh, equations,
                                           initial_condition_convergence_test, solver,
                                           boundary_conditions = (;))
cache_parent = semi_parent.cache

cell_ids1 = Int[]  # outer ring
cell_ids2 = Int[]  # inner square
for element in 1:Trixi.ncells(parent_mesh)
    x_c = cache_parent.elements.node_coordinates[1, 2, 2, element]
    y_c = cache_parent.elements.node_coordinates[2, 2, 2, element]
    if abs(x_c) < 0.5 && abs(y_c) < 0.5
        push!(cell_ids2, element)
    else
        push!(cell_ids1, element)
    end
end

mesh1 = P4estMeshView(parent_mesh, cell_ids1)
mesh2 = P4estMeshView(parent_mesh, cell_ids2)

# Define trivial coupling functions (identity, same equation on both sides).
coupling_functions = Array{Function}(undef, 2, 2)
coupling_functions[1, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[1, 2] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 1] = (x, u, equations_other, equations_own) -> u
coupling_functions[2, 2] = (x, u, equations_other, equations_own) -> u

boundary_conditions = (; x_neg = BoundaryConditionCoupledP4est(coupling_functions),
                       y_neg = BoundaryConditionCoupledP4est(coupling_functions),
                       y_pos = BoundaryConditionCoupledP4est(coupling_functions),
                       x_pos = BoundaryConditionCoupledP4est(coupling_functions))

semi1 = SemidiscretizationHyperbolic(mesh1, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = boundary_conditions)
semi2 = SemidiscretizationHyperbolic(mesh2, equations, initial_condition_convergence_test,
                                     solver,
                                     boundary_conditions = boundary_conditions)

# Create a semidiscretization that bundles semi1 and semi2
semi = SemidiscretizationCoupledP4est(semi1, semi2; coupling_functions = coupling_functions)

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
analysis_callback = AnalysisCallbackCoupledP4est(semi, analysis_callback1,
                                                 analysis_callback2)

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
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
