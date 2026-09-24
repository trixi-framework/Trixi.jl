using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

# This example showcases the second-order BlockFV reconstruction mode.
# `reconstruction_O2_full` reconstructs on every FV cell, including those next
# to element faces (unlimited central slope there). This gives it 
# second-order accuracy on this problem.
# `monotonized_central` is used so that the limiter is lower dissipation
solver = BlockFV(n_nodes = 4, surface_flux = flux_hllc,
                 reconstruction_mode = reconstruction_O2_full,
                 slope_limiter = monotonized_central)

coordinates_min = (0.0,)
coordinates_max = (2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max, initial_refinement_level = 4,
                periodicity = false)

# Assign a single boundary condition to all boundaries
boundary_conditions = BoundaryConditionDirichlet(initial_condition)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 2.0
tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)
summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 100)
stepsize_callback = StepsizeCallback(cfl = 0.5)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, ORK256(), dt = 1, save_everystep = false, callback = callbacks);
