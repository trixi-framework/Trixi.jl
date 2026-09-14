using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

# Standard 2nd order block FV setup with a less dissipative reconstruction limiter
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

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, ORK256(), dt = 1, save_everystep = false, callback = callbacks);
