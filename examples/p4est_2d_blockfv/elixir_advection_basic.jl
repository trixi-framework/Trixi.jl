# The same setup as tree_2d_blockfv/elixir_advection_basic.jl
# to verify the P4estMesh implementation against TreeMesh

using OrdinaryDiffEqLowOrderRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create a block finite volume solver with n_nodes volumes per coordinate direction
# in each cell of the mesh and the (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = BlockFV(n_nodes = 4, surface_flux = flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)

trees_per_dimension = (8, 8)

# Create P4estMesh with 8 x 8 trees and periodic boundaries
mesh = P4estMesh(trees_per_dimension, polydeg = 3,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                 initial_refinement_level = 1,
                 periodicity = true)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver;
                                    boundary_conditions = boundary_condition_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval = 100)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.9)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, Euler();
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
