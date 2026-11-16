# Linear advection on a periodic curved mesh with the continuous Galerkin spectral
# element method (CGSEM), cf. examples/p4est_2d_dgsem/elixir_advection_basic.jl

using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create CGSEM solver with polynomial degree = 3. The surface flux enters only through
# weakly imposed boundary conditions, so it is unused on this periodic mesh.
solver = CGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

# Curved mapping of [-1, 1]^2 onto itself that is the identity on the boundary and hence
# compatible with periodic boundaries
function mapping(xi, eta)
    x = xi + 0.1 * sin(pi * xi) * sin(pi * eta)
    y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
    return SVector(x, y)
end

trees_per_dimension = (8, 8)

# Create a conforming, periodic P4estMesh with 8 x 8 trees and 16 x 16 elements
mesh = P4estMesh(trees_per_dimension, polydeg = 3, mapping = mapping,
                 initial_refinement_level = 1, periodicity = true)

# A semidiscretization collects data structures and functions for the spatial discretization
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
sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);
