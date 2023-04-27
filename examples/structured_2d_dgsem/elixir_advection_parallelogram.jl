# This elixir transforms the setup of elixir_advection_basic to a parallelogram.
# The nodal values of the initial condition and the exact solution are the same as
# in elixir_advection_basic. 
# However, on this non-rectangular mesh, the metric terms are non-trivial.
# The same errors as with elixir_advection_basic are expected.

using OrdinaryDiffEq
using Trixi


# initial_condition_convergence_test transformed to the parallelogram
function initial_condition_parallelogram(x, t, equation::LinearScalarAdvectionEquation2D)
  # Transform back to unit square
  x_transformed = SVector(x[1] - x[2], x[2])
  a = equation.advection_velocity
  a_transformed = SVector(a[1] - a[2], a[2])

  # Store translated coordinate for easy use of exact solution
  x_translated = x_transformed - a_transformed * t

  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_translated))
  return SVector(scalar)
end

###############################################################################
# semidiscretization of the linear advection equation

# Transformed advection_velocity = (0.2, -0.7) by transformation mapping
advection_velocity = (-0.5, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

# Define faces for a parallelogram that looks like this
#
#             (0,1) __________ (2, 1)
#                ⟋         ⟋
#             ⟋         ⟋
#          ⟋         ⟋
# (-2,-1) ‾‾‾‾‾‾‾‾‾‾ (0,-1)
mapping(xi, eta) = SVector(xi + eta, eta)

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements, periodic in both dimensions
mesh = StructuredMesh(cells_per_dimension, mapping; periodicity=(true, true))

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_parallelogram, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
