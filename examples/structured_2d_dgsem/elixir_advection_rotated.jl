# This elixir transforms the setup of elixir_advection_basic to a rotated square.
# The nodal values of the initial condition and the exact solution are the same as
# in elixir_advection_basic. 
# However, on this rotated mesh, the metric terms are non-trivial.
# The same errors as with elixir_advection_basic are expected (except for rounding errors).

using OrdinaryDiffEq
using Trixi


# Define new structs inside a module to allow re-evaluating the file.
# This module name needs to be unique among all examples, otherwise Julia will throw warnings
# if multiple test cases using the same module name are run in the same session.
module TrixiExtensionAdvectionRotated

using Trixi

# initial_condition_convergence_test transformed to the rotated rectangle
struct InitialConditionConvergenceTestRotated
  sin_alpha::Float64
  cos_alpha::Float64
end

function InitialConditionConvergenceTestRotated(alpha)
  sin_alpha, cos_alpha = sincos(alpha)

  InitialConditionConvergenceTestRotated(sin_alpha, cos_alpha)
end

function (initial_condition::InitialConditionConvergenceTestRotated)(x, t, equation::LinearScalarAdvectionEquation2D)
  sin_ = initial_condition.sin_alpha
  cos_ = initial_condition.cos_alpha

  # Rotate back to unit square

  # Clockwise rotation by α and translation by 1
  # Multiply with [  cos(α)  sin(α);
  #                 -sin(α)  cos(α)]
  x_rot = SVector(cos_ * x[1] + sin_ * x[2], -sin_ * x[1] + cos_ * x[2])
  a = equation.advection_velocity
  a_rot = SVector(cos_ * a[1] + sin_ * a[2], -sin_ * a[1] + cos_ * a[2])

  # Store translated coordinate for easy use of exact solution
  x_trans = x_rot - a_rot * t

  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans))

  return SVector(scalar)
end

end # module TrixiExtensionAdvectionRotated

import .TrixiExtensionAdvectionRotated

###############################################################################
# semidiscretization of the linear advection equation

alpha = pi * 0.1
initial_condition = TrixiExtensionAdvectionRotated.InitialConditionConvergenceTestRotated(alpha)
sin_ = initial_condition.sin_alpha
cos_ = initial_condition.cos_alpha
T = [cos_ -sin_; sin_ cos_]

advection_velocity = Tuple(T * [0.2, -0.7])
equations = LinearScalarAdvectionEquation2D(advection_velocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

mapping(xi, eta) = T * SVector(xi, eta)

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, mapping)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


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
