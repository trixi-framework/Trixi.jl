
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0, 1.0)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(3, flux_lax_friedrichs)

# Mapping described in https://arxiv.org/abs/2012.12040,
# but on [-1,1]^3 instead of [0,3]^3
function mapping(xi, eta, zeta)
  y = eta + 0.25 * cos(1.5 * pi * xi) * cos(0.5 * pi * eta) * cos(0.5 * pi * zeta)

  x = xi + 0.25 * cos(0.5 * pi * xi) * cos(2 * pi * y) * cos(0.5 * pi * zeta)

  z = zeta + 0.25 * cos(0.5 * pi * x) * cos(pi * y) * cos(0.5 * pi * zeta)
  
  return SVector(x, y, z)
end

cells_per_dimension = (8, 8, 8)

# Create curved mesh with 8 x 8 x 8 elements
mesh = CurvedMesh(cells_per_dimension, mapping)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_constant, solver)


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

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=2.0)

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
