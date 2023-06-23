using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

advection_velocity = (1.5, 1.0)
equations = LinearScalarAdvectionEquation2D(advection_velocity)
diffusivity() = 5.0e-2
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                periodicity=true,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advection_velocity * t

  nu = diffusivity()
  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
  return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh,
                                             (equations, equations_parabolic),
                                             initial_condition, solver;
                                             boundary_conditions=(boundary_conditions,
                                                                  boundary_conditions_parabolic))


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.5
tspan = (0.0, 1.5)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=analysis_interval)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-11
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)

# Print the timer summary
summary_callback()
