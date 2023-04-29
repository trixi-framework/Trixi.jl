
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection diffusion equation

advection_velocity = 0.1
equations = LinearScalarAdvectionEquation1D(advection_velocity)
diffusivity() = 0.1
equations_parabolic = LaplaceDiffusion1D(diffusivity(), equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = -pi # minimum coordinate
coordinates_max =  pi # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000, # set maximum capacity of tree data structure
                periodicity=true)

function x_trans_periodic(x, domain_length = SVector(2*pi), center = SVector(0.0))
  x_normalized = x .- center
  x_shifted = x_normalized .% domain_length
  x_offset = ((x_shifted .< -0.5*domain_length) - (x_shifted .> 0.5*domain_length)) .* domain_length
  return center + x_shifted + x_offset
end

# Define initial condition
function initial_condition_diffusive_convergence_test(x, t, equation::LinearScalarAdvectionEquation1D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x_trans_periodic(x - equation.advection_velocity * t)
  
  nu = diffusivity()
  c = 0.0
  A = 1.0
  L = 2
  f = 1/L
  omega = 1.0
  scalar = c + A * sin(omega * sum(x_trans)) * exp(-nu * omega^2 * t)
  return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test
  
# define periodic boundary conditions everywhere
boundary_conditions = boundary_condition_periodic
boundary_conditions_parabolic = boundary_condition_periodic

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), 
  initial_condition_diffusive_convergence_test, solver;
  boundary_conditions=(boundary_conditions, boundary_conditions_parabolic))


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The AliveCallback prints short status information in regular intervals
alive_callback = AliveCallback(analysis_interval=100)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
time_int_tol = 1.0e-10
time_abs_tol = 1.0e-10
sol = solve(ode, KenCarp4(autodiff=false), abstol=time_abs_tol, reltol=time_int_tol,
            save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()
