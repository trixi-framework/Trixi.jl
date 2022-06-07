using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection-diffusion equation

equations = CompressibleEulerEquations2D(1.4)
# Note: If you change the Navier-Stokes parameters here, also change them in the initial condition
# I really do not like this structure but it should work for now
equations_parabolic = CompressibleNaiverStokes2D(1.4,  # gamma
                                                 1000, # Reynolds number
                                                 0.72, # Prandtl number
                                                 0.5,  # free-stream Mach number
                                                 1.0,  # thermal diffusivity
                                                 equations)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Define initial condition
# Note: If you change the diffusion parameter here, also change it in the parabolic equation definition
function initial_condition_diffusive_convergence_test(x, t, equation::LinearScalarAdvectionEquation2D)
  # Store translated coordinate for easy use of exact solution
  x_trans = x - equation.advection_velocity * t

  # @unpack nu = equation
  nu = 5.0e-2
  c = 1.0
  A = 0.5
  L = 2
  f = 1/L
  omega = 2 * pi * f
  scalar = c + A * sin(omega * sum(x_trans)) * exp(-2 * nu * omega^2 * t)
  return SVector(scalar)
end

initial_condition = initial_condition_diffusive_convergence_test



# TODO: Wasn't sure of the call structure, but this should be what we need
function boundary_condition_no_slip_adiabatic_wall(u_inner, orientation, direction,
                                                   x, t, surface_flux_function,
                                                   equations::CompressibleEulerEquations2D)
  u_boundary = SVector(u_inner[1], -u_inner[2],  -u_inner[3], u_inner[4])

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


# TODO: Wasn't sure of the call structure, but this should be what we need
function boundary_condition_no_slip_adiabatic_wall_neumann(grad_u_inner, orientation, direction,
                                                           x, t, surface_flux_function,
                                                           equations::CompressibleNaiverStokes2D)
  # Copy the inner gradients to an external state array
  grad_u_boundary .= grad_u_inner

  # Project out the appropriate temperature gradient pieces. Wasn't sure of array shape
  if orientation == 1
    grad_u_norm = grad_u[1,3] # temperature gradient in x-direction
    u_x_tangent = grad_u[1,3] - grad_u_norm
    u_y_tangent = grad_u[2,3]

    # Update the external state gradients
    grad_u_boundary[1,3] = u_x_tangent - grad_u_norm
    grad_u_boundary[2,3] = u_y_tangent
  else # orientation == 2
    grad_u_norm = grad_u[2,3] # temperature gradient in y-direction
    u_x_tangent = grad_u[1,3]
    u_y_tangent = grad_u[2,3] - grad_u_norm

    # Update the external state gradients
    grad_u_boundary[1,3] = u_x_tangent
    grad_u_boundary[2,3] = u_y_tangent - grad_u_norm
  end

  # Calculate boundary flux (just averages so has no orientation I think)
  flux = surface_flux_function(grad_u_inner, grad_u_boundary, orientation, equations)

  return flux
end


# Below is my best guess as to how to set periodic in x direction and walls
# in the y direcitons
boundary_condition= (
                     x_neg=boundary_condition_periodic,
                     x_pos=boundary_condition_periodic,
                     y_neg=boundary_condition_no_slip_adiabatic_wall,
                     y_pos=boundary_condition_no_slip_adiabatic_wall,
                    )
boundary_conditions_parabolic = (
                                 x_neg=boundary_condition_periodic,
                                 x_pos=boundary_condition_periodic,
                                 y_neg=boundary_condition_no_slip_adiabatic_wall_neumann,
                                 y_pos=boundary_condition_no_slip_adiabatic_wall_neumann,
                                )

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
ode = semidiscretize(semi, tspan);

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
time_int_tol = 1.0e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol,
            save_everystep=false, callback=callbacks)

# Print the timer summary
summary_callback()
