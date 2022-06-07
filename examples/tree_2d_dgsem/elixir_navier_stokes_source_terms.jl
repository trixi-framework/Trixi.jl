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
# Note: If you change the parameters here, also change it in the corresponding source terms
function initial_condition_navier_stokes_convergence_test(x, t, equation::CompressibleEulerEquations2D)

  # Amplitude and shift
  A    = 0.5
  c    = 2.0

  # convenience values for trig. functions
  pi_x = pi * x[1]
  pi_y = pi * x[2]
  pi_t = pi * t

  rho = c + A * sin(pi_x) * cos(pi_y) * cos(pi_t)
  v1  = sin(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0)) ) * cos(pi_t)
  v2  = v1
  p   = rho^2

  return prim2cons(SVector(rho, v1, v2, p), equations)
end


initial_condition = initial_condition_navier_stokes_convergence_test


@inline function initial_condition_navier_stokes_convergence_test(u, x, t,
                                                                  equations::CompressibleNaiverStokes2D)
  # Same settings as in `initial_condition`
  # Amplitude and shift
  A    = 0.5
  c    = 2.0

  # convenience values for trig. functions
  pi_x = pi * x[1]
  pi_y = pi * x[2]
  pi_t = pi * t

  # compute the manufactured solution and all necessary derivatives
  rho    = c +   A * sin(pi_x) * cos(pi_y) * cos(pi_t)
  rho_t  = -pi * A * sin(pi_x) * cos(pi_y) * sin(pi_t)
  rho_x  =  pi * A * cos(pi_x) * cos(pi_y) * cos(pi_t)
  rho_y  = -pi * A * sin(pi_x) * sin(pi_y) * cos(pi_t)
  rho_xx = -pi * pi * A * sin(pi_x) * cos(pi_y) * cos(pi_t)
  rho_yy = -pi * pi * A * sin(pi_x) * cos(pi_y) * cos(pi_t)

  v1    =       sin(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * cos(pi_t)
  v1_t  = -pi * sin(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * sin(pi_t)
  v1_x  =  pi * cos(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * cos(pi_t)
  v1_y  =       sin(pi_x) * (A * log(y + 2.0) * exp(-A * (y - 1.0)) + (1.0 - exp(-A * (y - 1.0))) / (y + 2.0)) * cos(pi_t)
  v1_xx = -pi * pi * sin(pi_x) * log(y + 2.0) * (1.0 - exp(-A * (y - 1.0))) * cos(pi_t)
  v1_xy =  pi * cos(pi_x) * (A * log(y + 2.0) * exp(-A * (y - 1.0)) + (1.0 - exp(-A * (y - 1.0))) / (y + 2.0)) * cos(pi_t)
  v1_yy = (sin(pi_x) * ( 2.0 * A * exp(-A * (y - 1.0)) / (y + 2.0)
                         - A * A * log(y + 2.0) * exp(-A * (y - 1.0))
                         - (1.0 - exp(-A * (y - 1.0))) / ((y + 2.0) * (y + 2.0))) * cos(pi_t))
  v2    = v1
  v2_t  = v1_t
  v2_x  = v1_x
  v2_y  = v1_y
  v2_xx = v1_xx
  v2_xy = v1_xy
  v2_yy = v1_yy

  p    = rho * rho
  p_t  = 2.0 * rho * rho_t
  p_x  = 2.0 * rho * rho_x
  p_y  = 2.0 * rho * rho_y
  p_xx = 2.0 * rho * rho_xx + 2.0 * rho_x * rho_x
  p_yy = 2.0 * rho * rho_yy + 2.0 * rho_y * rho_y

  # Note this simplifies slightly because the ansatz assumes that v1 = v2
  E   = p * equations.inv_gamma_minus_one + 0.5 * rho * (v1^2 + v2^2)
  E_t = p_t * equations.inv_gamma_minus_one + rho_t * v1^2 + 2.0 * rho * v1 * v1_t
  E_x = p_x * equations.inv_gamma_minus_one + rho_x * v1^2 + 2.0 * rho * v1 * v1_x
  E_y = p_y * equations.inv_gamma_minus_one + rho_y * v1^2 + 2.0 * rho * v1 * v1_y

  # Some convenience constants
  T_const = equations.gamma * equations.inv_gamma_minus_one * equations.kappa / equations.Pr
  inv_Re = 1.0 / equations.Re
  inv_rho_cubed = 1.0 / (rho^3)

  # compute the source terms
  # density equation
  du1 = rho_t + rho_x * v1 + rho * v1_x + rho_y * v2 + rho * v2_y

  # x-momentum equation
  du2 = ( rho_t * v1 + rho * v1_t + p_x + rho_x * v1^2
                                        + 2.0   * rho  * v1 * v1_x
                                        + rho_y * v1   * v2
                                        + rho   * v1_y * v2
                                        + rho   * v1   * v2_y
    # stress tensor from x-direction
                      - 4.0 / 3.0 * v1_xx * inv_Re
                      + 2.0 / 3.0 * v2_xy * inv_Re
                      - v1_yy             * inv_Re
                      - v2_xy             * inv_Re )
  # y-momentum equation
  du3 = ( rho_t * v2 + rho * v2_t + p_y + rho_x * v1    * v2
                                        + rho   * v1_x  * v2
                                        + rho   * v1    * v2_x
                                        +         rho_y * v2^2
                                        + 2.0   * rho   * v2 * v2_y
    # stress tensor from y-direction
                      - v1_xy             * inv_Re
                      - v2_xx             * inv_Re
                      - 4.0 / 3.0 * v2_yy * inv_Re
                      + 2.0 / 3.0 * v1_xy * inv_Re )
  # total energy equation
  du4 = ( E_t + v1_x * (E + p) + v1 * (E_x + p_x)
              + v2_y * (E + p) + v2 * (E_y + p_y)
    # stress tensor and temperature gradient terms from x-direction
                                - 4.0 / 3.0 * v1_xx * v1   * inv_Re
                                + 2.0 / 3.0 * v2_xy * v1   * inv_Re
                                - 4.0 / 3.0 * v1_x  * v1_x * inv_Re
                                + 2.0 / 3.0 * v2_y  * v1_x * inv_Re
                                - v1_xy     * v2           * inv_Re
                                - v2_xx     * v2           * inv_Re
                                - v1_y      * v2_x         * inv_Re
                                - v2_x      * v2_x         * inv_Re
         - T_const * inv_rho_cubed * (        p_xx * rho   * rho
                                      - 2.0 * p_x  * rho   * rho_x
                                      + 2.0 * p    * rho_x * rho_x
                                      -       p    * rho   * rho_xx ) * inv_Re
    # stress tensor and temperature gradient terms from y-direction
                                - v1_yy     * v1           * inv_Re
                                - v2_xy     * v1           * inv_Re
                                - v1_y      * v1_y         * inv_Re
                                - v2_x      * v1_y         * inv_Re
                                - 4.0 / 3.0 * v2_yy * v2   * inv_Re
                                + 2.0 / 3.0 * v1_xy * v2   * inv_Re
                                - 4.0 / 3.0 * v2_y  * v2_y * inv_Re
                                + 2.0 / 3.0 * v1_x  * v2_y * inv_Re
         - T_const * inv_rho_cubed * (        p_yy * rho   * rho
                                      - 2.0 * p_y  * rho   * rho_y
                                      + 2.0 * p    * rho_y * rho_y
                                      -       p    * rho   * rho_yy ) * inv_Re )

  return SVector(du1, du2, du3, du4)
end


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
