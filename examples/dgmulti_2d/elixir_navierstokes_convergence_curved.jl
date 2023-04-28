using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

prandtl_number() = 0.72
mu() = 0.01

equations = CompressibleEulerEquations2D(1.4)
# Note: If you change the Navier-Stokes parameters here, also change them in the initial condition
# I really do not like this structure but it should work for now
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu=mu(), Prandtl=prandtl_number(),
                                                          gradient_variables=GradientVariablesPrimitive())

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

top_bottom(x, tol=50*eps()) = abs(abs(x[2]) - 1) < tol
is_on_boundary = Dict(:top_bottom => top_bottom)

function mapping(xi, eta)
  x = xi  + 0.1 * sin(pi * xi) * sin(pi * eta)
  y = eta + 0.1 * sin(pi * xi) * sin(pi * eta)
  return SVector(x, y)
end
cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension, mapping; periodicity=(true, false), is_on_boundary)

# This initial condition is taken from `examples/dgmulti_2d/elixir_navierstokes_convergence.jl`

# Note: the initial condition cannot be specialized to `CompressibleNavierStokesDiffusion2D`
#       since it is called by both the parabolic solver (which passes in `CompressibleNavierStokesDiffusion2D`)
#       and by the initial condition (which passes in `CompressibleEulerEquations2D`).
# This convergence test setup was originally derived by Andrew Winters (@andrewwinters5000)
function initial_condition_navier_stokes_convergence_test(x, t, equations)
  # Amplitude and shift
  A = 0.5
  c = 2.0

  # convenience values for trig. functions
  pi_x = pi * x[1]
  pi_y = pi * x[2]
  pi_t = pi * t

  rho = c + A * sin(pi_x) * cos(pi_y) * cos(pi_t)
  v1  = sin(pi_x) * log(x[2] + 2.0) * (1.0 - exp(-A * (x[2] - 1.0)) ) * cos(pi_t)
  v2  = v1
  p   = rho^2

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function source_terms_navier_stokes_convergence_test(u, x, t, equations)
  y = x[2]

  # TODO: parabolic
  # we currently need to hardcode these parameters until we fix the "combined equation" issue
  # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
  inv_gamma_minus_one = inv(equations.gamma - 1)
  Pr = prandtl_number()
  mu_ = mu()

  # Same settings as in `initial_condition`
  # Amplitude and shift
  A = 0.5
  c = 2.0

  # convenience values for trig. functions
  pi_x = pi * x[1]
  pi_y = pi * x[2]
  pi_t = pi * t

  # compute the manufactured solution and all necessary derivatives
  rho    =  c  + A * sin(pi_x) * cos(pi_y) * cos(pi_t)
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
  E   = p * inv_gamma_minus_one + 0.5 * rho * (v1^2 + v2^2)
  E_t = p_t * inv_gamma_minus_one + rho_t * v1^2 + 2.0 * rho * v1 * v1_t
  E_x = p_x * inv_gamma_minus_one + rho_x * v1^2 + 2.0 * rho * v1 * v1_x
  E_y = p_y * inv_gamma_minus_one + rho_y * v1^2 + 2.0 * rho * v1 * v1_y

  # Some convenience constants
  T_const = equations.gamma * inv_gamma_minus_one / Pr
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
                      - 4.0 / 3.0 * v1_xx * mu_
                      + 2.0 / 3.0 * v2_xy * mu_
                      - v1_yy             * mu_
                      - v2_xy             * mu_ )
  # y-momentum equation
  du3 = ( rho_t * v2 + rho * v2_t + p_y + rho_x * v1    * v2
                                        + rho   * v1_x  * v2
                                        + rho   * v1    * v2_x
                                        +         rho_y * v2^2
                                        + 2.0   * rho   * v2 * v2_y
    # stress tensor from y-direction
                      - v1_xy             * mu_
                      - v2_xx             * mu_
                      - 4.0 / 3.0 * v2_yy * mu_
                      + 2.0 / 3.0 * v1_xy * mu_ )
  # total energy equation
  du4 = ( E_t + v1_x * (E + p) + v1 * (E_x + p_x)
              + v2_y * (E + p) + v2 * (E_y + p_y)
    # stress tensor and temperature gradient terms from x-direction
                                - 4.0 / 3.0 * v1_xx * v1   * mu_
                                + 2.0 / 3.0 * v2_xy * v1   * mu_
                                - 4.0 / 3.0 * v1_x  * v1_x * mu_
                                + 2.0 / 3.0 * v2_y  * v1_x * mu_
                                - v1_xy     * v2           * mu_
                                - v2_xx     * v2           * mu_
                                - v1_y      * v2_x         * mu_
                                - v2_x      * v2_x         * mu_
         - T_const * inv_rho_cubed * (        p_xx * rho   * rho
                                      - 2.0 * p_x  * rho   * rho_x
                                      + 2.0 * p    * rho_x * rho_x
                                      -       p    * rho   * rho_xx ) * mu_
    # stress tensor and temperature gradient terms from y-direction
                                - v1_yy     * v1           * mu_
                                - v2_xy     * v1           * mu_
                                - v1_y      * v1_y         * mu_
                                - v2_x      * v1_y         * mu_
                                - 4.0 / 3.0 * v2_yy * v2   * mu_
                                + 2.0 / 3.0 * v1_xy * v2   * mu_
                                - 4.0 / 3.0 * v2_y  * v2_y * mu_
                                + 2.0 / 3.0 * v1_x  * v2_y * mu_
         - T_const * inv_rho_cubed * (        p_yy * rho   * rho
                                      - 2.0 * p_y  * rho   * rho_y
                                      + 2.0 * p    * rho_y * rho_y
                                      -       p    * rho   * rho_yy ) * mu_ )

  return SVector(du1, du2, du3, du4)
end

initial_condition = initial_condition_navier_stokes_convergence_test

# BC types
velocity_bc_top_bottom = NoSlip((x, t, equations) -> initial_condition_navier_stokes_convergence_test(x, t, equations)[2:3])
heat_bc_top_bottom = Adiabatic((x, t, equations) -> 0.0)
boundary_condition_top_bottom = BoundaryConditionNavierStokesWall(velocity_bc_top_bottom, heat_bc_top_bottom)

# define inviscid boundary conditions
boundary_conditions = (; :top_bottom => boundary_condition_slip_wall)

# define viscous boundary conditions
boundary_conditions_parabolic = (; :top_bottom => boundary_condition_top_bottom)

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, dg;
                                             boundary_conditions=(boundary_conditions, boundary_conditions_parabolic),
                                             source_terms=source_terms_navier_stokes_convergence_test)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(); abstol=time_int_tol, reltol=time_int_tol,
            ode_default_options()..., callback=callbacks)
summary_callback() # print the timer summary
