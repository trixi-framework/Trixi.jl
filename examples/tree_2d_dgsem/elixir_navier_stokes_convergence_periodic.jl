using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

reynolds_number() = 100
prandtl_number() = 0.72

equations = CompressibleEulerEquations2D(2.0)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, Reynolds=reynolds_number(), Prandtl=prandtl_number(),
                                                          Mach_freestream=0.5)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs,
#                volume_integral=VolumeIntegralWeakForm())

volume_flux = flux_ranocha
solver = DGSEM(polydeg=3, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 1.0,  1.0) # maximum coordinates (max(x), max(y))

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# Note: the initial condition cannot be specialized to `CompressibleNavierStokesDiffusion2D`
#       since it is called by both the parabolic solver (which passes in `CompressibleNavierStokesDiffusion2D`)
#       and by the initial condition (which passes in `CompressibleEulerEquations2D`).
# This convergence test setup was originally derived by Andrew Winters (@andrewwinters5000)
function initial_condition_navier_stokes_convergence_test_periodic(x, t, equations)
  # Amplitude and shift
  A = 0.2
  c = 2.0

  # convenience values for trig. functions
  alpha = pi * ( x[1] + x[2] - t )

  rho = c + A * sin(alpha)
  v1  = 1.0
  v2  = 1.0
  p   = rho^2

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

@inline function source_terms_navier_stokes_convergence_test_periodic(u, x, t, equations)
  # For this manufactured solution ansatz the following terms are zero
  # u_t = v_t =u_x = v_x = u_y = v_x = u_xx = u_xy = u_yy = v_xx = v_xy = v_yy = 0

  # TODO: parabolic
  # we currently need to hardcode these parameters until we fix the "combined equation" issue
  # see also https://github.com/trixi-framework/Trixi.jl/pull/1160
  inv_gamma_minus_one = inv(equations.gamma - 1)
  Pr = prandtl_number()
  Re = reynolds_number()

  # Same settings as in `initial_condition`
  # Amplitude and shift
  A = 0.2
  c = 2.0

  # convenience values for trig. functions
  alpha = pi * ( x[1] + x[2] - t )

  # compute the manufactured solution and all necessary derivatives
  rho    =   c + A * sin(alpha)
  rho_x  =  pi * A * cos(alpha)
  rho_t  = -rho_x
  rho_y  =  rho_x
  rho_xx = -pi * pi * A * sin(alpha)
  rho_yy =  rho_xx

  v1    = 1.0
  v1_t  = 0.0
  v1_x  = 0.0
  v1_y  = 0.0
  v1_xx = 0.0
  v1_xy = 0.0
  v1_yy = 0.0

  v2    = v1
  v2_t  = v1_t
  v2_x  = v1_x
  v2_y  = v1_y
  v2_xx = v1_xx
  v2_xy = v1_xy
  v2_yy = v1_yy

  p    = rho * rho
  p_x  = 2.0 * rho * rho_x
  p_y  = p_x
  p_xx = 2.0 * rho * rho_xx + 2.0 * rho_x * rho_x
  p_yy = p_xx

  # Note this simplifies slightly because the ansatz assumes that v1 = v2
  E   =  p + rho
  E_x =  p_x + rho_x
  E_y =  E_x
  E_t = -E_x

  # Some convenience constants
  T_const = equations.gamma * inv_gamma_minus_one / Pr
  inv_Re = 1.0 / Re
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

initial_condition = initial_condition_navier_stokes_convergence_test_periodic

semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic), initial_condition, solver;
                                             source_terms=source_terms_navier_stokes_convergence_test_periodic)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span `tspan`
tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-8
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol, dt = 1e-5,
            save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary

