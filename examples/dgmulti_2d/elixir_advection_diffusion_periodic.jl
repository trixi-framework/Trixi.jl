using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(0.0, 0.0)
equations_parabolic = LaplaceDiffusion2D(5.0e-2, equations)

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

mesh = DGMultiMesh(dg, cells_per_dimension = (16, 16), periodicity=true)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg)

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(), abstol=time_int_tol, reltol=time_int_tol,
            dt = time_int_tol, save_everystep=false, callback=callbacks)

summary_callback() # print the timer summary