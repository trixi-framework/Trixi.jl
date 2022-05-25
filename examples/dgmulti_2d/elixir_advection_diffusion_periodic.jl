
using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(1.0, 1.0)
parabolic_equations = LaplaceDiffusion2D(1e-2)

function initial_condition_sharp_gaussian(x, t, equations::LinearScalarAdvectionEquation2D)
  return SVector(exp(-100 * (x[1]^2 + x[2]^2)))
end
initial_condition = initial_condition_sharp_gaussian

mesh = DGMultiMesh(dg, cells_per_dimension = (16, 16), periodicity=true)
semi = SemidiscretizationHyperbolicParabolic(mesh, equations, parabolic_equations, initial_condition, dg)

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval=10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval, uEltype=real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(), abstol=tol, reltol=tol, save_everystep=false, callback=callbacks)
summary_callback() # print the timer summary
