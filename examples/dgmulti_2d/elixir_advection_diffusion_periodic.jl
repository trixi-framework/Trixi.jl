using Trixi, OrdinaryDiffEq

dg = DGMulti(polydeg = 1, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(0.0, 0.0)
equations_parabolic = LaplaceDiffusion2D(5.0e-1, equations)

function initial_condition_sharp_gaussian(x, t, equations::LinearScalarAdvectionEquation2D)
    return SVector(exp(-100 * (x[1]^2 + x[2]^2)))
end
initial_condition = initial_condition_sharp_gaussian

cells_per_dimension = (16, 16)
mesh = DGMultiMesh(dg, cells_per_dimension, periodicity = true)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg)

tspan = (0.0, 0.1)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-6
sol = solve(ode, RDPK3SpFSAL49(); abstol = time_int_tol, reltol = time_int_tol,
            dt = time_int_tol, ode_default_options()..., callback = callbacks)

summary_callback() # print the timer summary
