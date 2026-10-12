using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# semidiscretization of the linear diffusion (heat) equation

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_lax_friedrichs),
             volume_integral = VolumeIntegralWeakForm())

equations = LinearScalarAdvectionEquation2D(0.0, 0.0)
diffusivity() = 0.1
equations_parabolic = LaplaceDiffusion2D(diffusivity(), equations)

function initial_condition_diffusive_convergence_test(x, t,
                                                      equation::LinearScalarAdvectionEquation2D)
    nu = diffusivity()
    scalar = exp(-2 * nu * pi^2 * t) * sinpi(x[1]) * sinpi(x[2])
    return SVector(scalar)
end
initial_condition = initial_condition_diffusive_convergence_test

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension, periodicity = true)
semi = SemidiscretizationHyperbolicParabolic(mesh, (equations, equations_parabolic),
                                             initial_condition, dg;
                                             solver_parabolic = ParabolicFormulationLocalDG(),
                                             boundary_conditions = (boundary_condition_periodic,
                                                                    boundary_condition_periodic))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback)

###############################################################################
# run the simulation

time_int_tol = 1e-11
sol = solve(ode, RDPK3SpFSAL49(thread = Trixi.Threaded());
            abstol = time_int_tol, reltol = time_int_tol,
            ode_default_options()..., callback = callbacks)
