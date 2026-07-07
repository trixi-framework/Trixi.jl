using OrdinaryDiffEqLowStorageRK
using Trixi

# Continuous Galerkin SBP operator
D = couple_continuously(legendre_derivative_operator(xmin = -1.0, xmax = 1.0, N = 4),
                        SummationByPartsOperators.UniformMesh1D(xmin = -1.0, xmax = 1.0,
                                                                Nx = 8))

surface_flux = flux_lax_friedrichs
dg = DGMulti(element_type = Line(),
             approximation_type = D,
             surface_integral = SurfaceIntegralWeakForm(surface_flux),
             volume_integral = VolumeIntegralWeakForm())

advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

left(x, tol = 50 * eps()) = abs(x[1] + 1) < tol
right(x, tol = 50 * eps()) = abs(x[1] - 1) < tol
is_on_boundary = (; left = left, right = right)

mesh = DGMultiMesh(dg, (1,),
                   coordinates_min = (-1.0,), coordinates_max = (1.0,),
                   is_on_boundary = is_on_boundary,
                   periodicity = false)

# nonperiodic BCs
initial_condition = initial_condition_convergence_test
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    boundary_conditions = (;
                                                           left = BoundaryConditionDirichlet(initial_condition_convergence_test),
                                                           right = boundary_condition_do_nothing))

tspan = (0.0, 1.7)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_callback = AnalysisCallback(semi, interval = 100, uEltype = real(dg))
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); ode_default_options()..., callback = callbacks);
