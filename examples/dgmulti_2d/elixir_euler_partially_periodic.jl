using OrdinaryDiffEqLowStorageRK
using Trixi

dg = DGMulti(polydeg = 3, element_type = Tri(), approximation_type = Polynomial(),
             surface_integral = SurfaceIntegralWeakForm(flux_hll),
             volume_integral = VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)
# The manufactured solution is periodic in x on the domain [-1, 1]^2
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

# Tag all four sides of the domain. Since the mesh is periodic in x, the faces at the
# left and right sides are connected periodically and are not boundary faces.
left(x, tol = 50 * eps()) = abs(x[1] + 1) < tol
right(x, tol = 50 * eps()) = abs(x[1] - 1) < tol
bottom(x, tol = 50 * eps()) = abs(x[2] + 1) < tol
top(x, tol = 50 * eps()) = abs(x[2] - 1) < tol
is_on_boundary = (; left, right, bottom, top)

cells_per_dimension = (8, 8)
mesh = DGMultiMesh(dg, cells_per_dimension; is_on_boundary,
                   periodicity = (true, false))

# `boundary_condition_periodic` can be used in the `NamedTuple` of boundary conditions
# for the boundaries that are connected periodically by the mesh.
boundary_condition_convergence_test = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = (; left = boundary_condition_periodic,
                       right = boundary_condition_periodic,
                       bottom = boundary_condition_convergence_test,
                       top = boundary_condition_convergence_test)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_conditions)

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl = 1.5)
save_solution = SaveSolutionCallback(interval = analysis_interval,
                                     solution_variables = cons2prim)
callbacks = CallbackSet(summary_callback, alive_callback, stepsize_callback,
                        analysis_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode,
            CarpenterKennedy2N54(williamson_condition = false, thread = Trixi.Threaded());
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()...,
            callback = callbacks);
