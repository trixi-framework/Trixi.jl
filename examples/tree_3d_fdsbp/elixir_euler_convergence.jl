
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test

D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                         derivative_order=1,
                         accuracy_order=4,
                         xmin=-1.0, xmax=1.0,
                         N=16)
flux_splitting = steger_warming_splitting
solver = DG(D_upw, nothing #= mortar =#,
            SurfaceIntegralUpwind(flux_splitting),
            VolumeIntegralUpwind(flux_splitting))

coordinates_min = (0.0, 0.0, 0.0)
coordinates_max = (2.0, 2.0, 2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.1)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
