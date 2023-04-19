# !!! warning "Experimental implementation (upwind SBP)"
#     This is an experimental feature and may change in future releases.

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_convergence_test

D_upw = upwind_operators(SummationByPartsOperators.Mattsson2017,
                          derivative_order=1,
                          accuracy_order=4,
                          xmin=-1.0, xmax=1.0,
                          N=32)
flux_splitting = splitting_steger_warming
solver = FDSBP(D_upw,
               surface_integral=SurfaceIntegralUpwind(flux_splitting),
               volume_integral=VolumeIntegralUpwind(flux_splitting))

coordinates_min = 0.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=1,
                n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution)


###############################################################################
# run the simulation

sol = solve(ode, SSPRK43(); abstol=1.0e-6, reltol=1.0e-6,
            ode_default_options()..., callback=callbacks)
summary_callback() # print the timer summary
