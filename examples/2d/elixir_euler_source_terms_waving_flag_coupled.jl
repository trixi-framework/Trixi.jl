
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg=3, surface_flux=FluxRotated(flux_lax_friedrichs))

boundary_1_top    = Trixi.BoundaryConditionCoupled(2, 2, (:i, 1),    2, Float64)
boundary_1_bottom = Trixi.BoundaryConditionCoupled(2, 2, (:i, :end), 2, Float64)
boundary_2_top    = Trixi.BoundaryConditionCoupled(1, 2, (:i, 1), 2, Float64)
boundary_2_bottom = Trixi.BoundaryConditionCoupled(1, 2, (:i, :end),    2, Float64)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, 0.5 * s - 1.5)
f2(s) = SVector( 1.0, 0.5 * s + 0.5)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, sin(0.5 * pi * s))

mesh = CurvedMesh((16, 8), (f1, f2, f3, f4))

semi1 = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic,
                                                          boundary_1_bottom, boundary_1_top))

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, 0.5 * s + 1.5)
f2(s) = SVector( 1.0, 0.5 * s + 3.5)
f3(s) = SVector(s, 2.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 3.0 + sin(0.5 * pi * s))

mesh = CurvedMesh((16, 8), (f1, f2, f3, f4))


semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                     source_terms=source_terms_convergence_test,
                                     boundary_conditions=(boundary_condition_periodic, boundary_condition_periodic,
                                                          boundary_2_bottom, boundary_2_top))


semi = SemidiscretizationHyperbolicCoupled((semi1, semi2))

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

# save_solution = SaveSolutionCallback(interval=100,
#                                      save_initial_solution=true,
#                                      save_final_solution=true,
#                                      solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
