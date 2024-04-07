# The same setup as tree_3d_dgsem/elixir_euler_source_terms.jl
# to verify the StructuredMesh implementation against TreeMesh

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# coordinates_min = (0.0, 0.0, 0.0)
# coordinates_max = (2.0, 2.0, 2.0)
f1(s, t) = SVector(0.0, s + 1.0, t + 1.0)
f2(s, t) = SVector(2.0, s + 1.0, t + 1.0)
f3(s, t) = SVector(s + 1.0, 0.0, t + 1.0)
f4(s, t) = SVector(s + 1.0, 2.0, t + 1.0)
f5(s, t) = SVector(s + 1.0, t + 1.0, 0.0)
f6(s, t) = SVector(s + 1.0, t + 1.0, 2.0)

cells_per_dimension = (4, 4, 4)

mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4, f5, f6))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms = source_terms_convergence_test)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
