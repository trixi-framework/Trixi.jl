using OrdinaryDiffEq
using Trixi

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

solver = FV(surface_flux = flux_lax_friedrichs)

mesh_function = Trixi.cmesh_new_periodic_hybrid
# mesh_function = Trixi.cmesh_new_periodic_quad
# mesh_function = Trixi.cmesh_new_periodic_tri
cmesh = mesh_function(Trixi.mpi_comm())
mesh = T8codeMesh(cmesh, solver,
                  initial_refinement_level = 3)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

ode = semidiscretize(semi, (0.0, 1.0));

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 5.0e-2, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback()
