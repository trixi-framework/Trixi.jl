
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (0.2, -0.7, 0.3)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

initial_condition = initial_condition_convergence_test
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1, -1, -1)
coordinates_max = ( 1,  1,  1)
refinement_patches = (
  (type="box", coordinates_min=(0.0, -1.0, -1.0), coordinates_max=(1.0, 1.0, 1.0)),
  (type="box", coordinates_min=(0.0, -0.5, -0.5), coordinates_max=(0.5, 0.5, 0.5)),
)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                refinement_patches=refinement_patches,
                n_cells_max=10_000,)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                   save_initial_solution=true,
                                   save_final_solution=true,
                                   solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_restart,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
