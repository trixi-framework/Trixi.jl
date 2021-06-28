
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

initial_condition = initial_condition_gauss
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-5, -5, -5)
coordinates_max = ( 5,  5,  5)
trees_per_dimension = (1, 1, 1)
mesh = P4estMesh(trees_per_dimension, polydeg=1,
                 coordinates_min=coordinates_min, coordinates_max=coordinates_max,
                 initial_refinement_level=4)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.3)
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

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=4,
                                      med_level=5, med_threshold=0.1,
                                      max_level=6, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.2)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_restart,
                        save_solution,
                        amr_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
