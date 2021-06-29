
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 2.0
equations_euler = CompressibleEulerEquations3D(gamma)

initial_condition = initial_condition_eoc_test_coupled_euler_gravity

polydeg = 3
solver_euler = DGSEM(polydeg, flux_hll)

coordinates_min = (0, 0, 0)
coordinates_max = (2, 2, 2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=10_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver_euler,
                                          source_terms=source_terms_eoc_test_coupled_euler_gravity)


###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations3D()

solver_gravity = DGSEM(polydeg, flux_lax_friedrichs)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition, solver_gravity,
                                            source_terms=source_terms_harmonic)


###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density=2.0, # aka rho0
                                    gravitational_constant=1.0, # aka G
                                    cfl=1.5,
                                    resid_tol=1.0e-10,
                                    n_iterations_max=1000,
                                    timestep_gravity=timestep_gravity_erk52_3Sstar!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)


###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                     save_analysis=true)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=10,
                                   save_initial_solution=true,
                                   save_final_solution=true,
                                   solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.1)

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
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
