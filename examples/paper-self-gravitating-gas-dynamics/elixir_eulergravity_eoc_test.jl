
using OrdinaryDiffEq
using Trixi


initial_condition = initial_condition_eoc_test_coupled_euler_gravity


###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 2.0
equations_euler = CompressibleEulerEquations2D(gamma)

polydeg = 3
solver_euler = DGSEM(polydeg, flux_hll)

coordinates_min = (0, 0)
coordinates_max = (2, 2)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=10_000)

semi_euler = SemidiscretizationHyperbolic(mesh, equations_euler, initial_condition, solver_euler,
                                         source_terms=source_terms_eoc_test_coupled_euler_gravity)


###############################################################################
# semidiscretization of the hyperbolic diffusion equations
equations_gravity = HyperbolicDiffusionEquations2D()

solver_gravity = DGSEM(polydeg, flux_godunov)

semi_gravity = SemidiscretizationHyperbolic(mesh, equations_gravity, initial_condition, solver_gravity,
                                            source_terms=source_terms_harmonic)


###############################################################################
# combining both semidiscretizations for Euler + self-gravity
parameters = ParametersEulerGravity(background_density=2.0, # aka rho0
                                    # rho0 is (ab)used to add a "+8π" term to the source terms
                                    # for the manufactured solution
                                    gravitational_constant=1.0, # aka G
                                    resid_tol=1.0e-11,       # TODO: Clean-up, 1.0e-10,   1.0e-11; should become abstol/reltol
                                    resid_tol_type=:l2_full, # TODO: Clean-up, :linf_phi, :l2_full
                                    cfl=2.4,                 # 2.4; 1.1 is safe for polydeg=4
                                    maxiters=1000,
                                    gravity_solver=timestep_gravity_erk52_3Sstar!,
                                    initial_gravity_solver=Trixi.bicgstabl!)

semi = SemidiscretizationEulerGravity(semi_euler, semi_gravity, parameters)


###############################################################################
# ODE solvers, callbacks etc.
tspan = (0.0, 0.5)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=0.8)

save_solution = SaveSolutionCallback(interval=10,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

analysis_callback = AnalysisCallback(semi_euler, interval=analysis_interval,
                                     save_analysis=true)

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        save_restart, save_solution,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
println("Number of gravity subcycles: ", semi.gravity_counter.ncalls_since_readout)
