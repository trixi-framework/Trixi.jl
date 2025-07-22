using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (1.0, 1.0, 1.0)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

solver = DGSEM(polydeg = 2, surface_flux = flux_central) # Entropy-conservative setup

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test,
                                    solver)

###############################################################################
# ODE solvers, callbacks etc.

ode = semidiscretize(semi, (0.0, 1.0))

summary_callback = SummaryCallback()

analysis_interval = 1
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_errors = Symbol[], # Switch off error computation
                                     # Note: `entropy` defaults to mathematical entropy
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "entropy_ER.dat",
                                     save_analysis = true)

stepsize_callback = StepsizeCallback(cfl = 1.0)

callbacks = CallbackSet(summary_callback, analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 3,
                                                 root_tol = eps(Float64),
                                                 gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationRK33(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0, save_everystep = false, callback = callbacks);
