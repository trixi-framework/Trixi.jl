using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations

equations = IdealGlmMhdEquations1D(2)

# Volume flux adds some (minimal) disspation, thus stabilizing the simulation - 
# in contrast to standard DGSEM with `flux = flux_hindenlang_gassner` only
flux = flux_hindenlang_gassner
solver = DGSEM(polydeg = 3, surface_flux = flux,
               volume_integral = VolumeIntegralFluxDifferencing(flux))

coordinates_min = -2.0
coordinates_max = 2.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000)

initial_condition = initial_condition_weak_blast_wave
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1,
                                     analysis_errors = Symbol[], # Switch off error computation
                                     # Note: entropy defaults to mathematical entropy
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "entropy_ER.dat",
                                     save_analysis = true)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5,
                                                 root_tol = eps(Float64),
                                                 gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationCKL54(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0, save_everystep = false, callback = callbacks);
