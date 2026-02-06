using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

equations = IdealGlmMhdEquations3D(5 / 3)

# Volume flux stabilizes the simulation - in contrast to standard DGSEM with 
# `surface_flux = (flux_hindenlang_gassner, flux_nonconservative_powell)` only which crashes.
# To turn this into a convergence test, use a flux with some dissipation, e.g.
flux = (flux_hindenlang_gassner, flux_nonconservative_powell)
solver = DGSEM(polydeg = 3, surface_flux = flux,
               volume_integral = VolumeIntegralFluxDifferencing(flux))

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

trees_per_dimension = (2, 2, 2)
mesh = P4estMesh(trees_per_dimension,
                 polydeg = 1, initial_refinement_level = 1,
                 coordinates_min = coordinates_min, coordinates_max = coordinates_max)

initial_condition = initial_condition_convergence_test
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 10,
                                     analysis_errors = Symbol[], # Switch off error computation
                                     # Note: `entropy` defaults to mathematical entropy
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "entropy_ER.dat",
                                     save_analysis = true)

cfl = 1.0
stepsize_callback = StepsizeCallback(cfl = cfl)

glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

# Ensure exact entropy conservation by employing a relaxation Runge-Kutta method
relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5,
                                                 root_tol = eps(Float64),
                                                 gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationCKL54(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg;
                  dt = 42.0, save_everystep = false, callback = callbacks);
