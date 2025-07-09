using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

# Volume flux stabilizes the simulation - in contrast to standard DGSEM with 
# `surface_flux = flux_ranocha` only which crashes.
solver = DGSEM(polydeg = 3, surface_flux = flux_ranocha,
               volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

coordinates_min = -2.0
coordinates_max = 2.0
cells_per_dimension = 32
mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)

initial_condition = initial_condition_weak_blast_wave
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_callback = AnalysisCallback(semi, interval = 1,
                                     analysis_errors = Symbol[], # Switch off error computation
                                     # Note: `entropy` defaults to mathematical entropy
                                     analysis_integrals = (entropy,),
                                     analysis_filename = "entropy_ER.dat",
                                     save_analysis = true)

stepsize_callback = StepsizeCallback(cfl = 0.25)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 5,
                                                 root_tol = eps(Float64),
                                                 gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationRK44(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0, save_everystep = false, callback = callbacks);
