using Trixi

###############################################################################
# semidiscretization of the compressible ideal GLM-MHD equations

gamma = 5 / 3
equations = IdealGlmMhdEquations2D(gamma)

volume_flux = (flux_central, flux_nonconservative_powell)
solver = DGSEM(polydeg = 2,
               surface_flux = (flux_lax_friedrichs, flux_nonconservative_powell),
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

function mapping(xi_, eta_)
    # Transform input variables between -1 and 1 onto [0, sqrt(2)]
    # Note, we use the domain [0, sqrt(2)]^2 for the Alfvén wave convergence test case
    x = 0.5 * sqrt(2) * (xi_ + 1)
    y = 0.5 * sqrt(2) * (eta_ + 1)

    return SVector(x, y)
end

N = 4 # Can use this for convergence check, i.e., crank up to 8, 16, 32, 64, ...
cells_per_dimension = (N, N)
mesh = StructuredMesh(cells_per_dimension, mapping)

initial_condition = initial_condition_convergence_test
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100_000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false,
                                     analysis_errors = [:l2_error, :l1_error, :linf_error],
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

cfl = 1.0
stepsize_callback = StepsizeCallback(cfl = cfl)
glm_speed_callback = GlmSpeedCallback(glm_scale = 0.5, cfl = cfl)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        stepsize_callback,
                        glm_speed_callback)

###############################################################################
# run the simulation

relaxation_solver = Trixi.RelaxationSolverNewton(max_iterations = 10, root_tol = 1e-15,
                                                 gamma_tol = eps(Float64))
ode_alg = Trixi.RelaxationRK33(relaxation_solver = relaxation_solver)

sol = Trixi.solve(ode, ode_alg,
                  dt = 42.0, save_everystep = false, callback = callbacks);
