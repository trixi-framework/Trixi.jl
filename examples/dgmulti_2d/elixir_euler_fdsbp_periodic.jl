
using Trixi, OrdinaryDiffEq

dg = DGMulti(element_type = Quad(),
             approximation_type = periodic_derivative_operator(derivative_order = 1,
                                                               accuracy_order = 4,
                                                               xmin = 0.0, xmax = 1.0,
                                                               N = 50),
             surface_flux = flux_hll,
             volume_integral = VolumeIntegralWeakForm())

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

mesh = DGMultiMesh(dg, coordinates_min = (-1.0, -1.0),
                   coordinates_max = (1.0, 1.0))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg,
                                    source_terms = source_terms)

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 10)
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval, uEltype = real(dg))
stepsize_callback = StepsizeCallback(cfl = 1.0)
callbacks = CallbackSet(summary_callback, alive_callback, stepsize_callback,
                        analysis_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 0.5 * estimate_dt(mesh, dg), save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
