using OrdinaryDiffEqLowStorageRK
using Trixi

D = couple_continuously(legendre_derivative_operator(xmin = 0.0, xmax = 1.0, N = 3),
                        UniformPeriodicMesh1D(xmin = -1.0, xmax = 1.0, Nx = 32))

dg = DGMulti(element_type = Line(),
             approximation_type = D,
             surface_flux = flux_hll,
             volume_integral = VolumeIntegralFluxDifferencing(flux_ranocha))

equations = CompressibleEulerEquations1D(1.4)
initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

mesh = DGMultiMesh(dg, coordinates_min = (-1.0,),
                   coordinates_max = (1.0,))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, dg;
                                    source_terms = source_terms,
                                    boundary_conditions = boundary_condition_periodic)

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

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 0.5 * estimate_dt(mesh, dg),
            ode_default_options()...,
            callback = callbacks);
