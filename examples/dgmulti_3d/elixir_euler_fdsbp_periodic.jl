
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations3D(1.4)

initial_condition = initial_condition_convergence_test
source_terms = source_terms_convergence_test

volume_flux = flux_ranocha
solver = DGMulti(element_type = Hex(),
                 approximation_type = periodic_derivative_operator(derivative_order = 1,
                                                                   accuracy_order = 4,
                                                                   xmin = 0.0, xmax = 1.0,
                                                                   N = 20),
                 surface_flux = flux_lax_friedrichs,
                 volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = DGMultiMesh(solver, coordinates_min = (-1.0, -1.0, -1.0),
                   coordinates_max = (1.0, 1.0, 1.0))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                    source_terms = source_terms)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     uEltype = real(solver))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback)

###############################################################################
# run the simulation

sol = solve(ode, RDPK3SpFSAL49(); abstol = 1.0e-7, reltol = 1.0e-7,
            ode_default_options()..., callback = callbacks)
summary_callback() # print the timer summary
