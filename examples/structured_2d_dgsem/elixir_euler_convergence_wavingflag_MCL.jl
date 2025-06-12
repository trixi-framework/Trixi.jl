using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_density_wave_highdensity

surface_flux = flux_lax_friedrichs
volume_flux = flux_ranocha
polydeg = 3
basis = LobattoLegendreBasis(polydeg)
limiter_mcl = SubcellLimiterMCL(equations, basis;
                                density_limiter = false,
                                density_coefficient_for_all = false,
                                sequential_limiter = false,
                                conservative_limiter = false,
                                positivity_limiter_density = true,
                                positivity_limiter_pressure = true,
                                positivity_limiter_pressure_exact = true,
                                Plotting = true)

volume_integral = VolumeIntegralSubcellLimiting(limiter_mcl;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

# Deformed rectangle that looks like a waving flag,
# lower and upper faces are sinus curves, left and right are vertical lines.
f1(s) = SVector(-1.0, s - 1.0)
f2(s) = SVector(1.0, s + 1.0)
f3(s) = SVector(s, -1.0 + sin(0.5 * pi * s))
f4(s) = SVector(s, 1.0 + sin(0.5 * pi * s))

cells_per_dimension = (4, 4)

mesh = StructuredMesh(cells_per_dimension, (f1, f2, f3, f4))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100000,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim,
                                     extra_node_variables = (:limiting_coefficient_rho,
                                                             :limiting_coefficient_rho_v1,
                                                             :limiting_coefficient_rho_v2,
                                                             :limiting_coefficient_rho_e,
                                                             :limiting_coefficient_pressure,
                                                             :limiting_coefficient_entropy))

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        stepsize_callback,
                        save_solution)
###############################################################################
# run the simulation

stage_callbacks = (BoundsCheckCallback(save_errors = false),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()..., callback = callbacks);
