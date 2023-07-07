
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the polytropic Euler equations

gamma = 1.001   # Quasi-isotropic simulation. gamma = 1.0 would lead to a singularity.
# gamma = 2.0   # Adiabatic monatomic gas in 2d.
kappa = 1.0     # Scaling factor for the pressure.
equations = PolytropicEulerEquations2D(gamma, kappa)

initial_condition = initial_condition_convergence_test

volume_flux = flux_winters_etal
solver = DGSEM(polydeg=2, surface_flux=flux_hll,
               volume_integral=VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)

cells_per_dimension = (64, 64)

boundary_conditions = (
                        x_neg=boundary_condition_periodic,
                        x_pos=boundary_condition_periodic,
                        y_neg=boundary_condition_periodic,
                        y_pos=boundary_condition_periodic,
                        )

mesh = StructuredMesh(cells_per_dimension,
coordinates_min,
coordinates_max)
  
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_errors=(:l2_error_primitive,
                                                            :linf_error_primitive))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=1.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
