using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleMoistEulerEquations2D() 

initial_condition = initial_condition_convergence_test_moist

polydeg = 3
basis = LobattoLegendreBasis(polydeg)

surface_flux = flux_lax_friedrichs

solver = DGSEM(basis, surface_flux)

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)

mesh = TreeMesh(coordinates_min, coordinates_max,
initial_refinement_level=2,
n_cells_max=10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    source_terms=source_terms_convergence_test_moist)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(energy_kinetic, energy_internal,
                                                               density, density_dry,
                                                               density_vapor, density_liquid,
                                                               ratio_vapor, ratio_liquid,
                                                               temperature, pressure,
                                                               moist_pottemp_thermodynamic,
                                                               aequivalent_pottemp_thermodynamic,
                                                               density_pressure))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2moistpot)

stepsize_callback = StepsizeCallback(cfl=1.0)

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
