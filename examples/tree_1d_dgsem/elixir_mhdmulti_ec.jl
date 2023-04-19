
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the ideal MHD equations
equations = IdealGlmMhdMulticomponentEquations1D(gammas = (2.0, 2.0, 2.0),
                                                 gas_constants = (2.0, 2.0, 2.0))

initial_condition = initial_condition_weak_blast_wave

volume_flux = flux_hindenlang_gassner
solver = DGSEM(polydeg = 3, surface_flux = flux_hindenlang_gassner,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

coordinates_min = 0.0
coordinates_max = 1.0
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 4,
                n_cells_max = 10_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
