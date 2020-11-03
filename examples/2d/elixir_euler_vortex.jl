
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_isentropic_vortex

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-10, -10)
coordinates_max = ( 10,  10)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

# FIXME Taal restore after Taam sync
# stepsize_callback = StepsizeCallback(cfl=1.4)
stepsize_callback = StepsizeCallback(cfl=0.6)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)

analysis_callback = AnalysisCallback(semi, interval=analysis_interval, save_analysis=true,
                                     extra_analysis_errors=(:conservation_error,),
                                     extra_analysis_integrals=(entropy, energy_total,
                                                               energy_kinetic, energy_internal))

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        save_restart, save_solution,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
