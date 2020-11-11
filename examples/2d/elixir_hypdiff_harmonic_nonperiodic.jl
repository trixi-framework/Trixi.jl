
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the hyperbolic diffusion equations

resid_tol = 5.0e-12 # TODO: Taal, move this parameter to the callback
equations = HyperbolicDiffusionEquations2D(resid_tol)

initial_condition = initial_condition_harmonic_nonperiodic
boundary_conditions = boundary_condition_harmonic_nonperiodic

surface_flux = flux_upwind
solver = DGSEM(4, surface_flux)

coordinates_min = (0, 0)
coordinates_max = (1, 1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000,
                periodicity=false)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions,
                                    source_terms=source_terms_harmonic)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 5.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

steady_state_callback = SteadyStateCallback(abstol=resid_tol, reltol=0.0)

stepsize_callback = StepsizeCallback(cfl=1.0)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

callbacks = CallbackSet(summary_callback, steady_state_callback, stepsize_callback,
                        save_solution,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
