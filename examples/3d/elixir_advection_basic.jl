
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0, 1.0)
# advectionvelocity = (0.2, -0.3, 0.7)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

initial_condition = initial_condition_convergence_test

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-1, -1, -1)
coordinates_max = ( 1,  1,  1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=1.2)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:conservative)
# TODO: Taal, IO
# restart_interval = 10

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy, energy_total))

callbacks = CallbackSet(summary_callback, stepsize_callback, save_solution, analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
