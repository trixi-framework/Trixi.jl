
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_density_wave

surface_flux = flux_lax_friedrichs
solver = DGSEM(5, surface_flux)

coordinates_min = -1
coordinates_max =  1
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=0.8)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=:primitive)
# TODO: Taal, restart
# restart_interval = 10

analysis_interval = 2000
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

callbacks = CallbackSet(summary_callback, stepsize_callback, save_solution, analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false), dt=stepsize_callback(ode),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
