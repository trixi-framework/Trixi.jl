
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0, 1.0)
# advectionvelocity = (0.2, -0.3, 0.7)
equations = LinearScalarAdvectionEquation3D(advectionvelocity)

initial_condition = initial_condition_convergence_test

# you can either use a single function to impose the BCs weakly in all
# 1*ndims == 2 directions or you can pass a tuple containing BCs for
# each direction
# Note: "boundary_condition_periodic" indicates that it is a periodic boundary and can be omitted on
#       fully periodic domains. Here, however, it is included to allow easy override during testing
boundary_conditions = boundary_condition_periodic

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-1, -1, -1)
coordinates_max = ( 1,  1,  1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000,
                periodicity=true)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)


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

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

analysis_interval = 100
alive_callback = AliveCallback(analysis_interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy, energy_total))

callbacks = CallbackSet(summary_callback, stepsize_callback,
                        save_restart, save_solution,
                        analysis_callback, alive_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);
summary_callback() # print the timer summary
