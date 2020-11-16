
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

# Create DG solver with polynomial degree 3 and the Lax-Friedrichs surface flux
solver = DGSEM(3, flux_lax_friedrichs)

coordinates_min = (-1, -1)
coordinates_max = ( 1,  1)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

summary_callback = SummaryCallback()

stepsize_callback = StepsizeCallback(cfl=1.6)

save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=:primitive)

analysis_callback = AnalysisCallback(semi, interval=100)

callbacks = CallbackSet(summary_callback, stepsize_callback, save_solution, analysis_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
