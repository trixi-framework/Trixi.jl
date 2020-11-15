
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

solver = DGSEM(3,                   # polynomial degree
               flux_lax_friedrichs) # surface flux function

coordinates_min = 
coordinates_max = 
mesh = TreeMesh((-1, -1), # minimum coordinates (min(x), min(y))
                ( 1,  1), # maximum coordinates (max(x), max(y))
                initial_refinement_level=4,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh,
                                    equations,
                                    initial_condition_convergence_test, # initial condition function
                                    solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

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
