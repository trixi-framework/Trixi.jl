
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressiblePotTempEulerEquations2D()

initial_condition = initial_condition_gaussian_bubble

boundary_conditions = (x_neg=boundary_condition_periodic,
                       x_pos=boundary_condition_periodic,
                       y_neg=boundary_condition_reflection,
                       y_pos=boundary_condition_reflection)

###############################################################################
# Get the DG approximation space


solver = DGSEM(polydeg=3, surface_flux=flux_LMARS)

###############################################################################
# Get the curved quad mesh from a mapping function

cells_per_dimension = (16, 16)

# Create curved mesh with 16 x 16 elements
mesh = StructuredMesh(cells_per_dimension, (0.0, 0.0), (1000.0, 1500.0))

###############################################################################
# create the semi discretization object

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 18.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true)

stepsize_callback = StepsizeCallback(cfl=1.0)

callbacks = CallbackSet(summary_callback,
                        analysis_callback,
                        alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
