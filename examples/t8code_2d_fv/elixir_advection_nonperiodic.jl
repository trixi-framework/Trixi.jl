using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the linear advection equation.

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

boundary_condition = BoundaryConditionDirichlet(initial_condition)
boundary_conditions = Dict(:all => boundary_condition)
# Problem: T8codeMesh interface with parameter cmesh cannot distinguish between boundaries
# boundary_conditions = Dict(:x_neg => boundary_condition,
#                            :x_pos => boundary_condition,
#                            :y_neg => boundary_condition_periodic,
#                            :y_pos => boundary_condition_periodic)

solver = FV(surface_flux = flux_lax_friedrichs)

initial_refinement_level = 5
cmesh = Trixi.cmesh_new_periodic_quad_nonperiodic(Trixi.mpi_comm())
mesh = T8codeMesh(cmesh, solver, initial_refinement_level = initial_refinement_level)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
alive_callback = AliveCallback(analysis_interval = analysis_interval)

stepsize_callback = StepsizeCallback(cfl = 0.8)


callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, stepsize_callback)

###############################################################################
# Run the simulation.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # Solve needs some value here but it will be overwritten by the stepsize_callback.
            save_everystep = false, callback = callbacks);
summary_callback()
