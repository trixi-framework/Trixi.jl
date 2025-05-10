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

solver = FV(order = 2, surface_flux = flux_lax_friedrichs)

# TODO: When using mesh construction as in elixir_advection_basic.jl boundary Symbol :all is not defined
initial_refinement_level = 5
cmesh = Trixi.cmesh_new_quad(periodicity = (false, false))

# **Note**: A non-periodic run with the tri mesh is unstable.
# - This only happens with **2nd order**
# - When increasing refinement_level to 6 and lower CFL to 0.4, it seems like the simulation is stable again (except of some smaller noises at the corners)
# - With a lower resolution (5) I cannot get the simulation stable. Even with a cfl of 0.01 etc.
# -> That can't be expected.
# -> Maybe, the reconstruction is just not fitted for this example/mesh/resolution?!
# cmesh = Trixi.cmesh_new_tri(periodicity = (false, false))

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

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.8)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution, stepsize_callback)

###############################################################################
# Run the simulation.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # Solve needs some value here but it will be overwritten by the stepsize_callback.
            save_everystep = false, callback = callbacks);
summary_callback()
