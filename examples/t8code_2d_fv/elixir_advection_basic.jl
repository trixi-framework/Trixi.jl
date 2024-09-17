using OrdinaryDiffEq
using Trixi

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

initial_condition = initial_condition_convergence_test

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

# Note:
# For now, it is completely irrelevant that coordinates_max/min are.
# The used t8code routine creates the mesh on [0, nx] x [0, ny], where (nx, ny) = trees_per_dimension.
# Afterwards and only inside Trixi, `tree_node_coordinates` are mapped back to [-1, 1]^2.
# But, this variable is not used for the FV method.
# That's why I use the cmesh interface in all other elixirs.
coordinates_min = (0.0, 0.0) # minimum coordinates (min(x), min(y))
coordinates_max = (8.0, 8.0) # maximum coordinates (max(x), max(y))

trees_per_dimension = (8, 8)

mesh = T8codeMesh(trees_per_dimension,
                  coordinates_min = coordinates_min, coordinates_max = coordinates_max,
                  initial_refinement_level = 4)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan);

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.9)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback, save_solution)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback()
