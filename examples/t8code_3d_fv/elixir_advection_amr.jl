using OrdinaryDiffEq
using Trixi

###############################################################################
# Semidiscretization of the linear advection equation.

advection_velocity = (0.2, -0.7, 0.5)
equations = LinearScalarAdvectionEquation3D(advection_velocity)

initial_condition = initial_condition_gauss

solver = FV(order = 2, extended_reconstruction_stencil = false,
            surface_flux = flux_lax_friedrichs)

coordinates_min = (-2.0, -2.0, -2.0) # minimum coordinates (min(x), min(y), min(z))
coordinates_max = (2.0, 2.0, 2.0) # maximum coordinates (max(x), max(y), max(z))

trees_per_dimension = (2, 2, 2)

# Disabling the gc for almost the entire elixir seems to work in order to fix the SegFault errors with trixi_t8_mapping_c and mapping_coordinates
# It's also possible to move this to the constructor.
GC.enable(false)
element_class = :hex
mesh = T8codeMesh(trees_per_dimension, element_class,
                  coordinates_max = coordinates_max, coordinates_min = coordinates_min,
                  initial_refinement_level = 3)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 10,
                                     solution_variables = cons2prim)

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 1,
                                      med_level = 3, med_threshold = 0.1,
                                      max_level = 4, max_threshold = 0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true,
                           dynamic_load_balancing = true)

stepsize_callback = StepsizeCallback(cfl = 0.7)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        amr_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# Run the simulation.

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);

summary_callback() # print the timer summary

GC.enable(true)
