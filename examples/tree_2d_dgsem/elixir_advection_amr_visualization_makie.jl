
using OrdinaryDiffEq
using Trixi
using GLMakie

###############################################################################
# semidiscretization of the linear advection equation

advection_velocity = (0.2, -0.7)
equations = LinearScalarAdvectionEquation2D(advection_velocity)

function initial_condition_gauss_largedomain(x, t,
                                             equation::LinearScalarAdvectionEquation2D)
    # Store translated coordinate for easy use of exact solution
    domain_length = SVector(10, 10)
    x_trans = Trixi.x_trans_periodic_2d(x - equation.advection_velocity * t, domain_length)

    return SVector(exp(-(x_trans[1]^2 + x_trans[2]^2)))
end
initial_condition = initial_condition_gauss_largedomain

solver = DGSEM(polydeg = 3, surface_flux = flux_lax_friedrichs)

coordinates_min = (-5.0, -5.0)
coordinates_max = (5.0, 5.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 30_000)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

# Enable in-situ visualization with a new plot generated every 100 time steps.
# plot_creator is set to show_plot_makie in order to use the GLMakie backend.
# Additional keyword arguments, such as colorrange, are passed to the respective plotting
# command.
visualization = VisualizationCallback(interval = 100,
                                      plot_creator = Trixi.show_plot_makie,
                                      colorrange = (0.0, 1.0))

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable = first),
                                      base_level = 3,
                                      med_level = 4, med_threshold = 0.1,
                                      max_level = 5, max_threshold = 0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval = 5,
                           adapt_initial_condition = true,
                           adapt_initial_condition_only_refine = true)

stepsize_callback = StepsizeCallback(cfl = 1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, visualization,
                        amr_callback, stepsize_callback);

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep = false, callback = callbacks);
summary_callback() # print the timer summary
