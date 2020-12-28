
using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, -0.5)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

initial_condition = initial_condition_gauss

surface_flux = flux_lax_friedrichs
solver = DGSEM(3, surface_flux)

coordinates_min = (-5, -5)
coordinates_max = ( 5,  5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 20.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_restart = SaveRestartCallback(interval=100,
                                   save_final_restart=true)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

# Optional: Instead of plotting the results, save results to file (e.g., to create GIFs or for
# running on headless servers)
function save_insitu_visualization(plot_data, variable_names;
                                   show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
                                   time=nothing, timestep=nothing)
  # Gather subplots
  plots = []
  for v in variable_names
    push!(plots, plot(plot_data[v]; plot_arguments...))
  end
  if show_mesh
    push!(plots, plot(getmesh(plot_data); plot_arguments...))
  end

  # Determine layout
  cols = ceil(Int, sqrt(length(plots)))
  rows = ceil(Int, length(plots)/cols)
  layout = (rows, cols)

  # Create plot
  plot(plots..., layout=layout)

  # Determine filename
  # We avoid the following "proper" implementation to not have to add a Printf dependency
  # filename = joinpath("out", @sprintf("solution_%06d.png", timestep))
  filler = timestep == 0 ? "000000" : "0"^(6 - (floor(Int, log10(timestep)) + 1))
  filename = joinpath("out", "solution_$(filler)$(timestep).png")

  # Save plot
  savefig(filename)
end

# Enable in-situ visualization with a new plot generated every 20 time steps
# and additional plotting options passed as keyword arguments
visualization = VisualizationCallback(interval=20, clims=(0,1))

amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first),
                                      base_level=3,
                                      med_level=4, med_threshold=0.1,
                                      max_level=5, max_threshold=0.6)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=5,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=1.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_restart, save_solution, visualization,
                        amr_callback, stepsize_callback);


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
