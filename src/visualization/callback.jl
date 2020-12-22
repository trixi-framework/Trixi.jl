
mutable struct VisualizationCallback{SolutionVariables, VariableNames, PlotDataCreator}
  interval::Int
  plot_arguments::Dict{Symbol,Any}
  solution_variables::SolutionVariables
  variable_names::VariableNames
  plot_data_creator::PlotDataCreator
  show_mesh::Bool
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:VisualizationCallback}
  visualization_callback = cb.affect!
  @unpack interval, plot_arguments, solution_variables, variable_names, plot_data_creator, show_mesh = visualization_callback
  print(io, "VisualizationCallback(",
            "interval=", interval, ", ",
            "plot_arguments=", plot_arguments, ", ",
            "solution_variables=", solution_variables, ", ",
            "variable_names=", variable_names, ", ",
            "plot_data_creator=", plot_data_creator, ", ",
            "show_mesh=", show_mesh, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:VisualizationCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    visualization_callback = cb.affect!

    setup = [
             "interval" => visualization_callback.interval,
             "plot arguments" => visualization_callback.plot_arguments,
             "solution variables" => visualization_callback.solution_variables,
             "variable names" => visualization_callback.variable_names,
             "plot data creator" => visualization_callback.plot_data_creator,
             "show mesh" => visualization_callback.show_mesh,
            ]
    summary_box(io, "VisualizationCallback", setup)
  end
end


function VisualizationCallback(; interval=0,
                                 solution_variables=cons2prim,
                                 variable_names=[],
                                 plot_data_creator=PlotData2D,
                                 show_mesh=false,
                                 plot_arguments...)
  mpi_isparallel() && error("this callback not work in parallel yet")

  visualization_callback = VisualizationCallback(interval, Dict{Symbol,Any}(plot_arguments),
                                                 solution_variables, variable_names,
                                                 plot_data_creator, show_mesh)

  DiscreteCallback(visualization_callback, visualization_callback, # the first one is the condition, the second the affect!
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:VisualizationCallback}
  visualization_callback = cb.affect!

  visualization_callback(integrator)

  return nothing
end


# this method is called to determine whether the callback should be activated
function (visualization_callback::VisualizationCallback)(u, t, integrator)
  @unpack interval = visualization_callback

  return interval > 0 && ((integrator.iter % interval == 0) || isfinished(integrator))
end


# this method is called when the callback is activated
function (visualization_callback::VisualizationCallback)(integrator)
  u_ode = integrator.u
  semi = integrator.p
  @unpack plot_arguments, solution_variables, variable_names, plot_data_creator, show_mesh = visualization_callback

  # Extract plot data
  plot_data = plot_data_creator(u_ode, semi, solution_variables=solution_variables)

  # If variable names were not specified, plot everything
  if isempty(variable_names)
    variable_names = String[keys(plot_data)...]
  end

  # Create plot
  create_plot(plot_data, variable_names, show_mesh, plot_arguments)

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


function create_plot(plot_data, variable_names::Vector{String}, show_mesh, plot_arguments)
  # Gather subplots
  plots = []
  for v in variable_names
    push!(plots, plot(plot_data[v], plot_arguments...))
  end
  if show_mesh
    push!(plots, plot(getmesh(plot_data), plot_arguments...))
  end

  # Determine layout
  cols = ceil(Int, sqrt(length(plots)))
  rows = ceil(Int, length(plots)/cols)
  layout = (rows, cols)

  # Show plot
  display(plot(plots..., layout=layout))
end
create_plot(plot_data, variable_name::String, args...) = create_plot(plot_data, [variable_name], args...)
