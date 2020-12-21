
mutable struct VisualizationCallback{SolutionVariables, PlotType}
  interval::Int
  solution_variables::SolutionVariables
  plot_type::PlotType
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:VisualizationCallback}
  visualization_callback = cb.affect!
  @unpack interval, solution_variables, plot_type = visualization_callback
  print(io, "VisualizationCallback(",
            "interval=", interval, ", ",
            "solution_variables=", solution_variables, ", ",
            "plot_type=", plot_type, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:VisualizationCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    visualization_callback = cb.affect!

    setup = [
             "interval" => visualization_callback.interval,
             "solution variables" => visualization_callback.solution_variables,
            ]
    summary_box(io, "VisualizationCallback", setup)
  end
end


function VisualizationCallback(; interval=0,
                                 solution_variables=cons2prim,
                                 plot_type=PlotData2D)
  mpi_isparallel() && error("this callback not work in parallel yet")

  visualization_callback = VisualizationCallback(interval, solution_variables, plot_type)

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

  # Extract plot data
  plot_data = visualization_callback.plot_type(u_ode, semi)

  # Create plot
  display(plot(plot_data))

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end
