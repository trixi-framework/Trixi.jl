# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

mutable struct VisualizationCallback{SolutionVariables, VariableNames, PlotDataCreator,
                                     PlotCreator}
    interval::Int
    solution_variables::SolutionVariables
    variable_names::VariableNames
    show_mesh::Bool
    plot_data_creator::PlotDataCreator
    plot_creator::PlotCreator
    figure
    axis
    plot_arguments::Dict{Symbol, Any}
end

function Base.show(io::IO,
                   cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                    Affect! <:
                                                                    VisualizationCallback
                                                                    }
    visualization_callback = cb.affect!
    @unpack interval, plot_arguments, solution_variables, variable_names, show_mesh, plot_creator, plot_data_creator = visualization_callback
    print(io, "VisualizationCallback(",
          "interval=", interval, ", ",
          "solution_variables=", solution_variables, ", ",
          "variable_names=", variable_names, ", ",
          "show_mesh=", show_mesh, ", ",
          "plot_data_creator=", plot_data_creator, ", ",
          "plot_creator=", plot_creator, ", ",
          "plot_arguments=", plot_arguments, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{Condition, Affect!}) where {Condition,
                                                                    Affect! <:
                                                                    VisualizationCallback
                                                                    }
    if get(io, :compact, false)
        show(io, cb)
    else
        visualization_callback = cb.affect!

        setup = [
            "interval" => visualization_callback.interval,
            "plot arguments" => visualization_callback.plot_arguments,
            "solution variables" => visualization_callback.solution_variables,
            "variable names" => visualization_callback.variable_names,
            "show mesh" => visualization_callback.show_mesh,
            "plot creator" => visualization_callback.plot_creator,
            "plot data creator" => visualization_callback.plot_data_creator
        ]
        summary_box(io, "VisualizationCallback", setup)
    end
end

"""
    VisualizationCallback(; interval=0,
                            solution_variables=cons2prim,
                            variable_names=[],
                            show_mesh=false,
                            plot_data_creator=PlotData2D,
                            plot_creator=show_plot,
                            plot_arguments...)

Create a callback that visualizes results during a simulation, also known as *in-situ
visualization*.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in any future releases.

The `interval` specifies the number of time step iterations after which a new plot is generated. The
available variables to plot are configured with the `solution_variables` parameter, which acts the
same way as for the [`SaveSolutionCallback`](@ref). The variables to be actually plotted can be
selected by providing a single string or a list of strings to `variable_names`, and if `show_mesh`
is `true`, an additional plot with the mesh will be generated.

To customize the generated figure, `plot_data_creator` allows to use different plot data types. With
`plot_creator` you can further specify an own function to visualize results, which must support the
same interface as the default implementation [`show_plot`](@ref). All remaining
keyword arguments are collected and passed as additional arguments to the plotting command.
"""
function VisualizationCallback(; interval = 0,
                               solution_variables = cons2prim,
                               variable_names = [],
                               show_mesh = false,
                               plot_data_creator = PlotData2D,
                               plot_creator = show_plot,
                               plot_arguments...)
    mpi_isparallel() && error("this callback does not work in parallel yet")

    if variable_names isa String
        variable_names = String[variable_names]
    end

    visualization_callback = VisualizationCallback(interval,
                                                   solution_variables, variable_names,
                                                   show_mesh,
                                                   plot_data_creator, plot_creator, nothing, [],
                                                   Dict{Symbol, Any}(plot_arguments))

    DiscreteCallback(visualization_callback, visualization_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: VisualizationCallback}
    visualization_callback = cb.affect!
    u_ode = integrator.u
    semi = integrator.p
    mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
    if ndims(mesh) == 3 && visualization_callback.plot_data_creator == PlotData2D
        visualization_callback.plot_data_creator = PlotData3D
    end
    visualization_callback(integrator)
    return nothing
end

# this method is called to determine whether the callback should be activated
function (visualization_callback::VisualizationCallback)(u, t, integrator)
    @unpack interval = visualization_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return interval > 0 && (integrator.stats.naccept % interval == 0 ||
            isfinished(integrator))
end

# this method is called when the callback is activated
function (visualization_callback::VisualizationCallback)(integrator)
    u_ode = integrator.u
    semi = integrator.p
    @unpack plot_arguments, solution_variables, variable_names, show_mesh, plot_data_creator, plot_creator, figure, axis = visualization_callback
    mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
    n = Trixi.ndims(mesh)

    # Extract plot data
    plot_data = plot_data_creator(u_ode, semi, solution_variables = solution_variables)

    # If variable names were not specified, plot everything
    if isempty(variable_names)
        variable_names = String[keys(plot_data)...]
    end

    # Create plot
    plot_creator(n, visualization_callback, plot_data, variable_names;
                 show_mesh = show_mesh, plot_arguments = plot_arguments,
                 time = integrator.t, timestep = integrator.stats.naccept, figure = figure, axis = axis)

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

"""
    show_plot(plot_data, variable_names;
              show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
              time=nothing, timestep=nothing, figure=nothing)

Visualize the plot data object provided in `plot_data` and display result, plotting only the
variables in `variable_names` and, optionally, the mesh (if `show_mesh` is `true`).  Additionally,
`plot_arguments` will be unpacked and passed as keyword arguments to the `Plots.plot` command.

This function is the default `plot_creator` argument for the [`VisualizationCallback`](@ref).
`time` and `timestep` are currently unused by this function.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`VisualizationCallback`](@ref), [`save_plot`](@ref)
"""
function show_plot(ndims, visualization_callback, plot_data, variable_names;
                   show_mesh = true, plot_arguments = Dict{Symbol, Any}(),
                   time = nothing, timestep = nothing, figure = nothing, axis = nothing)
    # Gather subplots
    plots = []
    for v in variable_names
        push!(plots, Plots.plot(plot_data[v]; plot_arguments...))
    end
    if show_mesh
        push!(plots, Plots.plot(getmesh(plot_data); plot_arguments...))
    end

    # Note, for the visualization callback to work for general equation systems
    # this layout construction would need to use the if-logic below.
    # Currently, there is no use case for this so it is left here as a note.
    #
    # Determine layout
    # if length(plots) <= 3
    #   cols = length(plots)
    #   rows = 1
    # else
    #   cols = ceil(Int, sqrt(length(plots)))
    #   rows = div(length(plots), cols, RoundUp)
    # end
    # layout = (rows, cols)

    # Determine layout
    cols = ceil(Int, sqrt(length(plots)))
    rows = div(length(plots), cols, RoundUp)
    layout = (rows, cols)

    # Show plot
    display(Plots.plot(plots..., layout = layout))
end

"""
    save_plot(plot_data, variable_names;
              show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
              time=nothing, timestep=nothing)

Visualize the plot data object provided in `plot_data` and save result as a PNG file in the `out`
directory, plotting only the variables in `variable_names` and, optionally, the mesh (if `show_mesh`
is `true`).  Additionally, `plot_arguments` will be unpacked and passed as keyword arguments to the
`Plots.plot` command.

The `timestep` is used in the filename. `time` is currently unused by this function.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`VisualizationCallback`](@ref), [`show_plot`](@ref)
"""
function save_plot(plot_data, variable_names;
                   show_mesh = true, plot_arguments = Dict{Symbol, Any}(),
                   time = nothing, timestep = nothing)
    # Gather subplots
    plots = []
    for v in variable_names
        push!(plots, Plots.plot(plot_data[v]; plot_arguments...))
    end
    if show_mesh
        push!(plots, Plots.plot(getmesh(plot_data); plot_arguments...))
    end

    # Determine layout
    cols = ceil(Int, sqrt(length(plots)))
    rows = div(length(plots), cols, RoundUp)
    layout = (rows, cols)

    # Create plot
    Plots.plot(plots..., layout = layout)

    # Determine filename and save plot
    filename = joinpath("out", @sprintf("solution_%09d.png", timestep))
    Plots.savefig(filename)
end


"""
    show_plot_makie(plot_data, variable_names;
                    show_mesh=true, plot_arguments=Dict{Symbol,Any}(),
                    time=nothing, timestep=nothing, figure=nothing)

Visualize the plot data object provided in `plot_data` and display result using Makie.
Only the variables in `variable_names` are plotted and, optionally, the mesh (if
`show_mesh` is `true`). Additionally, `plot_arguments` will be unpacked and passed as
keyword arguments to the `Plots.plot` command.

This function is the default `plot_creator` argument for the [`VisualizationCallback`](@ref).
`time` and `timestep` are currently unused by this function.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

See also: [`VisualizationCallback`](@ref), [`save_plot`](@ref)
"""
function show_plot_makie(ndims, visualization_callback, plot_data, variable_names;
    show_mesh = true, plot_arguments = Dict{Symbol, Any}(),
    time = nothing, timestep = nothing, figure = nothing, axis = []) end
end # @muladd
