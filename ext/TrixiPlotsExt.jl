module TrixiPlotsExt

# Load the required packages
using Plots: Plots
using Trixi
using Trixi: getmesh
using MuladdMacro: @muladd
using Printf: @sprintf

@muladd begin
#! format: noindent

function Trixi.show_plot(plot_data, variable_names;
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
    return display(Plots.plot(plots..., layout = layout))
end

function Trixi.save_plot(plot_data, variable_names;
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
    return Plots.savefig(filename)
end
end # @muladd
end # module TrixiPlotsExt
