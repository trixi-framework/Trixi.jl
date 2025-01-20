# Package extension for adding Makie-based features to Trixi.jl
module TrixiGLMakieExt

# Required for visualization code
using GLMakie: Figure, Axis, display, heatmap!

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, @unpack, @muladd, FigureAndAxes

@muladd begin
#! format: noindent

# converts a single int into a tuple of ints, to get a square arrangement
# example: f(1) = (1,1) f(2) = (2,1) f(3) = (2,2) f(4) = (1,2)
function makie_layout_helper(n)
    if n == 1
        return (1, 1)
    end
    t = makie_layout_helper(n - 1)
    if t[1] == 1
        return (t[2] + 1, 1)
    elseif t[1] > t[2]
        return (t[1], t[2] + 1)
    elseif t[2] >= t[1]
        return (t[1] - 1, t[2])
    end
end

function Trixi.show_plot_makie(visualization_callback, plot_data, variable_names;
                               show_mesh = true, plot_arguments = Dict{Symbol, Any}(),
                               time = nothing, timestep = nothing)
    nvars = size(variable_names)[1]
    if visualization_callback.figure_axes.fig === nothing
        @info "Creating new GLMakie figure"
        fig = GLMakie.Figure()
        axes = [GLMakie.Axis(fig[makie_layout_helper(v)...], aspect = DataAspect(),
                             title = variable_names[v])
                for v in 1:nvars]
        if show_mesh
            push!(axes,
                  GLMakie.Axis(fig[makie_layout_helper(nvars + 1)...],
                               aspect = DataAspect(), title = "mesh"))
        end
        visualization_callback.figure_axes = FigureAndAxes(fig, axes)
        GLMakie.display(visualization_callback.figure_axes.fig)
    end

    @unpack axes = visualization_callback.figure_axes
    for v in 1:nvars
        GLMakie.heatmap!(axes[v], plot_data.x, plot_data.y,
                         permutedims(plot_data.data[v]); plot_arguments...)
    end
    if show_mesh
        empty!(axes[nvars + 1])
        lines!(axes[nvars + 1], plot_data.mesh_vertices_x, plot_data.mesh_vertices_y,
               color = :black)
    end
end
end # @muladd
end # module TrixiGLMakieExt
