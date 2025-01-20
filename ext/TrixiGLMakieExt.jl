# Package extension for adding Makie-based features to Trixi.jl
module TrixiGLMakieExt

__precompile__(false)

using GLMakie

# Use all exported symbols to avoid having to rewrite `recipes_makie.jl`
using Trixi
using Trixi: @muladd

@muladd begin
#! format: noindent

#converts a single int into a tuple of ints, to get a square arrangement for example f(1) = (1,1) f(2) = (2,1) f(3) = (2,2) f(4) = (1,2)
function makieLayoutHelper(n)
    if n == 1
        return (1,1)
    end
    t = makieLayoutHelper(n-1)
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
ndims = (visualization_callback.plot_data_creator == PlotData2D) ? 2 : 3
maxes = [maximum(plot_data.data[v]) for v in 1:size(variable_names)[1]]
mins = [minimum(plot_data.data[v]) for v in 1:size(variable_names)[1]]
max = maximum(maxes)
min = minimum(mins)
limits = (min, max)
one_if_show_mesh = show_mesh ? 1 : 0
    if visualization_callback.figure === nothing
        @warn "Creating new figure"
        visualization_callback.figure = Figure()
        if show_mesh 
            if ndims == 2
                push!(visualization_callback.axis, Axis(visualization_callback.figure[makieLayoutHelper(1)...], title = "mesh"))
            else
                push!(visualization_callback.axis, Axis3(visualization_callback.figure[makieLayoutHelper(1)...], aspect=:equal, title = "mesh"))
                lines!(visualization_callback.axis[1], plot_data.mesh_vertices_x, plot_data.mesh_vertices_y, plot_data.mesh_vertices_z, color=:black)
            end
        end
        for v in 1:size(variable_names)[1]
            push!(visualization_callback.axis, (ndims == 2) ? Axis(visualization_callback.figure[makieLayoutHelper(v + one_if_show_mesh)...], title = variable_names[v]) : 
            Axis3(visualization_callback.figure[makieLayoutHelper(v + one_if_show_mesh)...], aspect=:equal, title = variable_names[v]))
        end
        visualization_callback.colorbar = Colorbar(visualization_callback.figure[makieLayoutHelper(size(variable_names)[1] + 2)...], colorrange = limits)
        display(visualization_callback.figure)
    else
        if ndims == 2
            for v in 1:size(variable_names)[1]
                empty!(visualization_callback.axis[v + one_if_show_mesh])
                heatmap!(visualization_callback.axis[v + one_if_show_mesh], plot_data.x, plot_data.y, plot_data.data[v], transparent = true, colorrange = limits)
                # if show_mesh
                #     lines!(visualization_callback.axis[v + one_if_show_mesh], plot_data.mesh_vertices_y, plot_data.mesh_vertices_x, color=:black)
                # end
            end
        else
            for v in 1:size(variable_names)[1]
                empty!(visualization_callback.axis[v + one_if_show_mesh])
                volume!(visualization_callback.axis[v + one_if_show_mesh], plot_data.x, plot_data.y, plot_data.z, plot_data.data[v], transparent = true, colorrange = limits)
                # if show_mesh 
                #     lines!(visualization_callback.axis[v + one_if_show_mesh], plot_data.mesh_vertices_z, plot_data.mesh_vertices_y, plot_data.mesh_vertices_x, color=:black)
                # end
            end
        end
    end

    visualization_callback.colorbar.colorrange = limits

    if show_mesh 
        if ndims == 2
            empty!(visualization_callback.axis[1])
            lines!(visualization_callback.axis[1], plot_data.mesh_vertices_y, plot_data.mesh_vertices_x, color=:black)
        else
            empty!(visualization_callback.axis[1])
            lines!(visualization_callback.axis[1], plot_data.mesh_vertices_z, plot_data.mesh_vertices_y, plot_data.mesh_vertices_x, color=:black)
        end
    end
# TODO: show_mesh
end

end # @muladd

end
