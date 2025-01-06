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

function Trixi.show_plot_makie(ndims, visualization_callback, plot_data, variable_names;
show_mesh = true, plot_arguments = Dict{Symbol, Any}(),
time = nothing, timestep = nothing, figure = nothing, axis = [])
    if figure === nothing
        @warn "Creating new figure"
        visualization_callback.figure = GLMakie.Figure()
        figure = visualization_callback.figure
        for v in 1:size(variable_names)[1]
            push!(axis, (ndims == 2) ? GLMakie.Axis(figure[makieLayoutHelper(v)...], title = variable_names[v]) : 
            GLMakie.Axis3(figure[makieLayoutHelper(v)...], aspect=:equal, title = variable_names[v]))
        end
        visualization_callback.axis = axis
        GLMakie.display(figure)
    else
        if ndims == 2
            for v in 1:size(variable_names)[1]
                GLMakie.heatmap!(axis[v], plot_data.x, plot_data.y, plot_data.data[v])
            end
        else
            for v in 1:size(variable_names)[1]
                GLMakie.volume!(axis[v], plot_data.data[v])
            end
        end
    end

# TODO: show_mesh
end

end # @muladd

end
