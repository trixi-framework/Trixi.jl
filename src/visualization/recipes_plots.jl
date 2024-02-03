# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Visualize a single variable in a 2D plot (default: heatmap)
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(pds::PlotDataSeries{<:AbstractPlotData{2}})
    @unpack plot_data, variable_id = pds
    @unpack x, y, data, variable_names, orientation_x, orientation_y = plot_data

    # Set geometric properties
    xlims --> (x[begin], x[end])
    ylims --> (y[begin], y[end])
    aspect_ratio --> :equal

    # Set annotation properties
    legend --> :none
    title --> variable_names[variable_id]
    colorbar --> :true
    xguide --> _get_guide(orientation_x)
    yguide --> _get_guide(orientation_y)

    # Set series properties
    seriestype --> :heatmap

    # Return data for plotting
    x, y, data[variable_id]
end

# Visualize the mesh in a 2D plot
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(pm::PlotMesh{<:AbstractPlotData{2}})
    @unpack plot_data = pm
    @unpack x, y, mesh_vertices_x, mesh_vertices_y = plot_data

    # Set geometric and annotation properties
    xlims --> (x[begin], x[end])
    ylims --> (y[begin], y[end])
    aspect_ratio --> :equal
    legend --> :none
    grid --> false

    # Set series properties
    seriestype --> :path
    linecolor --> :grey
    linewidth --> 1

    # Return data for plotting
    mesh_vertices_x, mesh_vertices_y
end

# Visualize the mesh in a 2D plot
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(pm::PlotMesh{<:PlotData2DCartesian{<:Any,
                                                                  <:AbstractVector{<:AbstractVector}}})
    @unpack plot_data = pm
    @unpack x, y, mesh_vertices_x, mesh_vertices_y = plot_data

    # Set geometric and annotation properties
    xlims --> (minimum(x), maximum(x))
    ylims --> (minimum(y), maximum(y))
    aspect_ratio --> :equal
    legend --> :none
    grid --> false

    # Set series properties
    seriestype --> :path
    linecolor --> :grey
    linewidth --> 1

    # Return data for plotting
    mesh_vertices_x, mesh_vertices_y
end

# Plot all available variables at once for convenience
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(pd::AbstractPlotData)
    # Create layout that is as square as possible, when there are more than 3 subplots.
    # This is done with a preference for more columns than rows if not.

    if length(pd) <= 3
        cols = length(pd)
        rows = 1
    else
        cols = ceil(Int, sqrt(length(pd)))
        rows = ceil(Int, length(pd) / cols)
    end

    layout := (rows, cols)

    # Plot all existing variables
    for (i, (variable_name, series)) in enumerate(pd)
        RecipesBase.@series begin
            subplot := i
            series
        end
    end

    # Fill remaining subplots with empty plot
    for i in (length(pd) + 1):(rows * cols)
        RecipesBase.@series begin
            subplot := i
            axis := false
            ticks := false
            legend := false
            [], []
        end
    end
end

# Plot a single variable.
RecipesBase.@recipe function f(pds::PlotDataSeries{<:AbstractPlotData{1}})
    @unpack plot_data, variable_id = pds
    @unpack x, data, variable_names, orientation_x = plot_data

    # Set geometric properties
    xlims --> (x[begin], x[end])

    # Set annotation properties
    legend --> :none
    title --> variable_names[variable_id]
    xguide --> _get_guide(orientation_x)

    # Return data for plotting
    x, data[:, variable_id]
end

# Plot the mesh as vertical lines from a PlotMesh object.
RecipesBase.@recipe function f(pm::PlotMesh{<:AbstractPlotData{1}})
    @unpack plot_data = pm
    @unpack x, mesh_vertices_x = plot_data

    # Set geometric and annotation properties
    xlims --> (x[begin], x[end])
    legend --> :none

    # Set series properties
    seriestype --> :vline
    linecolor --> :grey
    linewidth --> 1

    # Return data for plotting
    mesh_vertices_x
end

# Create a plot directly from a TrixiODESolution for convenience
# The plot is created by a PlotData1D or PlotData2D object.
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(sol::TrixiODESolution)
    # Redirect everything to the recipes below
    return sol.u[end], sol.prob.p
end

# Recipe for general semidiscretizations
# Note: If you change the defaults values here, you need to also change them in the PlotData1D or PlotData2D
#       constructor.
RecipesBase.@recipe function f(u, semi::AbstractSemidiscretization;
                               solution_variables = nothing)
    if ndims(semi) == 1
        return PlotData1D(u, semi; solution_variables = solution_variables)
    else
        return PlotData2D(u, semi; solution_variables = solution_variables)
    end
end

# Recipe specifically for TreeMesh-type solutions
# Note: If you change the defaults values here, you need to also change them in the PlotData1D or PlotData2D
#       constructor.
RecipesBase.@recipe function f(u, semi::SemidiscretizationHyperbolic{<:TreeMesh};
                               solution_variables = nothing,
                               grid_lines = true, max_supported_level = 11,
                               nvisnodes = nothing, slice = :xy,
                               point = (0.0, 0.0, 0.0), curve = nothing)
    # Create a PlotData1D or PlotData2D object depending on the dimension.
    if ndims(semi) == 1
        return PlotData1D(u, semi; solution_variables, nvisnodes, slice, point, curve)
    else
        return PlotData2D(u, semi;
                          solution_variables, grid_lines, max_supported_level,
                          nvisnodes, slice, point)
    end
end

# Series recipe for PlotData2DTriangulated
RecipesBase.@recipe function f(pds::PlotDataSeries{<:PlotData2DTriangulated})
    pd = pds.plot_data
    @unpack variable_id = pds
    @unpack x, y, data, t, variable_names = pd

    # extract specific solution field to plot
    data_field = zeros(eltype(first(data)), size(data))
    for (i, data_i) in enumerate(data)
        data_field[i] = data_i[variable_id]
    end

    legend --> false
    aspect_ratio --> 1
    title --> pd.variable_names[variable_id]
    xlims --> extrema(x)
    ylims --> extrema(y)
    xguide --> _get_guide(1)
    yguide --> _get_guide(2)
    seriestype --> :heatmap
    colorbar --> :true

    return DGTriPseudocolor(global_plotting_triangulation_triplot((x, y), data_field,
                                                                  t)...)
end

# Visualize a 2D mesh given an `PlotData2DTriangulated` object
RecipesBase.@recipe function f(pm::PlotMesh{<:PlotData2DTriangulated})
    pd = pm.plot_data
    @unpack x_face, y_face = pd

    # This line separates solution lines on each edge by NaNs to ensure that they are rendered
    # separately. The coordinates `xf`, `yf` and the solution `sol_f`` are assumed to be a matrix
    # whose columns correspond to different elements. We add NaN separators by appending a row of
    # NaNs to this matrix. We also flatten (e.g., apply `vec` to) the result, as this speeds up
    # plotting.
    x_face, y_face = map(x -> vec(vcat(x, fill(NaN, 1, size(x, 2)))), (x_face, y_face))

    xlims --> extrema(x_face)
    ylims --> extrema(y_face)
    aspect_ratio --> :equal
    legend --> :none

    # Set series properties
    seriestype --> :path
    linecolor --> :grey
    linewidth --> 1

    return x_face, y_face
end

# Visualizes a single scalar field. Intended for use with ScalarPlotData2D.
# Example usage: `plot(ScalarPlotData2D(u, semi))`.
RecipesBase.@recipe function f(pd::PlotData2DTriangulated{<:ScalarData})
    @unpack x, y, data, t, variable_names = pd

    title_string = isnothing(variable_names) ? "" : variable_names

    legend --> false
    aspect_ratio --> 1
    title --> title_string
    xlims --> extrema(x)
    ylims --> extrema(y)
    xguide --> _get_guide(1)
    yguide --> _get_guide(2)
    seriestype --> :heatmap
    colorbar --> :true

    # Since `data` is simply a ScalarData wrapper around the actual plot data, we pass in
    # `data.data` instead.
    return DGTriPseudocolor(global_plotting_triangulation_triplot((x, y), data.data,
                                                                  t)...)
end

RecipesBase.@recipe function f(cb::DiscreteCallback{<:Any, <:TimeSeriesCallback},
                               point_id::Integer)
    return cb.affect!, point_id
end

RecipesBase.@recipe function f(time_series_callback::TimeSeriesCallback,
                               point_id::Integer)
    return PlotData1D(time_series_callback, point_id)
end
end # @muladd
