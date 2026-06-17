# Package extension for adding Makie-based features to Trixi.jl
module TrixiMakieExt

# Required for visualization code
using Makie: Makie, GeometryBasics
using LaTeXStrings: latexstring

# Use all exported symbols to avoid having to rewrite `recipes_makie.jl`
using Trixi

# Use additional symbols that are not exported
using Trixi: @muladd, AbstractPlotData, PlotMesh, PlotDataSeries, ScalarData,
             PlotData1D, PlotData2D, PlotData2DCartesian, PlotData2DTriangulated,
             TrixiODESolution,
             wrap_array_native, mesh_equations_solver_cache

# Import functions such that they can be extended with new methods
import Trixi: iplot, iplot!, trixiheatmap, trixiheatmap!

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# First some utilities
# Given a reference plotting triangulation, this function generates a plotting triangulation for
# the entire global mesh. The output can be plotted using `Makie.mesh`.
function global_plotting_triangulation_makie(pds::PlotDataSeries{<:PlotData2DTriangulated};
                                             set_z_coordinate_zero = false)
    @unpack variable_id = pds
    pd = pds.plot_data
    @unpack x, y, data, t = pd

    makie_triangles = Makie.to_triangles(t)

    # trimesh[i] holds GeometryBasics.Mesh containing plotting information on the ith element.
    # Note: Float32 is required by GeometryBasics
    num_plotting_nodes, num_elements = size(x)
    trimesh = Vector{GeometryBasics.Mesh{3, Float32}}(undef, num_elements)
    coordinates = zeros(Float32, num_plotting_nodes, 3)
    for element in Base.OneTo(num_elements)
        for i in Base.OneTo(num_plotting_nodes)
            coordinates[i, 1] = x[i, element]
            coordinates[i, 2] = y[i, element]
            if set_z_coordinate_zero == false
                coordinates[i, 3] = data[i, element][variable_id]
            end
        end
        trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),
                                                      makie_triangles)
    end
    plotting_mesh = merge([trimesh...]) # merge meshes on each element into one large mesh
    return plotting_mesh
end

# helper function to extract the arguments for `Makie.tricontourf` from a `PlotDataSeries{<:PlotData2DTriangulated}`
function tricontourf_arguments(pds::PlotDataSeries{<:PlotData2DTriangulated})
    @unpack variable_id = pds
    pd = pds.plot_data
    @unpack x, y, data, t = pd

    num_plotting_nodes, num_elements = size(x)
    num_reference_triangles = size(t, 1)
    triangles = Matrix{Int}(undef, 3, num_reference_triangles * num_elements)

    for element in Base.OneTo(num_elements)
        offset = (element - 1) * num_plotting_nodes
        for triangle in Base.OneTo(num_reference_triangles)
            triangle_id = triangle + (element - 1) * num_reference_triangles
            triangles[:, triangle_id] .= @views t[triangle, :] .+ offset
        end
    end

    return vec(x), vec(y), vec(StructArrays.component(data, variable_id)), triangles
end

# Returns a list of `Makie.Point`s which can be used to plot the mesh, or a solution "wireframe"
# (e.g., a plot of the mesh lines but with the z-coordinate equal to the value of the solution).
function convert_PlotData2D_to_mesh_Points(pds::PlotDataSeries{<:PlotData2DTriangulated};
                                           set_z_coordinate_zero = false)
    @unpack variable_id = pds
    pd = pds.plot_data
    @unpack x_face, y_face, face_data = pd

    if set_z_coordinate_zero
        # plot 2d surface by setting z coordinate to zero.
        # Uses `x_face` since `face_data` may be `::Nothing`, as it's not used for 2D plots.
        sol_f = zeros(eltype(first(x_face)), size(x_face))
    else
        sol_f = StructArrays.component(face_data, variable_id)
    end

    # This line separates solution lines on each edge by NaNs to ensure that they are rendered
    # separately. The coordinates `xf`, `yf` and the solution `sol_f`` are assumed to be a matrix
    # whose columns correspond to different elements. We add NaN separators by appending a row of
    # NaNs to this matrix. We also flatten (e.g., apply `vec` to) the result, as this speeds up
    # plotting.
    xyz_wireframe = GeometryBasics.Point.(map(x -> vec(vcat(x,
                                                            fill(NaN, 1, size(x, 2)))),
                                              (x_face, y_face, sol_f))...)

    return xyz_wireframe
end

# Creates a GeometryBasics triangulation for the visualization of a ScalarData2D plot object.
function global_plotting_triangulation_makie(pd::PlotData2DTriangulated{<:ScalarData};
                                             set_z_coordinate_zero = false)
    @unpack x, y, data, t = pd

    makie_triangles = Makie.to_triangles(t)

    # trimesh[i] holds GeometryBasics.Mesh containing plotting information on the ith element.
    # Note: Float32 is required by GeometryBasics
    num_plotting_nodes, num_elements = size(x)
    trimesh = Vector{GeometryBasics.Mesh{3, Float32}}(undef, num_elements)
    coordinates = zeros(Float32, num_plotting_nodes, 3)
    for element in Base.OneTo(num_elements)
        for i in Base.OneTo(num_plotting_nodes)
            coordinates[i, 1] = x[i, element]
            coordinates[i, 2] = y[i, element]
            if set_z_coordinate_zero == false
                coordinates[i, 3] = data.data[i, element]
            end
        end
        trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates),
                                                      makie_triangles)
    end
    plotting_mesh = merge([trimesh...]) # merge meshes on each element into one large mesh
    return plotting_mesh
end

# Returns a list of `GeometryBasics.Point`s which can be used to plot the mesh, or a solution "wireframe"
# (e.g., a plot of the mesh lines but with the z-coordinate equal to the value of the solution).
function convert_PlotData2D_to_mesh_Points(pd::PlotData2DTriangulated{<:ScalarData};
                                           set_z_coordinate_zero = false)
    @unpack x_face, y_face, face_data = pd

    if set_z_coordinate_zero
        # plot 2d surface by setting z coordinate to zero.
        # Uses `x_face` since `face_data` may be `::Nothing`, as it's not used for 2D plots.
        sol_f = zeros(eltype(first(x_face)), size(x_face))
    else
        sol_f = face_data
    end

    # This line separates solution lines on each edge by NaNs to ensure that they are rendered
    # separately. The coordinates `xf`, `yf` and the solution `sol_f`` are assumed to be a matrix
    # whose columns correspond to different elements. We add NaN separators by appending a row of
    # NaNs to this matrix. We also flatten (e.g., apply `vec` to) the result, as this speeds up
    # plotting.
    xyz_wireframe = GeometryBasics.Point.(map(x -> vec(vcat(x,
                                                            fill(NaN, 1, size(x, 2)))),
                                              (x_face, y_face, sol_f))...)

    return xyz_wireframe
end

# We set the Makie default colormap to match Plots.jl, which uses `:inferno` by default.
default_Makie_colormap() = :inferno

function _makie_guide(orientation)
    label = orientation == 1 ? "x" :
            orientation == 2 ? "y" : orientation == 3 ? "z" : ""
    isempty(label) && return label
    return latexstring("\$", label, "\$")
end

# Format colorbar tick labels with enough significant figures to distinguish marks,
# avoiding Makie's default scientific notation
function _trixi_colorbar_tickformat(values)
    isempty(values) && return String[]
    vmin, vmax = extrema(values)
    range_val = vmax - vmin
    sigfigs = range_val > 0 ? max(3, ceil(Int, -log10(range_val)) + 2) : 4
    return [string(round(v; sigdigits = sigfigs)) for v in values]
end

# Return a non-degenerate (umin, umax) for colorbars; expands zero-width ranges
# so CairoMakie does not produce NaN when all data values are identical.
function _trixi_colorbar_limits(umin, umax)
    isapprox(umin, umax) || return (umin, umax)
    delta = max(one(umin), abs(umin))
    return (umin - delta, umax + delta)
end

# convenience struct for editing Makie plots after they're created.
struct FigureAndAxes{Axes}
    fig::Makie.Figure
    axes::Axes
end

# for "quiet" return arguments to Makie.plot(::TrixiODESolution) and
# Makie.plot(::PlotData2DTriangulated)
Base.show(io::IO, fa::FigureAndAxes) = nothing

# allows for returning fig, axes = Makie.plot(...)
function Base.iterate(fa::FigureAndAxes, state = 1)
    if state == 1
        return (fa.fig, 2)
    elseif state == 2
        return (fa.axes, 3)
    else
        return nothing
    end
end

# Enables `iplot(PlotData2D(sol))`.
function iplot(pd::PlotData2DTriangulated;
               plot_mesh = true, show_axis = false, colormap = default_Makie_colormap(),
               variable_to_plot_in = 1)
    @unpack variable_names = pd

    # Initialize a Makie figure that we'll add the solution and toggle switches to.
    fig = Makie.Figure()

    # Set up options for the drop-down menu
    menu_options = [zip(variable_names, 1:length(variable_names))...]
    menu = Makie.Menu(fig, options = menu_options)

    # Initialize toggle switches for viewing the mesh
    toggle_solution_mesh = Makie.Toggle(fig, active = plot_mesh)
    toggle_mesh = Makie.Toggle(fig, active = plot_mesh)

    # Add dropdown menu and toggle switches to the left side of the figure.
    fig[1, 1] = Makie.vgrid!(Makie.Label(fig, "Solution field", width = nothing), menu,
                             Makie.Label(fig, "Solution mesh visible"),
                             toggle_solution_mesh,
                             Makie.Label(fig, "Mesh visible"), toggle_mesh;
                             tellheight = false, width = 200)

    # Create a zoomable interactive axis object on top of which to plot the solution.
    ax = Makie.LScene(fig[1, 2], scenekw = (show_axis = show_axis,))

    # Initialize the dropdown menu to `variable_to_plot_in`
    # Since menu.selection is an Observable type, we need to dereference it using `[]` to set.
    menu.selection[] = variable_to_plot_in
    menu.i_selected[] = variable_to_plot_in

    # Since `variable_to_plot` is an Observable, these lines are re-run whenever `variable_to_plot[]`
    # is updated from the drop-down menu.
    plotting_mesh = Makie.@lift(global_plotting_triangulation_makie(getindex(pd,
                                                                             variable_names[$(menu.selection)])))
    solution_z = Makie.@lift(getindex.($plotting_mesh.position, 3))

    # Plot the actual solution.
    Makie.mesh!(ax, plotting_mesh; color = solution_z, colormap)

    # Create a mesh overlay by plotting a mesh both on top of and below the solution contours.
    wire_points = Makie.@lift(convert_PlotData2D_to_mesh_Points(getindex(pd,
                                                                         variable_names[$(menu.selection)])))
    wire_mesh_top = Makie.lines!(ax, wire_points, color = :white,
                                 visible = toggle_solution_mesh.active)
    wire_mesh_bottom = Makie.lines!(ax, wire_points, color = :white,
                                    visible = toggle_solution_mesh.active)
    Makie.translate!(wire_mesh_top, 0, 0, 1e-3)
    Makie.translate!(wire_mesh_bottom, 0, 0, -1e-3)

    # This draws flat mesh lines below the solution.
    function compute_z_offset(solution_z)
        zmin = minimum(solution_z)
        zrange = (x -> x[2] - x[1])(extrema(solution_z))
        return zmin - 0.25 * zrange
    end
    z_offset = Makie.@lift(compute_z_offset($solution_z))
    function get_flat_points(wire_points, z_offset)
        return [Makie.Point(point.data[1:2]..., z_offset) for point in wire_points]
    end
    flat_wire_points = Makie.@lift get_flat_points($wire_points, $z_offset)
    wire_mesh_flat = Makie.lines!(ax, flat_wire_points, color = :black,
                                  visible = toggle_mesh.active)

    # create a small variation in the extrema to avoid the Makie `range_step` cannot be zero error.
    # see https://github.com/MakieOrg/Makie.jl/issues/931 for more details.
    # the colorbar range is perturbed by 1e-5 * the magnitude of the solution.
    function scaled_extrema(x)
        ex = extrema(x)
        if ex[2] ≈ ex[1] # if solution is close to constant, perturb colorbar
            return ex .+ 1e-5 .* maximum(abs.(ex)) .* (-1, 1)
        else
            return ex
        end
    end

    # Resets the colorbar each time the solution changes.
    Makie.Colorbar(fig[1, 3], limits = Makie.@lift(scaled_extrema($solution_z)),
                   colormap = colormap)

    # On OSX, shift-command-4 for screenshots triggers a constant "up-zoom".
    # To avoid this, we remap up-zoom to the right shift button instead.
    Makie.cameracontrols(ax.scene).controls.up_key = Makie.Keyboard.right_shift

    # typing this pulls up the figure (similar to display(plot!()) in Plots.jl)
    return fig
end

function iplot(u, mesh, equations, solver, cache;
               solution_variables = nothing, nvisnodes = 2 * nnodes(solver), kwargs...)
    @assert ndims(mesh) == 2

    pd = PlotData2DTriangulated(u, mesh, equations, solver, cache;
                                solution_variables = solution_variables,
                                nvisnodes = nvisnodes)

    return iplot(pd; kwargs...)
end

# redirect `iplot(sol)` to dispatchable `iplot` signature.
iplot(sol::TrixiODESolution; kwargs...) = iplot(sol.u[end], sol.prob.p; kwargs...)
function iplot(u, semi; kwargs...)
    return iplot(wrap_array_native(u, semi), mesh_equations_solver_cache(semi)...;
                 kwargs...)
end

# Interactive visualization of user-defined ScalarData.
function iplot(pd::PlotData2DTriangulated{<:ScalarData};
               show_axis = false, colormap = default_Makie_colormap(),
               plot_mesh = false)
    fig = Makie.Figure()

    # Create a zoomable interactive axis object on top of which to plot the solution.
    ax = Makie.LScene(fig[1, 1], scenekw = (show_axis = show_axis,))

    # plot the user-defined ScalarData
    fig_axis_plt = iplot!(FigureAndAxes(fig, ax), pd; colormap = colormap,
                          plot_mesh = plot_mesh)

    fig
    return fig_axis_plt
end

function iplot!(fig_axis::Union{FigureAndAxes, Makie.FigureAxisPlot},
                pd::PlotData2DTriangulated{<:ScalarData};
                colormap = default_Makie_colormap(), plot_mesh = false)

    # destructure first two fields of either FigureAndAxes or Makie.FigureAxisPlot
    fig, ax = fig_axis

    # create triangulation of the scalar data to plot
    plotting_mesh = global_plotting_triangulation_makie(pd)
    solution_z = getindex.(plotting_mesh.position, 3)
    plt = Makie.mesh!(ax, plotting_mesh; color = solution_z, colormap)

    if plot_mesh
        wire_points = convert_PlotData2D_to_mesh_Points(pd)
        wire_mesh_top = Makie.lines!(ax, wire_points, color = :white)
        wire_mesh_bottom = Makie.lines!(ax, wire_points, color = :white)
        Makie.translate!(wire_mesh_top, 0, 0, 1e-3)
        Makie.translate!(wire_mesh_bottom, 0, 0, -1e-3)
    end

    # Add a colorbar to the rightmost part of the layout
    Makie.Colorbar(fig[1, end + 1], plt)

    fig
    return Makie.FigureAxisPlot(fig, ax, plt)
end

# ================== new Makie plot recipes ====================

# This initializes a Makie recipe, which creates a new type definition which Makie uses to create
# custom `trixiheatmap` plots. See also https://docs.makie.org/stable/documentation/recipes/
Makie.@recipe(TrixiHeatmap, plot_data_series) do scene
    return Makie.Theme(colormap = default_Makie_colormap(),
                       plot_mesh = false)
end

function Makie.plot!(myplot::TrixiHeatmap)
    pds = myplot[:plot_data_series][]

    plotting_mesh = global_plotting_triangulation_makie(pds;
                                                        set_z_coordinate_zero = true)

    pd = pds.plot_data
    solution_z = vec(StructArrays.component(pd.data, pds.variable_id))
    Makie.mesh!(myplot, plotting_mesh, color = solution_z, shading = Makie.NoShading,
                colormap = myplot[:colormap])
    umin, umax = extrema(solution_z)
    myplot.colorrange = if isapprox(umin, umax)
        delta = max(one(umin), abs(umin))
        (umin - delta, umax + delta)
    else
        (umin, umax)
    end

    if myplot.plot_mesh[]
        xyz_wireframe = convert_PlotData2D_to_mesh_Points(pds;
                                                          set_z_coordinate_zero = true)
        Makie.lines!(myplot, xyz_wireframe, color = :lightgrey)
    end

    return myplot
end

# redirects Makie.plot(pd::PlotDataSeries) to custom recipe TrixiHeatmap(pd)
Makie.plottype(::Trixi.PlotDataSeries{<:Trixi.PlotData2DTriangulated}) = TrixiHeatmap
Makie.plottype(::PlotDataSeries{<:AbstractPlotData{1}}) = Makie.Lines
Makie.plottype(::PlotDataSeries{<:AbstractPlotData{2}}) = Makie.Heatmap

# Makie recipe for 1D PlotDataSeries
function Makie.convert_arguments(::Type{<:Makie.Plot},
                                 pds::PlotDataSeries{<:AbstractPlotData{1}})
    @unpack plot_data, variable_id = pds
    @unpack x, data = plot_data
    return (x, data[:, variable_id])
end

function Makie.plot(pds::PlotDataSeries{<:AbstractPlotData{1}}, fig = Makie.Figure();
                    plot_mesh = false, kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack x, variable_names, mesh_vertices_x = plot_data
    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(plot_data.orientation_x))
    plt = Makie.lines!(ax, pds; kwargs...)
    Makie.xlims!(ax, x[begin], x[end])
    if plot_mesh
        Makie.vlines!(ax, mesh_vertices_x; color = :grey, linewidth = 1)
    end
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.plot!(pm::PlotMesh{<:AbstractPlotData{1}}; kwargs...)
    ax = Makie.current_axis()
    plt = Makie.vlines!(ax, pm.plot_data.mesh_vertices_x;
                        color = :grey, linewidth = 1, kwargs...)
    display(Makie.current_figure())
    return plt
end

function Makie.plot(pd::PlotData1D, fig = Makie.Figure(); plot_mesh = false)
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        @unpack x, mesh_vertices_x = pds.plot_data
        ax = Makie.Axis(fig[row, col],
                        title = variable_name,
                        xlabel = _makie_guide(pd.orientation_x))
        axes[row, col] = ax
        Makie.lines!(ax, pds)
        Makie.xlims!(ax, x[begin], x[end])
        if plot_mesh
            Makie.vlines!(ax, mesh_vertices_x; color = :grey, linewidth = 1)
        end
    end

    display(fig)
    return FigureAndAxes(fig, axes)
end

# Makie recipe for 2D PlotDataSeries
function Makie.convert_arguments(::Type{<:Makie.Plot},
                                 pds::PlotDataSeries{<:PlotData2DCartesian})
    @unpack plot_data, variable_id = pds
    @unpack x, y, data = plot_data
    return (x, y, permutedims(data[variable_id])) # permutedims to match the axis convention of Plots.jl
end

function Makie.plot(pds::PlotDataSeries{<:PlotData2DCartesian},
                    fig = Makie.Figure(); kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack x, y, variable_names = plot_data
    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(plot_data.orientation_x),
                    ylabel = _makie_guide(plot_data.orientation_y))
    plt = Makie.heatmap!(ax, pds; colormap = default_Makie_colormap(), kwargs...)
    Makie.Colorbar(fig[1, 2], plt; ticks = Makie.WilkinsonTicks(3; k_max = 4),
                   tickformat = _trixi_colorbar_tickformat)
    ax.aspect = Makie.DataAspect()
    Makie.xlims!(ax, x[begin], x[end])
    Makie.ylims!(ax, y[begin], y[end])
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.plot!(pm::PlotMesh{<:PlotData2DCartesian}; kwargs...)
    ax = Makie.current_axis()
    @unpack mesh_vertices_x, mesh_vertices_y = pm.plot_data
    plt = Makie.lines!(ax, mesh_vertices_x, mesh_vertices_y;
                       color = :grey, linewidth = 1, kwargs...)
    display(Makie.current_figure())
    return plt
end

function Makie.plot(pd::PlotData2DCartesian, fig = Makie.Figure();
                    plot_mesh = false, colormap = default_Makie_colormap())
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        @unpack x, y, mesh_vertices_x, mesh_vertices_y = pds.plot_data
        ax = Makie.Axis(fig[row, col][1, 1],
                        title = variable_name,
                        xlabel = _makie_guide(pd.orientation_x),
                        ylabel = _makie_guide(pd.orientation_y),
                        xticks = Makie.WilkinsonTicks(3; k_max = 4),
                        yticks = Makie.WilkinsonTicks(3; k_max = 4))
        axes[row, col] = ax
        data_matrix = pds.plot_data.data[pds.variable_id]
        umin, umax = extrema(data_matrix)
        colorrange = if isapprox(umin, umax)
            delta = max(one(umin), abs(umin))
            (umin - delta, umax + delta)
        else
            (umin, umax)
        end
        plt = Makie.heatmap!(ax, pds; colormap, colorrange)
        Makie.Colorbar(fig[row, col][1, 2], plt;
                       ticks = Makie.WilkinsonTicks(3; k_max = 4),
                       tickformat = _trixi_colorbar_tickformat)
        ax.aspect = Makie.DataAspect()
        Makie.xlims!(ax, x[begin], x[end])
        Makie.ylims!(ax, y[begin], y[end])
        if plot_mesh
            Makie.lines!(ax, mesh_vertices_x, mesh_vertices_y;
                         color = :grey, linewidth = 1)
        end
    end

    display(fig)
    return FigureAndAxes(fig, axes)
end

function Makie.convert_arguments(::Type{<:Makie.Contour},
                                 pds::PlotDataSeries{<:PlotData2DCartesian})
    @unpack plot_data, variable_id = pds
    @unpack x, y, data = plot_data
    # permutedims to match the axis convention of Plots.jl
    return (x, y, permutedims(data[variable_id]))
end

function Makie.contour(pds::PlotDataSeries{<:PlotData2DCartesian},
                       fig = Makie.Figure();
                       plot_mesh = false, colorbar = true,
                       colormap = default_Makie_colormap(), kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack x, y, variable_names, mesh_vertices_x, mesh_vertices_y = plot_data
    z = permutedims(plot_data.data[variable_id])
    umin, umax = extrema(z)
    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(plot_data.orientation_x),
                    ylabel = _makie_guide(plot_data.orientation_y))
    plt = isapprox(umin, umax) ? Makie.lines!(ax, Float64[], Float64[]) :
          Makie.contour!(ax, pds; colormap, kwargs...)
    if colorbar
        Makie.Colorbar(fig[1, 2]; colormap,
                       limits = _trixi_colorbar_limits(umin, umax),
                       ticks = Makie.WilkinsonTicks(3; k_max = 4),
                       tickformat = _trixi_colorbar_tickformat)
    end
    ax.aspect = Makie.DataAspect()
    Makie.xlims!(ax, x[begin], x[end])
    Makie.ylims!(ax, y[begin], y[end])
    if plot_mesh
        Makie.lines!(ax, mesh_vertices_x, mesh_vertices_y;
                     color = :grey, linewidth = 1)
    end
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.contour!(pds::PlotDataSeries{<:PlotData2DCartesian}; kwargs...)
    plt = Makie.contour!(Makie.current_axis(), pds; kwargs...)
    display(Makie.current_figure())
    return plt
end

function Makie.contour(pd::PlotData2DCartesian, fig = Makie.Figure();
                       plot_mesh = false, colorbar = true,
                       colormap = default_Makie_colormap(), kwargs...)
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        @unpack x, y, mesh_vertices_x, mesh_vertices_y = pds.plot_data
        z = permutedims(pds.plot_data.data[pds.variable_id])
        umin, umax = extrema(z)
        ax = Makie.Axis(fig[row, col][1, 1],
                        title = variable_name,
                        xlabel = _makie_guide(pd.orientation_x),
                        ylabel = _makie_guide(pd.orientation_y),
                        xticks = Makie.WilkinsonTicks(3; k_max = 4),
                        yticks = Makie.WilkinsonTicks(3; k_max = 4))
        axes[row, col] = ax
        isapprox(umin, umax) || Makie.contour!(ax, pds; colormap, kwargs...)
        if colorbar
            Makie.Colorbar(fig[row, col][1, 2]; colormap,
                           limits = _trixi_colorbar_limits(umin, umax),
                           ticks = Makie.WilkinsonTicks(3; k_max = 4),
                           tickformat = _trixi_colorbar_tickformat)
        end

        ax.aspect = Makie.DataAspect()
        Makie.xlims!(ax, x[begin], x[end])
        Makie.ylims!(ax, y[begin], y[end])
        if plot_mesh
            Makie.lines!(ax, mesh_vertices_x, mesh_vertices_y;
                         color = :grey, linewidth = 1)
        end
    end

    display(fig)
    return FigureAndAxes(fig, axes)
end

# Makie does not yet support layouts in its plot recipes, so we overload `Makie.plot` directly.
function Makie.plot(sol::TrixiODESolution; solution_variables = nothing, kwargs...)
    if ndims(sol.prob.p) == 1
        return Makie.plot(PlotData1D(sol; solution_variables); kwargs...)
    else
        pd = PlotData2D(sol; solution_variables) # use Julias dispatch here
        return Makie.plot(pd; kwargs...)
    end
end

function Makie.contour(sol::TrixiODESolution; solution_variables = nothing, kwargs...)
    if ndims(sol.prob.p) == 1
        throw(ArgumentError("Contour plots are not supported for 1D solutions."))
    else
        pd = PlotData2D(sol; solution_variables)
        return Makie.contour(pd; kwargs...)
    end
end

function Makie.tricontourf(sol::TrixiODESolution; solution_variables = nothing,
                           kwargs...)
    if ndims(sol.prob.p) == 1
        throw(ArgumentError("Tricontourf plots are not supported for 1D solutions."))
    else
        pd = PlotData2D(sol; solution_variables)
        if !(pd isa PlotData2DTriangulated)
            throw(ArgumentError("Tricontourf plots require triangulated 2D plot data."))
        end
        return Makie.contourf(pd; kwargs...)
    end
end

function Makie.convert_arguments(::Type{<:Makie.Contourf},
                                 pds::PlotDataSeries{<:PlotData2DCartesian})
    @unpack plot_data, variable_id = pds
    @unpack x, y, data = plot_data
    # permutedims to match the axis convention of Plots.jl
    return (x, y, permutedims(data[variable_id]))
end

function Makie.contourf(pds::PlotDataSeries{<:PlotData2DCartesian},
                        fig = Makie.Figure();
                        plot_mesh = false, colorbar = true,
                        colormap = default_Makie_colormap(), kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack x, y, variable_names, mesh_vertices_x, mesh_vertices_y = plot_data
    z = permutedims(plot_data.data[variable_id])
    umin, umax = extrema(z)
    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(plot_data.orientation_x),
                    ylabel = _makie_guide(plot_data.orientation_y))
    plt = isapprox(umin, umax) ? Makie.lines!(ax, Float64[], Float64[]) :
          Makie.contourf!(ax, pds; colormap, kwargs...)
    if colorbar
        cb_limits = _trixi_colorbar_limits(umin, umax)
        isapprox(umin, umax) ?
        Makie.Colorbar(fig[1, 2]; colormap, limits = cb_limits,
                       ticks = Makie.WilkinsonTicks(3; k_max = 4),
                       tickformat = _trixi_colorbar_tickformat) :
        Makie.Colorbar(fig[1, 2], plt; ticks = Makie.WilkinsonTicks(3; k_max = 4),
                       tickformat = _trixi_colorbar_tickformat)
    end
    ax.aspect = Makie.DataAspect()
    Makie.xlims!(ax, x[begin], x[end])
    Makie.ylims!(ax, y[begin], y[end])
    if plot_mesh
        Makie.lines!(ax, mesh_vertices_x, mesh_vertices_y;
                     color = :grey, linewidth = 1)
    end
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.contourf(pd::PlotData2DCartesian, fig = Makie.Figure();
                        plot_mesh = false, colorbar = true,
                        colormap = default_Makie_colormap(), kwargs...)
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        @unpack x, y, mesh_vertices_x, mesh_vertices_y = pds.plot_data
        z = permutedims(pds.plot_data.data[pds.variable_id])
        umin, umax = extrema(z)
        ax = Makie.Axis(fig[row, col][1, 1],
                        title = variable_name,
                        xlabel = _makie_guide(pd.orientation_x),
                        ylabel = _makie_guide(pd.orientation_y),
                        xticks = Makie.WilkinsonTicks(3; k_max = 4),
                        yticks = Makie.WilkinsonTicks(3; k_max = 4))
        axes[row, col] = ax
        if isapprox(umin, umax)
            if colorbar
                Makie.Colorbar(fig[row, col][1, 2]; colormap,
                               limits = _trixi_colorbar_limits(umin, umax),
                               ticks = Makie.WilkinsonTicks(3; k_max = 4),
                               tickformat = _trixi_colorbar_tickformat)
            end
        else
            plt = Makie.contourf!(ax, pds; colormap, kwargs...)
            if colorbar
                Makie.Colorbar(fig[row, col][1, 2], plt;
                               ticks = Makie.WilkinsonTicks(3; k_max = 4),
                               tickformat = _trixi_colorbar_tickformat)
            end
        end
        ax.aspect = Makie.DataAspect()
        Makie.xlims!(ax, x[begin], x[end])
        Makie.ylims!(ax, y[begin], y[end])
        if plot_mesh
            Makie.lines!(ax, mesh_vertices_x, mesh_vertices_y;
                         color = :grey, linewidth = 1)
        end
    end

    display(fig)
    return FigureAndAxes(fig, axes)
end

# Returns the (x, y) crossing points where isoline u=c crosses the edges of a triangle.
function edge_crossing(x_el, y_el, u_el, i, j, k, c)
    crossings = Tuple{Float64, Float64}[]

    for (a, b) in ((i, j), (j, k), (k, i))
        ua, ub = u_el[a], u_el[b]
        if (ua - c) * (ub - c) < 0
            t = (c - ua) / (ub - ua)
            push!(crossings,
                  (x_el[a] + t * (x_el[b] - x_el[a]),
                   y_el[a] + t * (y_el[b] - y_el[a])))
        end
    end
    return crossings
end

# Returns one (xs, ys) segment vector per level.
# Works element-wise, reusing the triangulation t for each element.
function contour_lines_triangulated(pd::PlotData2DTriangulated, variable_id,
                                    levels::AbstractVector)
    @unpack x, y, data, t = pd
    u = StructArrays.component(data, variable_id)
    n_levels = length(levels)
    xs = [Float64[] for _ in 1:n_levels]
    ys = [Float64[] for _ in 1:n_levels]
    for element in axes(x, 2)
        x_el = view(x, :, element)
        y_el = view(y, :, element)
        u_el = view(u, :, element)
        for tri in eachrow(t)
            i, j, k = tri[1], tri[2], tri[3]
            for (li, c) in enumerate(levels)
                crossings = edge_crossing(x_el, y_el, u_el, i, j, k, c)
                if length(crossings) == 2
                    push!(xs[li], crossings[1][1], crossings[2][1], NaN)
                    push!(ys[li], crossings[1][2], crossings[2][2], NaN)
                end
            end
        end
    end
    return xs, ys
end

function Makie.contour(pds::PlotDataSeries{<:PlotData2DTriangulated},
                       fig = Makie.Figure();
                       plot_mesh = false, colorbar = true, levels = 10,
                       colormap = default_Makie_colormap(), linewidth = 1.5,
                       kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack variable_names = plot_data
    u = StructArrays.component(plot_data.data, variable_id)
    umin, umax = extrema(u)
    level_values = levels isa Integer ? LinRange(umin, umax, levels) : levels

    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(1), ylabel = _makie_guide(2),
                    aspect = Makie.DataAspect())

    cmap = Makie.cgrad(colormap)
    xs, ys = contour_lines_triangulated(plot_data, variable_id, level_values)
    last_plt = nothing
    for (li, c) in enumerate(level_values)
        isempty(xs[li]) && continue
        t_color = umax > umin ? (c - umin) / (umax - umin) : 0.5
        last_plt = Makie.lines!(ax, xs[li], ys[li]; color = cmap[t_color],
                                linewidth, kwargs...)
    end
    if colorbar
        Makie.Colorbar(fig[1, 2]; colormap,
                       limits = _trixi_colorbar_limits(umin, umax),
                       ticks = Makie.WilkinsonTicks(3; k_max = 4),
                       tickformat = _trixi_colorbar_tickformat)
    end
    Makie.xlims!(ax, extrema(plot_data.x))
    Makie.ylims!(ax, extrema(plot_data.y))
    if plot_mesh
        @unpack x_face, y_face = plot_data
        x_wire = vec(vcat(x_face, fill(NaN, 1, size(x_face, 2))))
        y_wire = vec(vcat(y_face, fill(NaN, 1, size(y_face, 2))))
        Makie.lines!(ax, x_wire, y_wire; color = :grey, linewidth = 1)
    end
    return Makie.FigureAxisPlot(fig, ax,
                                isnothing(last_plt) ?
                                Makie.lines!(ax, Float64[],
                                             Float64[]) :
                                last_plt)
end

function Makie.contour(pd::PlotData2DTriangulated, fig = Makie.Figure();
                       plot_mesh = false, colorbar = true, levels = 10,
                       colormap = default_Makie_colormap(), linewidth = 1.5,
                       kwargs...)
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        @unpack plot_data, variable_id = pds
        u = StructArrays.component(plot_data.data, variable_id)
        umin, umax = extrema(u)
        level_values = levels isa Integer ? LinRange(umin, umax, levels) : levels

        ax = Makie.Axis(fig[row, col][1, 1],
                        title = variable_name,
                        xlabel = _makie_guide(1), ylabel = _makie_guide(2),
                        aspect = Makie.DataAspect(),
                        xticks = Makie.WilkinsonTicks(3; k_max = 4),
                        yticks = Makie.WilkinsonTicks(3; k_max = 4))
        axes[row, col] = ax

        cmap = Makie.cgrad(colormap)
        xs, ys = contour_lines_triangulated(plot_data, variable_id, level_values)
        for (li, c) in enumerate(level_values)
            isempty(xs[li]) && continue
            t_color = umax > umin ? (c - umin) / (umax - umin) : 0.5
            Makie.lines!(ax, xs[li], ys[li]; color = cmap[t_color], linewidth,
                         kwargs...)
        end
        if colorbar
            Makie.Colorbar(fig[row, col][1, 2]; colormap,
                           limits = _trixi_colorbar_limits(umin, umax),
                           ticks = Makie.WilkinsonTicks(3; k_max = 4),
                           tickformat = _trixi_colorbar_tickformat)
        end
        Makie.xlims!(ax, extrema(pd.x))
        Makie.ylims!(ax, extrema(pd.y))
        if plot_mesh
            @unpack x_face, y_face = plot_data
            x_wire = vec(vcat(x_face, fill(NaN, 1, size(x_face, 2))))
            y_wire = vec(vcat(y_face, fill(NaN, 1, size(y_face, 2))))
            Makie.lines!(ax, x_wire, y_wire; color = :grey, linewidth = 1)
        end
    end

    display(fig)
    return FigureAndAxes(fig, axes)
end

function Makie.contour!(ax, pds::PlotDataSeries{<:PlotData2DTriangulated};
                        levels = 10, color = nothing,
                        colormap = :viridis,
                        linewidth = 1.5, plot_mesh = false, kwargs...)
    @unpack plot_data, variable_id = pds
    u = StructArrays.component(plot_data.data, variable_id)
    umin, umax = extrema(u)
    isapprox(umin, umax) && return Makie.lines!(ax, Float64[], Float64[])
    level_values = levels isa Integer ? LinRange(umin, umax, levels) : levels
    xs, ys = contour_lines_triangulated(plot_data, variable_id, level_values)
    last_plt = nothing
    if isnothing(color)
        cmap = Makie.cgrad(colormap)
        for (li, c) in enumerate(level_values)
            isempty(xs[li]) && continue
            t_color = umax > umin ? (c - umin) / (umax - umin) : 0.5
            last_plt = Makie.lines!(ax, xs[li], ys[li]; color = cmap[t_color],
                                    linewidth, kwargs...)
        end
    else
        for li in eachindex(level_values)
            isempty(xs[li]) && continue
            last_plt = Makie.lines!(ax, xs[li], ys[li]; color, linewidth, kwargs...)
        end
    end
    if plot_mesh
        x_wire = vec(vcat(plot_data.x_face, fill(NaN, 1, size(plot_data.x_face, 2))))
        y_wire = vec(vcat(plot_data.y_face, fill(NaN, 1, size(plot_data.y_face, 2))))
        Makie.lines!(ax, x_wire, y_wire; color = :grey, linewidth = 1)
    end
    return isnothing(last_plt) ? Makie.lines!(ax, Float64[], Float64[]) : last_plt
end

function Makie.contour!(pds::PlotDataSeries{<:PlotData2DTriangulated}; kwargs...)
    plt = Makie.contour!(Makie.current_axis(), pds; kwargs...)
    display(Makie.current_figure())
    return plt
end

function Makie.contourf!(ax, pds::PlotDataSeries{<:PlotData2DTriangulated};
                         plot_mesh = false, colormap = default_Makie_colormap(),
                         triangulation = nothing, kwargs...)
    x, y, z, triangles = tricontourf_arguments(pds)
    umin, umax = extrema(z)
    plt = if isapprox(umin, umax)
        trixiheatmap!(ax, pds; plot_mesh = false, colormap)
    else
        triangulation_ = isnothing(triangulation) ? triangles : triangulation
        Makie.tricontourf!(ax, x, y, z; triangulation = triangulation_,
                           colormap, kwargs...)
    end

    if plot_mesh
        pd = pds.plot_data
        x_wire = vec(vcat(pd.x_face, fill(NaN, 1, size(pd.x_face, 2))))
        y_wire = vec(vcat(pd.y_face, fill(NaN, 1, size(pd.y_face, 2))))
        Makie.lines!(ax, x_wire, y_wire; color = :grey, linewidth = 1)
    end

    return plt
end

function Makie.contourf(pds::PlotDataSeries{<:PlotData2DTriangulated},
                        fig = Makie.Figure();
                        plot_mesh = false, colorbar = true,
                        colormap = default_Makie_colormap(), kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack variable_names = plot_data
    u = StructArrays.component(plot_data.data, variable_id)
    umin, umax = extrema(u)
    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(1), ylabel = _makie_guide(2),
                    aspect = Makie.DataAspect())
    plt = Makie.contourf!(ax, pds; plot_mesh, colormap, kwargs...)
    if colorbar
        if isapprox(umin, umax)
            Makie.Colorbar(fig[1, 2]; colormap,
                           limits = _trixi_colorbar_limits(umin, umax),
                           ticks = Makie.WilkinsonTicks(3; k_max = 4),
                           tickformat = _trixi_colorbar_tickformat)
        else
            Makie.Colorbar(fig[1, 2], plt;
                           ticks = Makie.WilkinsonTicks(3; k_max = 4),
                           tickformat = _trixi_colorbar_tickformat)
        end
    end
    Makie.xlims!(ax, extrema(plot_data.x))
    Makie.ylims!(ax, extrema(plot_data.y))
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.contourf(pd::PlotData2DTriangulated, fig = Makie.Figure();
                        plot_mesh = false, colorbar = true,
                        colormap = default_Makie_colormap(), kwargs...)
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        @unpack plot_data, variable_id = pds
        u = StructArrays.component(plot_data.data, variable_id)
        umin, umax = extrema(u)
        ax = Makie.Axis(fig[row, col][1, 1],
                        title = variable_name,
                        xlabel = _makie_guide(1), ylabel = _makie_guide(2),
                        aspect = Makie.DataAspect(),
                        xticks = Makie.WilkinsonTicks(3; k_max = 4),
                        yticks = Makie.WilkinsonTicks(3; k_max = 4))
        axes[row, col] = ax
        plt = Makie.contourf!(ax, pds; plot_mesh, colormap, kwargs...)
        if colorbar
            if isapprox(umin, umax)
                Makie.Colorbar(fig[row, col][1, 2]; colormap,
                               limits = _trixi_colorbar_limits(umin, umax),
                               ticks = Makie.WilkinsonTicks(3; k_max = 4),
                               tickformat = _trixi_colorbar_tickformat)
            else
                Makie.Colorbar(fig[row, col][1, 2], plt;
                               ticks = Makie.WilkinsonTicks(3; k_max = 4),
                               tickformat = _trixi_colorbar_tickformat)
            end
        end
        Makie.xlims!(ax, extrema(pd.x))
        Makie.ylims!(ax, extrema(pd.y))
    end

    display(fig)
    return FigureAndAxes(fig, axes)
end

function Makie.plot(pds::PlotDataSeries{<:PlotData2DTriangulated},
                    fig = Makie.Figure();
                    plot_mesh = false, colormap = default_Makie_colormap(), kwargs...)
    @unpack plot_data, variable_id = pds
    @unpack variable_names = plot_data
    ax = Makie.Axis(fig[1, 1],
                    title = variable_names[variable_id],
                    xlabel = _makie_guide(1), ylabel = _makie_guide(2),
                    aspect = Makie.DataAspect())
    plt = trixiheatmap!(ax, pds; plot_mesh, colormap, kwargs...)
    Makie.Colorbar(fig[1, 2], plt; ticks = Makie.WilkinsonTicks(3; k_max = 4),
                   tickformat = _trixi_colorbar_tickformat)
    Makie.xlims!(ax, extrema(plot_data.x))
    Makie.ylims!(ax, extrema(plot_data.y))
    return Makie.FigureAxisPlot(fig, ax, plt)
end

function Makie.plot!(pm::PlotMesh{<:PlotData2DTriangulated}; kwargs...)
    ax = Makie.current_axis()
    @unpack x_face, y_face = pm.plot_data
    x_wire = vec(vcat(x_face, fill(NaN, 1, size(x_face, 2))))
    y_wire = vec(vcat(y_face, fill(NaN, 1, size(y_face, 2))))
    plt = Makie.lines!(ax, x_wire, y_wire; color = :grey, linewidth = 1, kwargs...)
    display(Makie.current_figure())
    return plt
end

function Makie.plot(pd::PlotData2DTriangulated, fig = Makie.Figure();
                    plot_mesh = false, colormap = default_Makie_colormap())
    figAxes = Makie.plot!(fig, pd; plot_mesh, colormap)
    display(figAxes.fig)
    return figAxes
end

function Makie.plot!(fig, pd::PlotData2DTriangulated;
                     plot_mesh = false, colormap = default_Makie_colormap())
    n = length(pd)
    cols = n <= 3 ? n : ceil(Int, sqrt(n))
    rows = cld(n, cols)

    axes = Matrix{Makie.Axis}(undef, rows, cols)
    for (i, (variable_name, pds)) in enumerate(pd)
        row, col = cld(i, cols), mod1(i, cols)
        ax = Makie.Axis(fig[row, col][1, 1],
                        title = variable_name,
                        xlabel = _makie_guide(1), ylabel = _makie_guide(2),
                        xticks = Makie.WilkinsonTicks(3; k_max = 4),
                        yticks = Makie.WilkinsonTicks(3; k_max = 4))
        axes[row, col] = ax
        plt = trixiheatmap!(ax, pds; plot_mesh, colormap)
        Makie.Colorbar(fig[row, col][1, 2], plt;
                       ticks = Makie.WilkinsonTicks(3; k_max = 4),
                       tickformat = _trixi_colorbar_tickformat)
        ax.aspect = Makie.DataAspect()
        Makie.xlims!(ax, extrema(pd.x))
        Makie.ylims!(ax, extrema(pd.y))
    end

    return FigureAndAxes(fig, axes)
end
end # @muladd

end
