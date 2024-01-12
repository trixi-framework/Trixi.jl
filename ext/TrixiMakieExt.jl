# Package extension for adding Makie-based features to Trixi.jl
module TrixiMakieExt

# Required for visualization code
if isdefined(Base, :get_extension)
    using Makie: Makie, GeometryBasics
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Makie: Makie, GeometryBasics
end

# Use all exported symbols to avoid having to rewrite `recipes_makie.jl`
using Trixi

# Use additional symbols that are not exported
using Trixi: PlotData2DTriangulated, TrixiODESolution, PlotDataSeries, ScalarData, @muladd,
             wrap_array_native, mesh_equations_solver_cache

# Import functions such that they can be extended with new methods
import Trixi: iplot, iplot!

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

"""
    iplot(u, mesh::UnstructuredMesh2D, equations, solver, cache;
          plot_mesh=true, show_axis=false, colormap=default_Makie_colormap(),
          variable_to_plot_in=1)

Creates an interactive surface plot of the solution and mesh for an `UnstructuredMesh2D` type.

Keywords:
- variable_to_plot_in: variable to show by default

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function iplot end

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
    wire_mesh_top = Makie.lines!(ax, wire_points, color = :white)
    wire_mesh_bottom = Makie.lines!(ax, wire_points, color = :white)
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
        [Makie.Point(point.data[1:2]..., z_offset) for point in wire_points]
    end
    flat_wire_points = Makie.@lift get_flat_points($wire_points, $z_offset)
    wire_mesh_flat = Makie.lines!(ax, flat_wire_points, color = :black)

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

    # This syncs the toggle buttons to the mesh plots.
    Makie.connect!(wire_mesh_top.visible, toggle_solution_mesh.active)
    Makie.connect!(wire_mesh_bottom.visible, toggle_solution_mesh.active)
    Makie.connect!(wire_mesh_flat.visible, toggle_mesh.active)

    # On OSX, shift-command-4 for screenshots triggers a constant "up-zoom".
    # To avoid this, we remap up-zoom to the right shift button instead.
    Makie.cameracontrols(ax.scene).attributes[:up_key][] = Makie.Keyboard.right_shift

    # typing this pulls up the figure (similar to display(plot!()) in Plots.jl)
    fig
end

function iplot(u, mesh, equations, solver, cache;
               solution_variables = nothing, nvisnodes = 2 * nnodes(solver), kwargs...)
    @assert ndims(mesh) == 2

    pd = PlotData2DTriangulated(u, mesh, equations, solver, cache;
                                solution_variables = solution_variables,
                                nvisnodes = nvisnodes)

    iplot(pd; kwargs...)
end

# redirect `iplot(sol)` to dispatchable `iplot` signature.
iplot(sol::TrixiODESolution; kwargs...) = iplot(sol.u[end], sol.prob.p; kwargs...)
function iplot(u, semi; kwargs...)
    iplot(wrap_array_native(u, semi), mesh_equations_solver_cache(semi)...; kwargs...)
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
    Makie.Theme(colormap = default_Makie_colormap())
end

function Makie.plot!(myplot::TrixiHeatmap)
    pds = myplot[:plot_data_series][]

    plotting_mesh = global_plotting_triangulation_makie(pds;
                                                        set_z_coordinate_zero = true)

    pd = pds.plot_data
    solution_z = vec(StructArrays.component(pd.data, pds.variable_id))
    Makie.mesh!(myplot, plotting_mesh, color = solution_z, shading = false,
                colormap = myplot[:colormap])
    myplot.colorrange = extrema(solution_z)

    # Makie hides keyword arguments within `myplot`; see also
    # https://github.com/JuliaPlots/Makie.jl/issues/837#issuecomment-845985070
    plot_mesh = if haskey(myplot, :plot_mesh)
        myplot.plot_mesh[]
    else
        true # default to plotting the mesh
    end

    if plot_mesh
        xyz_wireframe = convert_PlotData2D_to_mesh_Points(pds;
                                                          set_z_coordinate_zero = true)
        Makie.lines!(myplot, xyz_wireframe, color = :lightgrey)
    end

    myplot
end

# redirects Makie.plot(pd::PlotDataSeries) to custom recipe TrixiHeatmap(pd)
Makie.plottype(::Trixi.PlotDataSeries{<:Trixi.PlotData2DTriangulated}) = TrixiHeatmap

# Makie does not yet support layouts in its plot recipes, so we overload `Makie.plot` directly.
function Makie.plot(sol::TrixiODESolution;
                    plot_mesh = false, solution_variables = nothing,
                    colormap = default_Makie_colormap())
    return Makie.plot(PlotData2DTriangulated(sol; solution_variables); plot_mesh,
                      colormap)
end

function Makie.plot(pd::PlotData2DTriangulated, fig = Makie.Figure();
                    plot_mesh = false, colormap = default_Makie_colormap())
    figAxes = Makie.plot!(fig, pd; plot_mesh, colormap)
    display(figAxes.fig)
    return figAxes
end

function Makie.plot!(fig, pd::PlotData2DTriangulated;
                     plot_mesh = false, colormap = default_Makie_colormap())
    # Create layout that is as square as possible, when there are more than 3 subplots.
    # This is done with a preference for more columns than rows if not.
    if length(pd) <= 3
        cols = length(pd)
        rows = 1
    else
        cols = ceil(Int, sqrt(length(pd)))
        rows = cld(length(pd), cols)
    end

    axes = [Makie.Axis(fig[i, j], xlabel = "x", ylabel = "y")
            for j in 1:rows, i in 1:cols]
    row_list, col_list = ([i for j in 1:rows, i in 1:cols],
                          [j for j in 1:rows, i in 1:cols])

    for (variable_to_plot, (variable_name, pds)) in enumerate(pd)
        ax = axes[variable_to_plot]
        plt = trixiheatmap!(ax, pds; plot_mesh, colormap)

        row = row_list[variable_to_plot]
        col = col_list[variable_to_plot]
        Makie.Colorbar(fig[row, col][1, 2], colormap = colormap)

        ax.aspect = Makie.DataAspect() # equal aspect ratio
        ax.title = variable_name
        Makie.xlims!(ax, extrema(pd.x))
        Makie.ylims!(ax, extrema(pd.y))
    end

    return FigureAndAxes(fig, axes)
end
end # @muladd

end
