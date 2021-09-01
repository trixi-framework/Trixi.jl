# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# We set the Makie default colormap to match Plots.jl, which uses `:inferno` by default.
default_Makie_colormap() = :inferno

"""
    iplot(u, mesh::UnstructuredMesh2D, equations, solver, cache;
          solution_variables=nothing, nvisnodes=nnodes(solver), variable_to_plot_in=1,
          colormap = default_Makie_colormap())

Creates an interactive surface plot of the solution and mesh for an `UnstructuredMesh2D` type.

Inputs:
- solution_variables: either `nothing` or a variable transformation function (e.g., `cons2prim`)
- nvisnodes: number of visualization nodes per dimension
- variable_to_plot_in: variable to show by default

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function iplot(u, mesh::UnstructuredMesh2D, equations, solver, cache;
               solution_variables=nothing, nvisnodes=2*nnodes(solver), variable_to_plot_in=1,
               colormap = default_Makie_colormap())

  pd = PlotData2D(u, mesh, equations, solver, cache;
                  solution_variables=solution_variables, nvisnodes=nvisnodes)

  @unpack variable_names = pd

  @assert ndims(mesh) == 2

  # Initialize a Makie figure that we'll add the solution and toggle switches to.
  fig = Makie.Figure()

  # Set up options for the drop-down menu
  menu_options = [zip(variable_names, 1:length(variable_names))...]
  menu = Makie.Menu(fig, options=menu_options)

  # Initialize toggle switches for viewing the mesh
  toggle_solution_mesh = Makie.Toggle(fig, active=true)
  toggle_mesh = Makie.Toggle(fig, active=true)

  # Add dropdown menu and toggle switches to the left side of the figure.
  fig[1, 1] = Makie.vgrid!(
      Makie.Label(fig, "Solution field", width=nothing), menu,
      Makie.Label(fig, "Solution mesh visible"), toggle_solution_mesh,
      Makie.Label(fig, "Mesh visible"), toggle_mesh;
      tellheight=false, width = 200
  )

  # Create a zoomable interactive axis object on top of which to plot the solution.
  ax = Makie.LScene(fig[1, 2], scenekw=(show_axis = false,))

  # Initialize the dropdown menu to `variable_to_plot_in`
  # Since menu.selection is an Observable type, we need to dereference it using `[]` to set.
  menu.selection[] = variable_to_plot_in
  menu.i_selected[] = variable_to_plot_in

  # Since `variable_to_plot` is an Observable, these lines are re-run whenever `variable_to_plot[]`
  # is updated from the drop-down menu.
  plotting_mesh = Makie.@lift(global_plotting_triangulation_makie(getindex(pd, variable_names[$(menu.selection)])))
  solution_z = Makie.@lift(getindex.($plotting_mesh.position, 3))

  # Plot the actual solution.
  Makie.mesh!(ax, plotting_mesh; color=solution_z, nvisnodes, colormap)

  # Create a mesh overlay by plotting a mesh both on top of and below the solution contours.
  wire_points = Makie.@lift(mesh_plotting_wireframe(getindex(pd, variable_names[$(menu.selection)])))
  wire_mesh_top = Makie.lines!(ax, wire_points, color=:white)
  wire_mesh_bottom = Makie.lines!(ax, wire_points, color=:white)
  Makie.translate!(wire_mesh_top, 0, 0, 1e-3)
  Makie.translate!(wire_mesh_bottom, 0, 0, -1e-3)

  # This draws flat mesh lines below the solution.
  function compute_z_offset(solution_z)
      zmin = minimum(solution_z)
      zrange = (x->x[2]-x[1])(extrema(solution_z))
      return zmin - .25*zrange
  end
  z_offset = Makie.@lift(compute_z_offset($solution_z))
  get_flat_points(wire_points, z_offset) = [Makie.Point(point.data[1:2]..., z_offset) for point in wire_points]
  flat_wire_points = Makie.@lift get_flat_points($wire_points, $z_offset)
  wire_mesh_flat = Makie.lines!(ax, flat_wire_points, color=:black)

  # Resets the colorbar each time the solution changes.
  Makie.Colorbar(fig[1, 3], limits = Makie.@lift(extrema($solution_z)), colormap=colormap, flipaxis = false)

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

# redirect `iplot(sol)` to dispatchable `iplot` signature.
iplot(sol::TrixiODESolution; kwargs...) = iplot(sol.u[end], sol.prob.p; kwargs...)
iplot(u, semi; kwargs...) = iplot(wrap_array_native(u, semi), mesh_equations_solver_cache(semi)...; kwargs...)


# ================== new Makie plot recipes ====================

# This initializes a Makie recipe, which creates a new type definition which Makie uses to create
# custom `trixiheatmap` plots. See also https://makie.juliaplots.org/stable/recipes.html
@Makie.recipe(TrixiHeatmap, plot_data_series) do scene
  Makie.Theme(
    colormap = default_Makie_colormap()
  )
end

function Makie.plot!(myplot::TrixiHeatmap)
  pds = myplot[:plot_data_series][]

  plotting_mesh = global_plotting_triangulation_makie(pds; set_z_coordinate_zero = true)

  @unpack variable_id = pds
  pd = pds.plot_data
  solution_z = vec(StructArrays.component(pd.data, variable_id))
  Makie.mesh!(myplot, plotting_mesh, color=solution_z, shading=false, colormap=myplot[:colormap])
  myplot.colorrange=extrema(solution_z)

  # Makie hides keyword arguments within `myplot`; see also
  # https://github.com/JuliaPlots/Makie.jl/issues/837#issuecomment-845985070
  plot_mesh = if haskey(myplot, :plot_mesh)
    myplot.plot_mesh[]
  else
    true # default to plotting the mesh
  end

  if plot_mesh
    xyz_wireframe = mesh_plotting_wireframe(pds; set_z_coordinate_zero = true)
    Makie.lines!(myplot, xyz_wireframe, color=:lightgrey)
  end

  myplot
end

# redirects Makie.plot(pd::PlotDataSeries) to custom recipe TrixiHeatmap(pd)
Makie.plottype(::Trixi.PlotDataSeries{<:Trixi.UnstructuredPlotData2D}) = TrixiHeatmap

# Makie does not yet support layouts in its plot recipes, so we overload `Makie.plot` directly.
function Makie.plot(sol::TrixiODESolution;
                    plot_mesh=true, solution_variables=nothing, colormap=default_Makie_colormap())
  return Makie.plot(PlotData2D(sol; solution_variables); plot_mesh, colormap)
end

# convenience struct for editing Makie plots after they're created.
struct FigureAndAxes{Axes}
  fig::Makie.Figure
  axes::Axes
end

# for "quiet" return arguments to Makie.plot(::TrixiODESolution) and
# Makie.plot(::UnstructuredPlotData2D)
Base.show(io::IO, fa::FigureAndAxes) = nothing

# allows for returning fig, axes = Makie.plot(...)
function Base.iterate(fa::FigureAndAxes, state=1)
  if state == 1
    return (fa.fig, 2)
  elseif state == 2
    return (fa.axes, 3)
  else
    return nothing
  end
end

function Makie.plot(pd::UnstructuredPlotData2D, fig=Makie.Figure();
                    plot_mesh=true, colormap=default_Makie_colormap())
  figAxes = Makie.plot!(fig, pd; plot_mesh, colormap)
  display(figAxes.fig)
  return figAxes
end

function Makie.plot!(fig, pd::UnstructuredPlotData2D;
                     plot_mesh=true, colormap=default_Makie_colormap())
  # Create layout that is as square as possible, when there are more than 3 subplots.
  # This is done with a preference for more columns than rows if not.
  if length(pd) <= 3
    cols = length(pd)
    rows = 1
  else
    cols = ceil(Int, sqrt(length(pd)))
    rows = cld(length(pd), cols)
  end

  axes = [Makie.Axis(fig[i,j], xlabel="x", ylabel="y") for j in 1:rows, i in 1:cols]

  for (variable_to_plot, (variable_name, pds)) in enumerate(pd)
    ax = axes[variable_to_plot]
    trixiheatmap!(ax, pds; plot_mesh, colormap)
    ax.aspect = Makie.DataAspect() # equal aspect ratio
    ax.title  = variable_name
    Makie.xlims!(ax, extrema(pd.x))
    Makie.ylims!(ax, extrema(pd.y))
  end

  return FigureAndAxes(fig, axes)
end


end # @muladd
