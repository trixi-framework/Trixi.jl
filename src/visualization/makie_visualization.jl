#     compute_triangle_area(tri)
#
# Computes the area of a triangle given `tri`, which is a tuple of three points (vectors),
# using the [Shoelace_formula](https://en.wikipedia.org/wiki/Shoelace_formula).
function compute_triangle_area(tri)
    A, B, C = tri
    return 0.5 * (A[1] * (B[2] - C[2]) + B[1] * (C[2]-A[2]) + C[1] * (A[2] - B[2]))
end

#   reference_plotting_triangulation(reference_plotting_coordinates)
#
# Computes a triangulation of the points in `reference_plotting_coordinates`, which is a tuple containing
# vectors of plotting points on the reference element (e.g., reference_plotting_coordinates = (r,s)).
# The reference element is assumed to be [-1,1]^d.
#
# This function returns `t` which is a `3 x N_tri` Matrix{Int} containing indices of triangles in the
# triangulation of the plotting points, with zero-volume triangles removed.
#
# For example, r[t[1, i]] returns the first reference coordinate of the 1st point on the ith triangle.
function reference_plotting_triangulation(reference_plotting_coordinates, tol=50*eps())
  # on-the-fly triangulation of plotting nodes on the reference element
  tri_in = Triangulate.TriangulateIO()
  tri_in.pointlist = permutedims(hcat(reference_plotting_coordinates...))
  tri_out, _ = Triangulate.triangulate("Q", tri_in)
  triangles = tri_out.trianglelist

  # filter out sliver triangles
  has_volume = fill(true, size(triangles, 2))
  for i in axes(triangles, 2)
      ids = @view triangles[:, i]
      x_points = @view tri_out.pointlist[1, ids]
      y_points = @view tri_out.pointlist[2, ids]
      area = compute_triangle_area(zip(x_points, y_points))
      if abs(area) < tol
          has_volume[i] = false
      end
  end
  return permutedims(triangles[:, findall(has_volume)])
end

#   plotting_interpolation_matrix(dg; kwargs...)
#
# Interpolation matrix which maps discretization nodes to a set of plotting nodes.
# Defaults to the identity matrix of size `length(solver.basis.nodes)`, and interpolates
# to equispaced nodes for DGSEM (set by kwarg `nvisnodes` in the plotting function).
#
# Example:
# ```julia
# A = plotting_interpolation_matrix(dg)
# A * dg.basis.nodes # => vector of nodes at which to plot the solution
# ```
#
# Note: we cannot use UniformScaling to define the interpolation matrix since we use it with `kron`
# to define a multi-dimensional interpolation matrix later.
plotting_interpolation_matrix(dg; kwargs...) = I(length(dg.basis.nodes))

function face_plotting_interpolation_matrix(dg::DGSEM; nvisnodes=2*length(dg.basis.nodes))
  return polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1, 1, nvisnodes))
end

function plotting_interpolation_matrix(dg::DGSEM; nvisnodes=2*length(dg.basis.nodes))
  Vp1D = polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1, 1, nvisnodes))
  # For quadrilateral elements, interpolation to plotting nodes involves applying a 1D interpolation
  # operator to each line of nodes. This is equivalent to multiplying the vector containing all node
  # node coordinates on an element by a Kronecker product of the 1D interpolation operator (e.g., a
  # multi-dimensional interpolation operator).
  return kron(Vp1D, Vp1D)
end

function reference_node_coordinates_2d(dg::DGSEM)
  @unpack nodes = dg.basis
  r = vec([nodes[i] for i in eachnode(dg), j in eachnode(dg)])
  s = vec([nodes[j] for i in eachnode(dg), j in eachnode(dg)])
  return r, s
end

function transform_to_solution_variables!(u, solution_variables, equations)
  for (i, u_i) in enumerate(u)
    u[i] = solution_variables(u_i, equations)
  end
end

# specializes the PlotData2D constructor to return an UnstructuredPlotData2D if the mesh is
# an UnstructuredMesh2D type.
function PlotData2D(u, mesh::UnstructuredMesh2D, equations, dg::DGSEM, cache;
                    solution_variables=nothing, nvisnodes=2*polydeg(dg))
  @assert ndims(mesh) == 2

  n_nodes_2d = nnodes(dg)^ndims(mesh)
  n_elements = nelements(dg, cache)

  # build nodes on reference element (seems to be the right ordering)
  r, s = reference_node_coordinates_2d(dg)

  # reference plotting nodes
  plotting_interp_matrix = plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)

  # create triangulation for plotting nodes
  r_plot, s_plot = (x->plotting_interp_matrix*x).((r, s)) # interpolate dg nodes to plotting nodes

  # construct a triangulation of the plotting nodes
  t = reference_plotting_triangulation((r_plot, s_plot))

  # extract x,y coordinates and solutions on each element
  uEltype = eltype(u)
  nvars = nvariables(equations)
  x, y = ntuple(_->zeros(real(dg), n_nodes_2d, n_elements), 2)
  u_extracted = StructArray{SVector{nvars, uEltype}}(ntuple(_->similar(x), nvars))
  for element in eachelement(dg, cache)
    sk = 1
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      xy_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      u_extracted[sk, element] = u_node
      x[sk, element] = xy_node[1]
      y[sk, element] = xy_node[2]
      sk += 1
    end
  end

  # interpolate to volume plotting points
  xplot, yplot = plotting_interp_matrix*x, plotting_interp_matrix*y
  uplot = StructArray{SVector{nvars, uEltype}}(map(x->plotting_interp_matrix*x,
                                                   StructArrays.components(u_extracted)))

  # extract indices of local face nodes for wireframe plotting
  tol = 100*eps()
  face_1 = findall(@. abs(s+1) < tol)
  face_2 = findall(@. abs(r-1) < tol)
  face_3 = findall(@. abs(s-1) < tol)
  face_4 = findall(@. abs(r+1) < tol)
  Fmask = hcat(face_1, face_2, face_3, face_4)
  plotting_interp_matrix1D = face_plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)

  # These 5 lines extract the face values on each element from the arrays x,y,sol_to_plot.
  # The resulting arrays are then reshaped so that xf, yf, sol_f are Matrix types of size
  # (Number of face plotting nodes) x (Number of faces).
  function face_first_reshape(x, num_nodes_1D, num_nodes, num_elements)
      num_reference_faces = 2 * ndims(mesh)
      xf = view(reshape(x, num_nodes, num_elements), vec(Fmask), :)
      return reshape(xf, num_nodes_1D, num_elements * num_reference_faces)
  end
  reshape_and_interpolate(x) = plotting_interp_matrix1D * face_first_reshape(x, nnodes(dg), n_nodes_2d, n_elements)
  xfp, yfp = map(reshape_and_interpolate, (x, y))
  ufp = StructArray{SVector{nvars, uEltype}}(map(reshape_and_interpolate, StructArrays.components(u_extracted)))

  # convert variables based on solution_variables mapping
  solution_variables_ = digest_solution_variables(equations, solution_variables)
  variable_names = SVector(varnames(solution_variables_, equations))

  transform_to_solution_variables!(uplot, solution_variables_, equations)
  transform_to_solution_variables!(ufp, solution_variables_, equations)

  return UnstructuredPlotData2D(xplot, yplot, uplot, t, xfp, yfp, ufp, variable_names)
end

# Given a reference plotting triangulation, this function generates a plotting triangulation for
# the entire global mesh. The output can be plotted using `Makie.mesh`.
function global_plotting_triangulation(pds::PlotDataSeries2D{<:UnstructuredPlotData2D};
                                         set_z_coordinate_zero = false)
  @unpack variable_id = pds
  pd = pds.plot_data
  @unpack x, y, u, t = pd

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
        coordinates[i, 3] = u[i, element][variable_id]
      end
    end
    trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates), makie_triangles)
  end
  plotting_mesh = merge([trimesh...]) # merge meshes on each element into one large mesh
  return plotting_mesh
end

# Returns a list of `Makie.Point`s which can be used to plot the mesh, or a solution "wireframe"
# (e.g., a plot of the mesh lines but with the z-coordinate equal to the value of the solution).
function mesh_plotting_wireframe(pds::PlotDataSeries2D{<:UnstructuredPlotData2D};
                                     set_z_coordinate_zero = false)
  @unpack variable_id = pds
  pd = pds.plot_data
  @unpack xf, yf, uf = pd

  if set_z_coordinate_zero
    sol_f = zeros(eltype(first(uf)), size(xf)) # plot 2d surface by setting z coordinate to zero
  else
    sol_f = StructArrays.component(uf, variable_id)
  end

  # This line separates solution lines on each edge by NaNs to ensure that they are rendered
  # separately. The coordinates `xf`, `yf` and the solution `sol_f`` are assumed to be a matrix
  # whose columns correspond to different elements. We add NaN separators by appending a row of
  # NaNs to this matrix. We also flatten (e.g., apply `vec` to) the result, as this speeds up
  # plotting.
  xyz_wireframe = Makie.Point.(map(x->vec(vcat(x, fill(NaN, 1, size(x, 2)))), (xf, yf, sol_f))...)

  return xyz_wireframe
end

"""
    iplot(u, mesh::UnstructuredMesh2D, equations, solver, cache;
          solution_variables=nothing, nvisnodes=nnodes(solver), variable_to_plot_in=1)

Creates an interactive surface plot of the solution and mesh for an `UnstructuredMesh2D` type.

Inputs:
- solution_variables: either `nothing` or a variable transformation function (e.g., `cons2prim`)
- nvisnodes: number of visualization nodes per dimension
"""
function iplot(u, mesh::UnstructuredMesh2D, equations, solver, cache;
               solution_variables=nothing, nvisnodes=2*nnodes(solver), variable_to_plot_in=1)

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
  plotting_mesh = Makie.@lift(global_plotting_triangulation(getindex(pd, variable_names[$(menu.selection)])))
  solution_z = Makie.@lift(getindex.($plotting_mesh.position, 3))

  # Plot the actual solution.
  Makie.mesh!(ax, plotting_mesh, color=solution_z, nvisnodes=nvisnodes)

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
  Makie.Colorbar(fig[1, 3], limits = Makie.@lift(extrema($solution_z)), colormap = :viridis, flipaxis = false)

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
  Makie.Theme(;)
end

function Makie.plot!(myplot::TrixiHeatmap)
  pds = myplot[:plot_data_series][]

  plotting_mesh = global_plotting_triangulation(pds; set_z_coordinate_zero = true)

  @unpack variable_id = pds
  pd = pds.plot_data
  solution_z = vec(StructArrays.component(pd.u, variable_id))
  Makie.mesh!(myplot, plotting_mesh, color=solution_z, shading=false)

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

# redirects Makie.plot(pd::PlotDataSeries2D) to custom recipe TrixiHeatmap(pd)
Makie.plottype(::Trixi.PlotDataSeries2D{<:Trixi.UnstructuredPlotData2D}) = TrixiHeatmap

# Makie does not yet support layouts in its plot recipes, so we overload `Makie.plot` directly.
function Makie.plot(sol::TrixiODESolution;
                    plot_mesh=true, solution_variables=nothing)
  return Makie.plot(PlotData2D(sol, solution_variables=solution_variables); plot_mesh=plot_mesh)
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
                    plot_mesh=true)
  figAxes = Makie.plot!(fig, pd; plot_mesh=plot_mesh)
  display(figAxes.fig)
  return figAxes
end

function Makie.plot!(fig, pd::UnstructuredPlotData2D;
                     plot_mesh = true)
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
    trixiheatmap!(ax, pds; plot_mesh=plot_mesh)
    ax.aspect = Makie.DataAspect() # equal aspect ratio
    ax.title  = variable_name
    Makie.xlims!(ax, extrema(pd.x))
    Makie.ylims!(ax, extrema(pd.y))
  end

  return FigureAndAxes(fig, axes)
end
