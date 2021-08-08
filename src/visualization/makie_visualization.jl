#     compute_triangle_area(tri)
#
# Computes the area of a triangle given `tri`, which is a tuple of three points (vectors),
# using the [Shoelace_formula](https://en.wikipedia.org/wiki/Shoelace_formula).
function compute_triangle_area(tri)
    A, B, C = tri
    return 0.5 * (A[1] * (B[2] - C[2]) + B[1] * (C[2]-A[2]) + C[1] * (A[2] - B[2]))
end

#   plotting_triangulation(reference_plotting_coordinates)
#
# Computes a triangulation of the points in `reference_plotting_coordinates`, which is a tuple containing
# vectors of plotting points on the reference element (e.g., reference_plotting_coordinates = (r,s)).
# The reference element is assumed to be [-1,1]^d.
#
# This function returns `t` which is a `3 x N_tri` Matrix{Int} containing indices of triangles in the
# triangulation of the plotting points, with zero-volume triangles removed.
#
# For example, r[t[1, i]] returns the first reference coordinate of the 1st point on the ith triangle.

function plotting_triangulation(reference_plotting_coordinates,tol=50*eps())

  # on-the-fly triangulation of plotting nodes on the reference element
  triin = Triangulate.TriangulateIO()
  triin.pointlist = permutedims(hcat(reference_plotting_coordinates...))
  triout,_ = Triangulate.triangulate("Q", triin)
  t = triout.trianglelist

  # filter out sliver triangles
  has_volume = fill(true, size(t,2))
  for i in axes(t, 2)
      ids = @view t[:, i]
      x_points = @view triout.pointlist[1, ids]
      y_points = @view triout.pointlist[2, ids]
      area = compute_triangle_area(zip(x_points, y_points))
      if abs(area) < tol
          has_volume[i] = false
      end
  end
  return t[:, findall(has_volume)]
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

function face_plotting_interpolation_matrix(dg::DGSEM; nvisnodes=nvisnodes)
  return polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1, 1, nvisnodes))
end

function plotting_interpolation_matrix(dg::DGSEM; nvisnodes = 2*length(dg.basis.nodes))
  Vp1D = polynomial_interpolation_matrix(dg.basis.nodes, LinRange(-1, 1, nvisnodes))
  return kron(Vp1D, Vp1D)
end

function reference_node_coordinates(dg::DGSEM)
  r1D = dg.basis.nodes
  num_nodes_1D = length(r1D)
  r = vec([r1D[i] for i in 1:num_nodes_1D, j in 1:num_nodes_1D])
  s = vec([r1D[j] for i in 1:num_nodes_1D, j in 1:num_nodes_1D])
  return r, s
end

function PlotData2D(u_input, mesh::UnstructuredMesh2D, equations, dg::DGSEM, cache;
                    solution_variables=nothing, nvisnodes=2*polydeg(dg))

  @assert ndims(mesh) == 2

  num_nodes_1D = length(dg.basis.nodes)
  num_nodes = num_nodes_1D^ndims(mesh)
  num_elements = nelements(dg, cache)

  # build nodes on reference element (seems to be the right ordering)
  r, s = reference_node_coordinates(dg)

  # reference plotting nodes
  Vplot = plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)

  # create triangulation for plotting nodes
  rp, sp = (x->Vplot*x).((r, s)) # interpolate dg nodes to plotting nodes

  # construct a triangulation of the plotting nodes
  t = permutedims(plotting_triangulation((rp, sp)))

  # extract x,y coordinates and solutions on each element
  uEltype = eltype(u_input)
  nvars = nvariables(equations)
  x, y = ntuple(_->zeros(real(dg), num_nodes, num_elements), 2)
  u = StructArray{SVector{nvars, uEltype}}(ntuple(_->similar(x), nvars))
  for element in eachelement(dg, cache)
    sk = 1
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u_input, equations, dg, i, j, element)
      xy_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      u[sk, element] = u_node
      x[sk, element] = xy_node[1]
      y[sk, element] = xy_node[2]
      sk += 1
    end
  end

  # interpolate to volume plotting points
  xplot, yplot = Vplot*x, Vplot*y
  uplot = StructArray{SVector{nvars, uEltype}}(map(x->Vplot*x, StructArrays.components(u)))

  # extract indices of local face nodes for wireframe plotting
  tol = 100*eps()
  face_1 = findall(@. abs(s+1) < tol)
  face_2 = findall(@. abs(r-1) < tol)
  face_3 = findall(@. abs(s-1) < tol)
  face_4 = findall(@. abs(r+1) < tol)
  Fmask = hcat(face_1, face_2, face_3, face_4)
  Vplot1D = face_plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)

  # These 5 lines extract the face values on each element from the arrays x,y,sol_to_plot.
  # The resulting arrays are then reshaped so that xf, yf, sol_f are Matrix types of size
  # (Number of face plotting nodes) x (Number of faces).
  function face_first_reshape(x, num_nodes_1D, num_nodes, num_elements)
      num_reference_faces = 2 * ndims(mesh)
      xf = view(reshape(x, num_nodes, num_elements), vec(Fmask), :)
      return reshape(xf, num_nodes_1D, num_elements * num_reference_faces)
  end
  xf, yf, uf = face_first_reshape.((x, y, u), num_nodes_1D, num_nodes, num_elements)
  xfp, yfp, ufp = map(xf->Vplot1D * xf, (xf, yf, uf))

  # convert variables based on solution_variables mapping
  solution_variables_ = digest_solution_variables(equations, solution_variables)

  for (i, u_i) in enumerate(uplot)
    uplot[i] = solution_variables_(u_i, equations)
  end

  for (i, u_i) in enumerate(ufp)
    ufp[i] = solution_variables_(u_i, equations)
  end

  variable_names = SVector(varnames(solution_variables_, equations))
  return UnstructuredPlotData2D(xplot, yplot, uplot, t, xfp, yfp, ufp, variable_names)
end

#   generate_plotting_triangulation(sol::TrixiODESolution, variable_to_plot::Int; nvisnodes=5)
#
# Generates a plotting triangulation which can be used by Makie's `mesh` function.
#
# Arguments:
# - sol: TrixiODESolution
# - variable_to_plot: index of solution field to plot
#
# Example:
# ```julia
# tri = generate_plotting_triangulation(sol,1)
# Makie.mesh(tri,color=getindex.(tri.position,3))
# ```
# The color is specified to be the z-value of the solution, which there does not seem to be a default solution for yet.

function generate_plotting_triangulation(sol::TrixiODESolution, variable_to_plot::Int;
                                         nvisnodes=10, compute_3d_surface = true)

  semi = sol.prob.p
  dg = semi.solver
  @unpack equations, cache, mesh = semi

  @assert ndims(mesh) == 2

  # make solution
  u = wrap_array(sol.u[end], mesh, equations, dg, cache)

  num_nodes_1D = length(dg.basis.nodes)
  num_nodes = num_nodes_1D^ndims(mesh)
  num_elements = nelements(dg, cache)

  # build nodes on reference element (seems to be the right ordering)
  r, s = reference_node_coordinates(dg)

  # reference plotting nodes
  Vp = plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)
  n_plot_nodes = size(Vp, 1)

  # create triangulation for plotting nodes
  rp, sp = (x->Vp*x).((r, s)) # interpolate dg nodes to plotting nodes

  # construct a triangulation of the plotting nodes
  t = permutedims(plotting_triangulation((rp, sp)))
  makie_triangles = Makie.to_triangles(t)

  # trimesh[i] holds the GeometryBasics.Mesh containing solution plotting
  # information on the ith element.
  trimesh = Vector{GeometryBasics.Mesh{3, Float32}}(undef, num_elements)
  coordinates_tmp = zeros(Float32, num_nodes, 3)
  solution_z_tmp = zeros(Float32, num_nodes)
  solution_z = zeros(Float32, n_plot_nodes, num_elements)
  coordinates = zeros(Float32, n_plot_nodes, 3)
  for element in eachelement(dg, cache)

    # extract x,y coordinates and solutions on each element
    sk = 1
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      xy_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      coordinates_tmp[sk, 1:2] .= xy_node
      if compute_3d_surface == true
        coordinates_tmp[sk, 3] = u_node[variable_to_plot]
      end
      solution_z_tmp[sk] = u_node[variable_to_plot]
      sk += 1
    end

    # interpolates both xy coordinates and solution values to plotting points
    mul!(coordinates, Vp, coordinates_tmp)
    mul!(view(solution_z, :, element), Vp, solution_z_tmp)

    trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates), makie_triangles)
  end

  # merge meshes on each element into one large mesh
  plotting_mesh = merge([trimesh...])
  return plotting_mesh, vec(solution_z)
end

function generate_plotting_wireframe(sol::TrixiODESolution, variable_to_plot::Int;
                                     nvisnodes=5, compute_3d_surface = true)

  semi = sol.prob.p
  @unpack equations, mesh, cache = semi
  dg = semi.solver

  @assert ndims(mesh) == 2

  num_nodes_1D = length(dg.basis.nodes)
  num_nodes = num_nodes_1D^ndims(mesh)
  num_elements = nelements(dg,cache)

  # reference interpolation operators
  Vp1D = face_plotting_interpolation_matrix(dg; nvisnodes=nvisnodes)

  # reconstruct reference nodes (assumes 2D ordering on reference quad is x_i,y_j with i first).
  r, s = reference_node_coordinates(dg)

  # extract indices of local face nodes
  tol = 50*eps()
  face_1 = findall(@. abs(s+1) < tol)
  face_2 = findall(@. abs(r-1) < tol)
  face_3 = findall(@. abs(s-1) < tol)
  face_4 = findall(@. abs(r+1) < tol)
  Fmask = hcat(face_1, face_2, face_3, face_4)

  # need to call `wrap_array` before using get_node_vars, etc...
  u = wrap_array(sol.u[end], mesh, equations, dg, cache)

  x, y, sol_to_plot = ntuple(_ -> zeros(Float32, num_nodes, num_elements), 3)
  for element in eachelement(dg, cache)
      sk = 1
      for j in eachnode(dg), i in eachnode(dg)
          u_node = get_node_vars(u, equations, dg, i, j, element)
          xy_node = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
          x[sk,element] = xy_node[1]
          y[sk,element] = xy_node[2]
          if compute_3d_surface == true
            sol_to_plot[sk, element] = u_node[variable_to_plot]
          end
          sk += 1
      end
  end

  # These 5 lines extract the face values on each element from the arrays x,y,sol_to_plot.
  # The resulting arrays are then reshaped so that xf, yf, sol_f are Matrix types of size
  # (Number of face plotting nodes) x (Number of faces).
  function face_first_reshape(x, num_nodes_1D, num_nodes, num_elements)
      num_reference_faces = 2 * ndims(mesh)
      xf = view(reshape(x, num_nodes, num_elements), vec(Fmask), :)
      return reshape(xf, num_nodes_1D, num_elements * num_reference_faces)
  end
  xf, yf, sol_f = face_first_reshape.((x, y, sol_to_plot), num_nodes_1D, num_nodes, num_elements)

  # this line does three things:
  # - interpolates face nodal points to plotting points via Vp1D * xf
  # - use NaNs to break up lines corresponding to each face
  # - converts the final array to a vector
  xfp, yfp, ufp = map(xf->vec(vcat(Vp1D * xf, fill(NaN, 1, size(xf, 2)))), (xf, yf, sol_f))

  return Makie.Point.(xfp, yfp, ufp)
end

function trixi_plot(sol::TrixiODESolution; variable_to_plot_in = 1, nvisnodes=5)

  semi = sol.prob.p
  dg = semi.solver
  @unpack equations, cache, mesh = semi

  @assert ndims(mesh) == 2

  fig = Makie.Figure()

  # set up menu and toggle switch
  variable_names = varnames(cons2cons, equations)
  options = [zip(variable_names, 1:length(variable_names))...]
  menu = Makie.Menu(fig, options = options)
  toggle_solution_mesh = Makie.Toggle(fig, active=true)
  toggle_mesh = Makie.Toggle(fig, active=true)
  fig[1, 1] = Makie.vgrid!(
      Makie.Label(fig, "Solution field", width = nothing), menu,
      Makie.Label(fig, "Solution mesh visible"), toggle_solution_mesh,
      Makie.Label(fig, "Mesh visible"), toggle_mesh;
      tellheight=false, width = 200
  )

  ax = Makie.LScene(fig[1, 2], scenekw = (show_axis = false,))

  # interactive menu variable_to_plot
  menu.selection[] = variable_to_plot_in
  menu.i_selected[] = variable_to_plot_in # initialize a menu

  # these lines get re-run whenever variable_to_plot[] is updated
  plotting_mesh = Makie.@lift(generate_plotting_triangulation(sol, $(menu.selection), nvisnodes=nvisnodes)[1])
  solution_z = Makie.@lift(getindex.($plotting_mesh.position, 3))
  Makie.mesh!(ax, plotting_mesh, color=solution_z, nvisnodes=nvisnodes)

  # mesh overlay: we plot a mesh both on top of and below the solution contours
  wire_points = Makie.@lift(generate_plotting_wireframe(sol, $(menu.selection), nvisnodes=nvisnodes))
  wire_mesh_top = Makie.lines!(ax, wire_points, color=:white)
  wire_mesh_bottom = Makie.lines!(ax, wire_points, color=:white)
  Makie.translate!(wire_mesh_top, 0, 0, 1e-3)
  Makie.translate!(wire_mesh_bottom, 0, 0, -1e-3)

  # draw a flattened mesh wireframe below the solution
  function compute_z_offset(solution_z)
      zmin = minimum(solution_z)
      zrange = (x->x[2]-x[1])(extrema(solution_z))
      return zmin - .25*zrange
  end
  z_offset = Makie.@lift(compute_z_offset($solution_z))
  get_flat_points(wire_points, z_offset) = [Makie.Point(point.data[1:2]..., z_offset) for point in wire_points]
  flat_wire_points = Makie.@lift get_flat_points($wire_points, $z_offset)
  wire_mesh_flat = Makie.lines!(ax, flat_wire_points, color=:black)

  # reset colorbar each time solution changes
  Makie.Colorbar(fig[1, 3], limits = Makie.@lift(extrema($solution_z)), colormap = :viridis, flipaxis = false)

  # syncs the toggle to the mesh
  Makie.connect!(wire_mesh_top.visible, toggle_solution_mesh.active)
  Makie.connect!(wire_mesh_bottom.visible, toggle_solution_mesh.active)
  Makie.connect!(wire_mesh_flat.visible, toggle_mesh.active)

  # maybe trigger this if using OSX? Otherwise shift-command-4 for screenshots triggers an "up-zoom"
  # cameracontrols(ax.scene).attributes[:up_key][] = Makie.Keyboard.right_shift

  # typing this pulls up the figure (similar to display(plot!()) in Plots.jl)
  fig
end

function trixi_heatmap(sol::TrixiODESolution; variable_to_plot = 1, nvisnodes=5)

  # Create layout that is as square as possible, when there are more than 3 subplots.
  # This is done with a preference for more columns than rows if not.
  pd = ones(nvariables(sol.prob.p.equations)) # TODO: replace this with PlotData2D
  if length(pd) <= 3
    cols = length(pd)
    rows = 1
  else
    cols = ceil(Int, sqrt(length(pd)))
    rows = ceil(Int, length(pd)/cols)
  end

  fig = Makie.Figure()
  axes = [Makie.Axis(fig[i,j]) for j in 1:rows, i in 1:cols]

  for (variable_to_plot, ax) in enumerate(axes)

    plotting_mesh, solution_z = generate_plotting_triangulation(sol, variable_to_plot;
                                                                nvisnodes=nvisnodes,
                                                                compute_3d_surface = false)
    x_coordinates = getindex.(plotting_mesh.position, 1)
    y_coordinates = getindex.(plotting_mesh.position, 2)

    xyz_wireframe = generate_plotting_wireframe(sol, variable_to_plot; nvisnodes=nvisnodes,
                                                compute_3d_surface=false)

    Makie.mesh!(ax, plotting_mesh, color=solution_z, shading=false)
    Makie.xlims!(ax, extrema(x_coordinates))
    Makie.ylims!(ax, extrema(y_coordinates))
    ax.aspect = Makie.DataAspect() # equal aspect ratio
    ax.title = "variable $variable_to_plot"

    Makie.lines!(ax, xyz_wireframe, color=:lightgrey)
  end
  fig
end

# ================== new Makie plot recipes ====================

@Makie.recipe(TrixiHeatmap, plot_data_series) do scene
  Makie.Theme(;)
end

function Makie.plot!(myplot::TrixiHeatmap)
  pds = myplot[:plot_data_series][]
  @unpack variable_id = pds
  pd = pds.plot_data
  @unpack x, y, u, t = pd

  makie_triangles = Makie.to_triangles(t)

  # trimesh[i] holds GeometryBasics.Mesh containing plotting information on the ith element.
  num_plotting_nodes, num_elements = size(x)
  trimesh = Vector{GeometryBasics.Mesh{3, Float32}}(undef, num_elements)
  coordinates = zeros(Float32, num_plotting_nodes, 3)
  for element in Base.OneTo(num_elements)
    for i in Base.OneTo(num_plotting_nodes)
      coordinates[i, 1] = x[i, element]
      coordinates[i, 2] = y[i, element]
    end
    trimesh[element] = GeometryBasics.normal_mesh(Makie.to_vertices(coordinates), makie_triangles)
  end
  plotting_mesh = merge([trimesh...]) # merge meshes on each element into one large mesh
  solution_z = vec(StructArrays.component(u, variable_id))

  Makie.mesh!(myplot, plotting_mesh, color=solution_z, shading=false)
  # TODO: visualization. Figure out how to use axes with plot recipes
  # Makie.xlims!(ax, extrema(x))
  # Makie.ylims!(ax, extrema(y))
  # ax.aspect = Makie.DataAspect() # equal aspect ratio
  # ax.title = variable_names[variable_id]

  # TODO: visualization. Decide how to trigger grid plotting (maybe kwargs?)
  @unpack xf, yf = pd
  sol_f = zeros(size(xf)) # plot 2d surface by setting z coordinate to zero
  xyz_wireframe = Makie.Point.(map(x->vec(vcat(x, fill(NaN, 1, size(x, 2)))), (xf, yf, sol_f))...)
  Makie.lines!(myplot, xyz_wireframe, color=:lightgrey)

  myplot
end

# redirects Makie.plot(pd::PlotDataSeries2D) to TrixiHeatmap(pd)
Makie.plottype(::Trixi.PlotDataSeries2D{<:Trixi.UnstructuredPlotData2D}) = TrixiHeatmap

function Makie.plot(pd::UnstructuredPlotData2D)
  # Create layout that is as square as possible, when there are more than 3 subplots.
  # This is done with a preference for more columns than rows if not.
  if length(pd) <= 3
    cols = length(pd)
    rows = 1
  else
    cols = ceil(Int, sqrt(length(pd)))
    rows = ceil(Int, length(pd)/cols)
  end

  fig = Makie.Figure()
  axes = [Makie.Axis(fig[i,j]) for j in 1:rows, i in 1:cols]

  for (variable_to_plot, (variable_name, pds)) in enumerate(pd)
    ax = axes[variable_to_plot]
    trixiheatmap!(ax, pds)
    ax.aspect = Makie.DataAspect() # equal aspect ratio
    ax.title  = variable_name
  end

  fig
end