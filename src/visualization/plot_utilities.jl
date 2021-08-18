@inline num_faces(elem::Tri) = 3
@inline num_faces(elem::Quad) = 4

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

# This function is used to avoid type instabilities when calling `digest_solution_variables`.
function transform_to_solution_variables!(u, solution_variables, equations)
  for (i, u_i) in enumerate(u)
    u[i] = solution_variables(u_i, equations)
  end
end

#     global_plotting_triangulation_triplot(u_plot, rst_plot, xyz_plot)
#
# Returns (plotting_coordinates_x, plotting_coordinates_y, ..., plotting_values, plotting_triangulation).
# Output can be used with TriplotRecipes.DGTriPseudocolor(...).
#
# Inputs:
#   - xyz_plot = plotting points (tuple of matrices of size (Nplot, K))
#   - u_plot = matrix of size (Nplot, K) representing solution to plot.
#   - t = triangulation of reference plotting points
function global_plotting_triangulation_triplot(xyz_plot, u_plot, t)

  @assert size(first(xyz_plot), 1) == size(u_plot, 1) "Row dimension of u_plot does not match row dimension of xyz_plot"

  # build discontinuous data on plotting triangular mesh
  num_plotting_points, num_elements = size(u_plot)
  num_reference_plotting_triangles = size(t, 1)
  num_plotting_elements_total = num_reference_plotting_triangles * num_elements

  # each column of `tp` corresponds to a vertex of a plotting triangle
  tp = zeros(Int32, 3, num_plotting_elements_total)
  zp = similar(tp, eltype(u_plot))
  for e = 1:num_elements
    for i = 1:num_reference_plotting_triangles
      tp[:, i + (e-1)*num_reference_plotting_triangles] .= @views t[i, :] .+ (e-1) * num_plotting_points
      zp[:, i + (e-1)*num_reference_plotting_triangles] .= @views u_plot[t[i, :], e]
    end
  end
  return vec.(xyz_plot)..., zp, tp
end

#     mesh_plotting_wireframe(rd::RefElemData{2}, md::MeshData{2})
#
# Generates data for plotting a mesh wireframe given StartUpDG data types.
# Returns (plotting_coordinates_x, plotting_coordinates_y) for a 2D mesh wireframe.
function mesh_plotting_wireframe(rd::RefElemData{2}, md::MeshData{2}; num_plotting_points=25)

  # Construct 1D plotting interpolation matrix `Vp1D` for a single face
  @unpack N, Fmask = rd
  vandermonde_matrix_1D = StartUpDG.vandermonde(Line(), N, StartUpDG.nodes(Line(), N))
  rplot = LinRange(-1, 1, num_plotting_points)
  Vp1D = StartUpDG.vandermonde(Line(), N, rplot) / vandermonde_matrix_1D

  num_face_points = N+1
  num_faces_total = num_faces(rd.elementType) * md.num_elements
  xf, yf = map(x->reshape(view(x, Fmask, :), num_face_points, num_faces_total), md.xyz)

  num_face_plotting_points = size(Vp1D, 1)
  x_mesh, y_mesh = ntuple(_->zeros(num_face_plotting_points, num_faces_total), 2)
  for f in 1:num_faces_total
    mul!(view(x_mesh, :, f), Vp1D, view(xf, :, f))
    mul!(view(y_mesh, :, f), Vp1D, view(yf, :, f))
  end

  return x_mesh, y_mesh
end


# These methods are used internally to set the default value of the solution variables:
# - If a `cons2prim` for the given `equations` exists, use it
# - Otherwise, use `cons2cons`, which is defined for all systems of equations
digest_solution_variables(equations, solution_variables) = solution_variables
function digest_solution_variables(equations, solution_variables::Nothing)
  if hasmethod(cons2prim, Tuple{AbstractVector, typeof(equations)})
    return cons2prim
  else
    return cons2cons
  end
end


"""
    adapt_to_mesh_level!(u_ode, semi, level)
    adapt_to_mesh_level!(sol::Trixi.TrixiODESolution, level)

Like [`adapt_to_mesh_level`](@ref), but modifies the solution and parts of the
semidiscretization (mesh and caches) in place.
"""
function adapt_to_mesh_level!(u_ode, semi, level)
  # Create AMR callback with controller that refines everything towards a single level
  amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first), base_level=level)
  amr_callback = AMRCallback(semi, amr_controller, interval=0)

  # Adapt mesh until it does not change anymore
  has_changed = amr_callback.affect!(u_ode, semi, 0.0, 0)
  while has_changed
    has_changed = amr_callback.affect!(u_ode, semi, 0.0, 0)
  end

  return u_ode, semi
end

adapt_to_mesh_level!(sol::TrixiODESolution, level) = adapt_to_mesh_level!(sol.u[end], sol.prob.p, level)


"""
    adapt_to_mesh_level(u_ode, semi, level)
    adapt_to_mesh_level(sol::Trixi.TrixiODESolution, level)

Use the regular adaptive mesh refinement routines to adaptively refine/coarsen the solution `u_ode`
with semidiscretization `semi` towards a uniformly refined grid with refinement level `level`. The
solution and semidiscretization are copied such that the original objects remain *unaltered*.

A convenience method accepts an ODE solution object, from which solution and semidiscretization are
extracted as needed.

See also: [`adapt_to_mesh_level!`](@ref)
"""
function adapt_to_mesh_level(u_ode, semi, level)
  # Create new semidiscretization with copy of the current mesh
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  new_semi = remake(semi, mesh=deepcopy(mesh))

  return adapt_to_mesh_level!(deepcopy(u_ode), new_semi, level)
end

adapt_to_mesh_level(sol::TrixiODESolution, level) = adapt_to_mesh_level(sol.u[end], sol.prob.p, level)


# Extract data from a 2D/3D DG solution and prepare it for visualization as a heatmap/contour plot.
#
# Returns a tuple with
# - x coordinates
# - y coordinates
# - nodal 2D data
# - x vertices for mesh visualization
# - y vertices for mesh visualization
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function get_data_2d(center_level_0, length_level_0, leaf_cells, coordinates, levels, ndims,
                     unstructured_data, n_nodes,
                     grid_lines=false, max_supported_level=11, nvisnodes=nothing,
                     slice=:xy, point=(0.0, 0.0, 0.0))
  # Determine resolution for data interpolation
  max_level = maximum(levels)
  if max_level > max_supported_level
    error("Maximum refinement level $max_level is higher than " *
          "maximum supported level $max_supported_level")
  end
  max_available_nodes_per_finest_element = 2^(max_supported_level - max_level)
  if nvisnodes === nothing
    max_nvisnodes = 2 * n_nodes
  elseif nvisnodes == 0
    max_nvisnodes = n_nodes
  else
    max_nvisnodes = nvisnodes
  end
  nvisnodes_at_max_level = min(max_available_nodes_per_finest_element, max_nvisnodes)
  resolution = nvisnodes_at_max_level * 2^max_level
  nvisnodes_per_level = [2^(max_level - level)*nvisnodes_at_max_level for level in 0:max_level]
  # nvisnodes_per_level is an array (accessed by "level + 1" to accommodate
  # level-0-cell) that contains the number of visualization nodes for any
  # refinement level to visualize on an equidistant grid

  if ndims == 3
    (unstructured_data, coordinates, levels,
        center_level_0) = unstructured_3d_to_2d(unstructured_data,
        coordinates, levels, length_level_0, center_level_0, slice,
        point)
  end

  # Normalize element coordinates: move center to (0, 0) and domain size to [-1, 1]²
  n_elements = length(levels)
  normalized_coordinates = similar(coordinates)
  for element_id in 1:n_elements
    @views normalized_coordinates[:, element_id] .= (
          (coordinates[:, element_id] .- center_level_0) ./ (length_level_0 / 2 ))
  end

  # Interpolate unstructured DG data to structured data
  (structured_data =
      unstructured2structured(unstructured_data, normalized_coordinates,
                              levels, resolution, nvisnodes_per_level))

  # Interpolate cell-centered values to node-centered values
  node_centered_data = cell2node(structured_data)

  # Determine axis coordinates for contour plot
  xs = collect(range(-1, 1, length=resolution+1)) .* length_level_0/2 .+ center_level_0[1]
  ys = collect(range(-1, 1, length=resolution+1)) .* length_level_0/2 .+ center_level_0[2]

  # Determine element vertices to plot grid lines
  if grid_lines
    mesh_vertices_x, mesh_vertices_y = calc_vertices(coordinates, levels, length_level_0)
  else
    mesh_vertices_x = Vector{Float64}(undef, 0)
    mesh_vertices_y = Vector{Float64}(undef, 0)
  end

  return xs, ys, node_centered_data, mesh_vertices_x, mesh_vertices_y
end


# Extract data from a 1D DG solution and prepare it for visualization as a line plot.
# This returns a tuple with
# - x coordinates
# - nodal 1D data
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function get_data_1d(original_nodes, unstructured_data, nvisnodes)
  # Get the dimensions of u; where n_vars is the number of variables, n_nodes the number of nodal values per element and n_elements the total number of elements.
  n_nodes, n_elements, n_vars = size(unstructured_data)

  # Set the amount of nodes visualized according to nvisnodes.
  if nvisnodes === nothing
    max_nvisnodes = 2 * n_nodes
  elseif nvisnodes == 0
    max_nvisnodes = n_nodes
  else
    @assert nvisnodes >= 2 "nvisnodes must be zero or >= 2"
    max_nvisnodes = nvisnodes
  end

  interpolated_nodes = Array{eltype(original_nodes),    2}(undef, max_nvisnodes, n_elements)
  interpolated_data  = Array{eltype(unstructured_data), 3}(undef, max_nvisnodes, n_elements, n_vars)

  for j in 1:n_elements
    # Interpolate on an equidistant grid.
    interpolated_nodes[:, j] .= range(original_nodes[1,1,j], original_nodes[1,end,j], length = max_nvisnodes)
  end

  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes)
  nodes_out = collect(range(-1, 1, length = max_nvisnodes))

  # Calculate vandermonde matrix for interpolation.
  vandermonde = polynomial_interpolation_matrix(nodes_in, nodes_out)

  # Iterate over all variables.
  for v in 1:n_vars
    # Interpolate data for each element.
    for element in 1:n_elements
      multiply_scalar_dimensionwise!(@view(interpolated_data[:, element, v]),
        vandermonde, @view(unstructured_data[:, element, v]))
    end
  end
  # Return results after data is reshaped
  return vec(interpolated_nodes), reshape(interpolated_data, :, n_vars), vcat(original_nodes[1, 1, :], original_nodes[1, end, end])
end

# Change order of dimensions (variables are now last) and convert data to `solution_variables`
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function get_unstructured_data(u, solution_variables, mesh, equations, solver, cache)

  if solution_variables === cons2cons
    raw_data = u
    n_vars = size(raw_data, 1)
  else
    # FIXME: Remove this comment once the implementation following it has been verified
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    # raw_data = Array(reinterpret(eltype(u),
    #        solution_variables.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
    #                   Ref(equations))))
    # n_vars = size(raw_data, 1)
    n_vars_in = nvariables(equations)
    n_vars = length(solution_variables(get_node_vars(u, equations, solver), equations))
    raw_data = Array{eltype(u)}(undef, n_vars, Base.tail(size(u))...)
    reshaped_u = reshape(u, n_vars_in, :)
    reshaped_r = reshape(raw_data, n_vars, :)
    for idx in axes(reshaped_u, 2)
      reshaped_r[:, idx] = solution_variables(get_node_vars(reshaped_u, equations, solver, idx), equations)
    end
  end

  unstructured_data = Array{eltype(raw_data)}(undef,
                                              ntuple((d) -> nnodes(solver), ndims(equations))...,
                                              nelements(solver, cache), n_vars)
  for variable in 1:n_vars
    @views unstructured_data[.., :, variable] .= raw_data[variable, .., :]
  end

  return unstructured_data
end



# Convert cell-centered values to node-centered values by averaging over all
# four neighbors and making use of the periodicity of the solution
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function cell2node(cell_centered_data)
  # Create temporary data structure to make the averaging algorithm as simple
  # as possible (by using a ghost layer)
  tmp = similar(first(cell_centered_data), size(first(cell_centered_data)) .+ (2, 2))

  # Create output data structure
  resolution_in, _ = size(first(cell_centered_data))
  resolution_out = resolution_in + 1
  node_centered_data = [Matrix{Float64}(undef, resolution_out, resolution_out)
                        for _ in 1:length(cell_centered_data)]


  for (cell_data, node_data) in zip(cell_centered_data, node_centered_data)
    # Fill center with original data
    tmp[2:end-1, 2:end-1] .= cell_data

    # Fill sides with opposite data (periodic domain)
    # x-direction
    tmp[1,   2:end-1] .= cell_data[end, :]
    tmp[end, 2:end-1] .= cell_data[1,   :]
    # y-direction
    tmp[2:end-1, 1, ] .= cell_data[:, end]
    tmp[2:end-1, end] .= cell_data[:, 1, ]
    # Corners
    tmp[1,   1, ] = cell_data[end, end]
    tmp[end, 1, ] = cell_data[1,   end]
    tmp[1,   end] = cell_data[end, 1, ]
    tmp[end, end] = cell_data[1,   1, ]

    # Obtain node-centered value by averaging over neighboring cell-centered values
    for j in 1:resolution_out
      for i in 1:resolution_out
        node_data[i, j] = (tmp[i,   j, ] +
                           tmp[i+1, j, ] +
                           tmp[i,   j+1] +
                           tmp[i+1, j+1]) / 4
      end
    end
  end

  # Transpose
  for (index, data) in enumerate(node_centered_data)
    node_centered_data[index] = permutedims(data)
  end

  return node_centered_data
end


# Convert 3d unstructured data to 2d data.
# Additional to the new unstructured data updated coordinates, levels and
# center coordinates are returned.
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function unstructured_3d_to_2d(unstructured_data, coordinates, levels,
                               length_level_0, center_level_0, slice,
                               point)
  if slice === :yz
    slice_dimension = 1
    other_dimensions = [2, 3]
  elseif slice === :xz
    slice_dimension = 2
    other_dimensions = [1, 3]
  elseif slice === :xy
    slice_dimension = 3
    other_dimensions = [1, 2]
  else
    error("illegal dimension '$slice', supported dimensions are :yz, :xz, and :xy")
  end

  # Limits of domain in slice dimension
  lower_limit = center_level_0[slice_dimension] - length_level_0 / 2
  upper_limit = center_level_0[slice_dimension] + length_level_0 / 2

  @assert length(point) >= 3 "Point must be three-dimensional."
  if point[slice_dimension] < lower_limit || point[slice_dimension] > upper_limit
    error(string("Slice plane is outside of domain.",
        " point[$slice_dimension]=$(point[slice_dimension]) must be between $lower_limit and $upper_limit"))
  end

  # Extract data shape information
  n_nodes_in, _, _, n_elements, n_variables = size(unstructured_data)

  # Get node coordinates for DG locations on reference element
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)

  # New unstructured data has one dimension less.
  # The redundant element ids are removed later.
  @views new_unstructured_data = similar(unstructured_data[1, ..])

  # Declare new empty arrays to fill in new coordinates and levels
  new_coordinates = Array{Float64}(undef, 2, n_elements)
  new_levels = Array{eltype(levels)}(undef, n_elements)

  # Counter for new element ids
  new_id = 0

  # Save vandermonde matrices in a Dict to prevent redundant generation
  vandermonde_to_2d = Dict()

  # Permute dimensions such that the slice dimension is always the
  # third dimension of the array. Below we can always interpolate in the
  # third dimension.
  if slice === :yz
    unstructured_data = permutedims(unstructured_data, [2, 3, 1, 4, 5])
  elseif slice === :xz
    unstructured_data = permutedims(unstructured_data, [1, 3, 2, 4, 5])
  end

  for element_id in 1:n_elements
    # Distance from center to border of this element (half the length)
    element_length = length_level_0 / 2^levels[element_id]
    min_coordinate = coordinates[:, element_id] .- element_length / 2
    max_coordinate = coordinates[:, element_id] .+ element_length / 2

    # Check if slice plane and current element intersect.
    # The first check uses a "greater but not equal" to only match one cell if the
    # slice plane lies between two cells.
    # The second check is needed if the slice plane is at the upper border of
    # the domain due to this.
    if !((min_coordinate[slice_dimension] <= point[slice_dimension] &&
          max_coordinate[slice_dimension] > point[slice_dimension]) ||
        (point[slice_dimension] == upper_limit &&
          max_coordinate[slice_dimension] == upper_limit))
      # Continue for loop if they don't intersect
      continue
    end

    # This element is of interest
    new_id += 1

    # Add element to new coordinates and levels
    new_coordinates[:, new_id] = coordinates[other_dimensions, element_id]
    new_levels[new_id] = levels[element_id]

    # Construct vandermonde matrix (or load from Dict if possible)
    normalized_intercept =
        (point[slice_dimension] - min_coordinate[slice_dimension]) /
        element_length * 2 - 1

    if haskey(vandermonde_to_2d, normalized_intercept)
      vandermonde = vandermonde_to_2d[normalized_intercept]
    else
      # Generate vandermonde matrix to interpolate values at nodes_in to one value
      vandermonde = polynomial_interpolation_matrix(nodes_in, [normalized_intercept])
      vandermonde_to_2d[normalized_intercept] = vandermonde
    end

    # 1D interpolation to specified slice plane
    # We permuted the dimensions above such that now the dimension in which
    # we will interpolate is always the third one.
    for i in 1:n_nodes_in
      for ii in 1:n_nodes_in
        # Interpolate in the third dimension
        data = unstructured_data[i, ii, :, element_id, :]

        value = multiply_dimensionwise(vandermonde, permutedims(data))
        new_unstructured_data[i, ii, new_id, :] = value[:, 1]
      end
    end
  end

  # Remove redundant element ids
  unstructured_data = new_unstructured_data[:, :, 1:new_id, :]
  new_coordinates = new_coordinates[:, 1:new_id]
  new_levels = new_levels[1:new_id]

  center_level_0 = center_level_0[other_dimensions]

  return unstructured_data, new_coordinates, new_levels, center_level_0
end

# Convert 2d unstructured data to 1d slice and interpolate them.
function unstructured_2d_to_1d(original_nodes, unstructured_data, nvisnodes, slice, point)

  if slice === :x
    slice_dimension = 2
    other_dimension = 1
  elseif slice === :y
    slice_dimension = 1
    other_dimension = 2
  else
    error("illegal dimension '$slice', supported dimensions are :x and :y")
  end

  # Set up data structures to stroe new 1D data.
  @views new_unstructured_data = similar(unstructured_data[1, ..])
  @views new_nodes = similar(original_nodes[1, 1, ..])

  n_nodes_in, _, n_elements, n_variables = size(unstructured_data)
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)

  # Test if point lies in the domain.
  lower_limit = original_nodes[1, 1, 1, 1]
  upper_limit = original_nodes[1, n_nodes_in, n_nodes_in, n_elements]

  @assert length(point) >= 2 "Point must be two-dimensional."
  if point[slice_dimension] < lower_limit || point[slice_dimension] > upper_limit
    error(string("Slice axis is outside of domain. ",
        " point[$slice_dimension]=$(point[slice_dimension]) must be between $lower_limit and $upper_limit"))
  end

  # Count the amount of new elements.
  new_id = 0

  # Permute dimensions so that the slice dimension is always in the correct place for later use.
  if slice === :y
    original_nodes = permutedims(original_nodes, [1, 3, 2, 4])
    unstructured_data = permutedims(unstructured_data, [2, 1, 3, 4])
  end

  # Iterate over all elements to find the ones that lie on the slice axis.
  for element_id in 1:n_elements
    min_coordinate = original_nodes[:, 1, 1, element_id]
    max_coordinate = original_nodes[:, n_nodes_in, n_nodes_in, element_id]
    element_length = max_coordinate - min_coordinate

    # Test if the element is on the slice axis. If not just continue with the next element.
    if !((min_coordinate[slice_dimension] <= point[slice_dimension] &&
        max_coordinate[slice_dimension] > point[slice_dimension]) ||
        (point[slice_dimension] == upper_limit && max_coordinate[slice_dimension] == upper_limit))

        continue
    end

    new_id += 1

    # Construct vandermonde matrix for interpolation of each 2D element to a 1D element.
    normalized_intercept =
          (point[slice_dimension] - min_coordinate[slice_dimension]) /
          element_length[1] * 2 - 1
    vandermonde = polynomial_interpolation_matrix(nodes_in, normalized_intercept)

    # Interpolate to each node of new 1D element.
    for v in 1:n_variables
      for node in 1:n_nodes_in
        new_unstructured_data[node, new_id, v] = (vandermonde*unstructured_data[node, :, element_id, v])[1]
      end
    end

    new_nodes[:, new_id] = original_nodes[other_dimension, :, 1, element_id]
  end

  return get_data_1d(reshape(new_nodes[:, 1:new_id], 1, n_nodes_in, new_id), new_unstructured_data[:, 1:new_id, :], nvisnodes)
end

# Calculate the arc length of a curve given by ndims x npoints point coordinates (piece-wise linear approximation)
function calc_arc_length(coordinates)
  n_points = size(coordinates)[2]
  arc_length = zeros(n_points)
  for i in 1:n_points-1
    arc_length[i+1] = arc_length[i] + sqrt(sum((coordinates[:,i]-coordinates[:,i+1]).^2))
  end
  return arc_length
end

# Convert 2d unstructured data to 1d data at given curve.
function unstructured_2d_to_1d_curve(original_nodes, unstructured_data, nvisnodes, curve, mesh, solver, cache)

  n_points_curve = size(curve)[2]
  n_nodes, _, n_elements, n_variables = size(unstructured_data)
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes)

  # Check if input is correct.
  min = original_nodes[:, 1, 1, 1]
  max = max_coordinate = original_nodes[:, n_nodes, n_nodes, n_elements]
  @assert size(curve) == (2, size(curve)[2]) "Coordinates along curve must be 2xn dimensional."
  for element in 1:n_points_curve
    @assert (prod(vcat(curve[:, n_points_curve] .>= min, curve[:, n_points_curve]
            .<= max))) "Some coordinates from `curve` are outside of the domain.."
  end

  # Set nodes acording to the length of the curve.
  arc_length = calc_arc_length(curve)

  # Setup data structures.
  data_on_curve = Array{Float64}(undef, n_points_curve, n_variables)
  temp_data = Array{Float64}(undef, n_nodes, n_points_curve, n_variables)

  # For each coordinate find the corresponding element with its id.
  element_ids = get_elements_by_coordinates(curve, mesh, solver, cache)

  # Iterate over all found elements.
  for element in 1:n_points_curve

    min_coordinate = original_nodes[:, 1, 1, element_ids[element]]
    max_coordinate = original_nodes[:, n_nodes, n_nodes, element_ids[element]]
    element_length = max_coordinate - min_coordinate

    normalized_coordinates = (curve[:, element] - min_coordinate)/element_length[1]*2 .-1

    # Interpolate to a single point in each element.
    vandermonde_x = polynomial_interpolation_matrix(nodes_in, normalized_coordinates[1])
    vandermonde_y = polynomial_interpolation_matrix(nodes_in, normalized_coordinates[2])
    for v in 1:n_variables
      for i in 1:n_nodes
        temp_data[i, element, v] = (vandermonde_y*unstructured_data[i, :, element_ids[element], v])[1]
      end
      data_on_curve[element, v] = (vandermonde_x*temp_data[:, element, v])[]
    end
  end

  return arc_length, data_on_curve, nothing
end

# Convert 3d unstructured data to 1d data at given curve.
function unstructured_3d_to_1d_curve(original_nodes, unstructured_data, nvisnodes, curve, mesh, solver, cache)

  n_points_curve = size(curve)[2]
  n_nodes, _, _, n_elements, n_variables = size(unstructured_data)
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes)

  # Check if input is correct.
  min = original_nodes[:, 1, 1, 1, 1]
  max = max_coordinate = original_nodes[:, n_nodes, n_nodes, n_nodes, n_elements]
  @assert size(curve) == (3, n_points_curve) "Coordinates along curve must be 3xn dimensional."
  for element in 1:n_points_curve
    @assert (prod(vcat(curve[:, n_points_curve] .>= min, curve[:, n_points_curve]
            .<= max))) "Some coordinates from `curve` are outside of the domain.."
  end

  # Set nodes acording to the length of the curve.
  arc_length = calc_arc_length(curve)

  # Setup data structures.
  data_on_curve = Array{Float64}(undef, n_points_curve, n_variables)
  temp_data = Array{Float64}(undef, n_nodes, n_nodes+1, n_points_curve, n_variables)

  # For each coordinate find the corresponding element with its id.
  element_ids = get_elements_by_coordinates(curve, mesh, solver, cache)

  # Iterate over all found elements.
  for element in 1:n_points_curve

    min_coordinate = original_nodes[:, 1, 1, 1, element_ids[element]]
    max_coordinate = original_nodes[:, n_nodes, n_nodes, n_nodes, element_ids[element]]
    element_length = max_coordinate - min_coordinate

    normalized_coordinates = (curve[:, element] - min_coordinate)/element_length[1]*2 .-1

    # Interpolate to a single point in each element.
    vandermonde_x = polynomial_interpolation_matrix(nodes_in, normalized_coordinates[1])
    vandermonde_y = polynomial_interpolation_matrix(nodes_in, normalized_coordinates[2])
    vandermonde_z = polynomial_interpolation_matrix(nodes_in, normalized_coordinates[3])
    for v in 1:n_variables
      for i in 1:n_nodes
        for ii in 1:n_nodes
          temp_data[i, ii, element, v] = (vandermonde_z*unstructured_data[i, ii, :, element_ids[element], v])[1]
        end
        temp_data[i, n_nodes+1, element, v] = (vandermonde_y*temp_data[i, 1:n_nodes, element, v])[1]
      end
      data_on_curve[element, v] = (vandermonde_x*temp_data[:, n_nodes+1, element, v])[1]
    end
  end

  return arc_length, data_on_curve, nothing
end

# Convert 3d unstructured data to 1d slice and interpolate them.
function unstructured_3d_to_1d(original_nodes, unstructured_data, nvisnodes, slice, point)

  if slice === :x
    slice_dimension = 1
    other_dimensions = [2,3]
  elseif slice === :y
    slice_dimension = 2
    other_dimensions = [1,3]
  elseif slice === :z
    slice_dimension = 3
    other_dimensions = [1,2]
  else
    error("illegal dimension '$slice', supported dimensions are :x, :y and :z")
  end

  # Set up data structures to stroe new 1D data.
  @views new_unstructured_data = similar(unstructured_data[1, 1, ..])
  @views temp_unstructured_data = similar(unstructured_data[1, ..])
  @views new_nodes = similar(original_nodes[1, 1, 1,..])

  n_nodes_in, _, _, n_elements, n_variables = size(unstructured_data)
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)

  # Test if point lies in the domain.
  lower_limit = original_nodes[1, 1, 1, 1, 1]
  upper_limit = original_nodes[1, n_nodes_in, n_nodes_in, n_nodes_in, n_elements]

  @assert length(point) >= 3 "Point must be three-dimensional."
  if prod(point[other_dimensions] .< lower_limit) || prod(point[other_dimensions] .> upper_limit)
    error(string("Slice axis is outside of domain. ",
        " point[$other_dimensions]=$(point[other_dimensions]) must be between $lower_limit and $upper_limit"))
  end

  # Count the amount of new elements.
  new_id = 0

  # Permute dimensions so that the slice dimensions are always the in correct places for later use.
  if slice === :x
    original_nodes = permutedims(original_nodes, [1, 3, 4, 2, 5])
    unstructured_data = permutedims(unstructured_data, [2, 3, 1, 4, 5])
  elseif slice === :y
    original_nodes = permutedims(original_nodes, [1, 2, 4, 3, 5])
    unstructured_data = permutedims(unstructured_data, [1, 3, 2, 4, 5])
  end

  # Iterate over all elements to find the ones that lie on the slice axis.
  for element_id in 1:n_elements
    min_coordinate = original_nodes[:, 1, 1, 1, element_id]
    max_coordinate = original_nodes[:, n_nodes_in, n_nodes_in, n_nodes_in, element_id]
    element_length = max_coordinate - min_coordinate

    # Test if the element is on the slice axis. If not just continue with the next element.
    if !((prod(min_coordinate[other_dimensions] .<= point[other_dimensions]) &&
        prod(max_coordinate[other_dimensions] .> point[other_dimensions])) ||
        (point[other_dimensions] == upper_limit && prod(max_coordinate[other_dimensions] .== upper_limit)))

        continue
    end

    new_id += 1

    # Construct vandermonde matrix for interpolation of each 2D element to a 1D element.
    normalized_intercept =
          (point[other_dimensions] .- min_coordinate[other_dimensions]) /
          element_length[1] * 2 .- 1
    vandermonde_i = polynomial_interpolation_matrix(nodes_in, normalized_intercept[1])
    vandermonde_ii = polynomial_interpolation_matrix(nodes_in, normalized_intercept[2])

    # Interpolate to each node of new 1D element.
    for v in 1:n_variables
      for i in 1:n_nodes_in
        for ii in 1:n_nodes_in
          temp_unstructured_data[i, ii, new_id, v] = (vandermonde_ii*unstructured_data[ii, :, i, element_id, v])[1]
        end
        new_unstructured_data[i, new_id, v] = (vandermonde_i*temp_unstructured_data[i, :, new_id, v])[1]
      end
    end

    new_nodes[:, new_id] = original_nodes[slice_dimension, 1, 1, :, element_id]
  end

  return get_data_1d(reshape(new_nodes[:, 1:new_id], 1, n_nodes_in, new_id), new_unstructured_data[:, 1:new_id, :], nvisnodes)
end

# Interpolate unstructured DG data to structured data (cell-centered)
#
# This function takes DG data in an unstructured, Cartesian layout and converts it to a uniformely
# distributed 2D layout.
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function unstructured2structured(unstructured_data, normalized_coordinates,
                                 levels, resolution, nvisnodes_per_level)
  # Extract data shape information
  n_nodes_in, _, n_elements, n_variables = size(unstructured_data)

  # Get node coordinates for DG locations on reference element
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)

  # Calculate interpolation vandermonde matrices for each level
  max_level = length(nvisnodes_per_level) - 1
  vandermonde_per_level = []
  for l in 0:max_level
    n_nodes_out = nvisnodes_per_level[l + 1]
    dx = 2 / n_nodes_out
    nodes_out = collect(range(-1 + dx/2, 1 - dx/2, length=n_nodes_out))
    push!(vandermonde_per_level, polynomial_interpolation_matrix(nodes_in, nodes_out))
  end

  # For each element, calculate index position at which to insert data in global data structure
  lower_left_index = element2index(normalized_coordinates, levels, resolution, nvisnodes_per_level)

  # Create output data structure
  structured = [Matrix{Float64}(undef, resolution, resolution) for _ in 1:n_variables]

  # For each variable, interpolate element data and store to global data structure
  for v in 1:n_variables
    # Reshape data array for use in multiply_dimensionwise function
    reshaped_data = reshape(unstructured_data[:, :, :, v], 1, n_nodes_in, n_nodes_in, n_elements)

    for element_id in 1:n_elements
      # Extract level for convenience
      level = levels[element_id]

      # Determine target indices
      n_nodes_out = nvisnodes_per_level[level + 1]
      first = lower_left_index[:, element_id]
      last = first .+ (n_nodes_out - 1)

      # Interpolate data
      vandermonde = vandermonde_per_level[level + 1]
      structured[v][first[1]:last[1], first[2]:last[2]] .= (
          reshape(multiply_dimensionwise(vandermonde, reshaped_data[:, :, :, element_id]),
                  n_nodes_out, n_nodes_out))
    end
  end

  return structured
end


# For a given normalized element coordinate, return the index of its lower left
# contribution to the global data structure
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function element2index(normalized_coordinates, levels, resolution, nvisnodes_per_level)
  @assert size(normalized_coordinates, 1) == 2 "only works in 2D"

  n_elements = length(levels)

  # First, determine lower left coordinate for all cells
  dx = 2 / resolution
  ndim = 2
  lower_left_coordinate = Array{Float64}(undef, ndim, n_elements)
  for element_id in 1:n_elements
    nvisnodes = nvisnodes_per_level[levels[element_id] + 1]
    lower_left_coordinate[1, element_id] = (
        normalized_coordinates[1, element_id] - (nvisnodes - 1)/2 * dx)
    lower_left_coordinate[2, element_id] = (
        normalized_coordinates[2, element_id] - (nvisnodes - 1)/2 * dx)
  end

  # Then, convert coordinate to global index
  indices = coordinate2index(lower_left_coordinate, resolution)

  return indices
end


# Find 2D array index for a 2-tuple of normalized, cell-centered coordinates (i.e., in [-1,1])
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function coordinate2index(coordinate, resolution::Integer)
  # Calculate 1D normalized coordinates
  dx = 2/resolution
  mesh_coordinates = collect(range(-1 + dx/2, 1 - dx/2, length=resolution))

  # Find index
  id_x = searchsortedfirst.(Ref(mesh_coordinates), coordinate[1, :], lt=(x,y)->x .< y .- dx/2)
  id_y = searchsortedfirst.(Ref(mesh_coordinates), coordinate[2, :], lt=(x,y)->x .< y .- dx/2)
  return transpose(hcat(id_x, id_y))
end


# Calculate the vertices for each mesh cell such that it can be visualized as a closed box
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function calc_vertices(coordinates, levels, length_level_0)
  ndim = size(coordinates, 1)
  @assert ndim == 2 "only works in 2D"

  # Initialize output arrays
  n_elements = length(levels)
  n_points_per_element = 2^ndim+2
  x = Vector{Float64}(undef, n_points_per_element*n_elements)
  y = Vector{Float64}(undef, n_points_per_element*n_elements)

  # Calculate vertices for all coordinates at once
  for element_id in 1:n_elements
    length = length_level_0 / 2^levels[element_id]
    index = n_points_per_element*(element_id-1)
    x[index+1] = coordinates[1, element_id] - 1/2 * length
    x[index+2] = coordinates[1, element_id] + 1/2 * length
    x[index+3] = coordinates[1, element_id] + 1/2 * length
    x[index+4] = coordinates[1, element_id] - 1/2 * length
    x[index+5] = coordinates[1, element_id] - 1/2 * length
    x[index+6] = NaN

    y[index+1] = coordinates[2, element_id] - 1/2 * length
    y[index+2] = coordinates[2, element_id] - 1/2 * length
    y[index+3] = coordinates[2, element_id] + 1/2 * length
    y[index+4] = coordinates[2, element_id] + 1/2 * length
    y[index+5] = coordinates[2, element_id] - 1/2 * length
    y[index+6] = NaN
  end

  return x, y
end


# Calculate the vertices to plot each grid line for StructuredMesh
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function calc_vertices(node_coordinates, mesh)
  @unpack cells_per_dimension = mesh
  @assert size(node_coordinates, 1) == 2 "only works in 2D"

  linear_indices = LinearIndices(size(mesh))

  # Initialize output arrays
  n_lines = sum(cells_per_dimension) + 2
  max_length = maximum(cells_per_dimension)
  n_nodes = size(node_coordinates, 2)

  # Create output as two matrices `x` and `y`, each holding the node locations for each of the `n_lines` grid lines
  # The # of rows in the matrices must be sufficient to store the longest dimension (`max_length`),
  # and for each the node locations without doubling the corner nodes (`n_nodes-1`), plus the final node (`+1`)
  # Rely on Plots.jl to ignore `NaN`s (i.e., they are not plotted) to handle shorter lines
  x = fill(NaN, max_length*(n_nodes-1)+1, n_lines)
  y = fill(NaN, max_length*(n_nodes-1)+1, n_lines)

  line_index = 1
  # Lines in x-direction
  # Bottom boundary
  i = 1
  for cell_x in axes(mesh, 1)
    for node in 1:(n_nodes-1)
      x[i, line_index] = node_coordinates[1, node, 1, linear_indices[cell_x, 1]]
      y[i, line_index] = node_coordinates[2, node, 1, linear_indices[cell_x, 1]]

      i += 1
    end
  end
  # Last point on bottom boundary
  x[i, line_index] = node_coordinates[1, end, 1, linear_indices[end, 1]]
  y[i, line_index] = node_coordinates[2, end, 1, linear_indices[end, 1]]

  # Other lines in x-direction
  line_index += 1
  for cell_y in axes(mesh, 2)
    i = 1
    for cell_x in axes(mesh, 1)
      for node in 1:(n_nodes-1)
        x[i, line_index] = node_coordinates[1, node, end, linear_indices[cell_x, cell_y]]
        y[i, line_index] = node_coordinates[2, node, end, linear_indices[cell_x, cell_y]]

        i += 1
      end
    end
    # Last point on line
    x[i, line_index] = node_coordinates[1, end, end, linear_indices[end, cell_y]]
    y[i, line_index] = node_coordinates[2, end, end, linear_indices[end, cell_y]]

    line_index += 1
  end


  # Lines in y-direction
  # Left boundary
  i = 1
  for cell_y in axes(mesh, 2)
    for node in 1:(n_nodes-1)
      x[i, line_index] = node_coordinates[1, 1, node, linear_indices[1, cell_y]]
      y[i, line_index] = node_coordinates[2, 1, node, linear_indices[1, cell_y]]

      i += 1
    end
  end
  # Last point on left boundary
  x[i, line_index] = node_coordinates[1, 1, end, linear_indices[1, end]]
  y[i, line_index] = node_coordinates[2, 1, end, linear_indices[1, end]]

  # Other lines in y-direction
  line_index +=1
  for cell_x in axes(mesh, 1)
    i = 1
    for cell_y in axes(mesh, 2)
      for node in 1:(n_nodes-1)
        x[i, line_index] = node_coordinates[1, end, node, linear_indices[cell_x, cell_y]]
        y[i, line_index] = node_coordinates[2, end, node, linear_indices[cell_x, cell_y]]

        i += 1
      end
    end
    # Last point on line
    x[i, line_index] = node_coordinates[1, end, end, linear_indices[cell_x, end]]
    y[i, line_index] = node_coordinates[2, end, end, linear_indices[cell_x, end]]

    line_index += 1
  end

  return x, y
end

# Convert `slice` to orientations (1 -> `x`, 2 -> `y`, 3 -> `z`) for the two axes in a 2D plot
function _get_orientations(mesh, slice)
  if ndims(mesh) == 2 || (ndims(mesh) == 3 && slice === :xy)
    orientation_x = 1
    orientation_y = 2
  elseif ndims(mesh) == 3 && slice === :xz
    orientation_x = 1
    orientation_y = 3
  elseif ndims(mesh) == 3 && slice === :yz
    orientation_x = 2
    orientation_y = 3
  else
    orientation_x = 0
    orientation_y = 0
  end
  return orientation_x, orientation_y
end


"""
    getmesh(pd::AbstractPlotData)

Extract grid lines from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
getmesh(pd::AbstractPlotData) = PlotMesh(pd)

# Convert `orientation` into a guide label (see also `_get_orientations`)
function _get_guide(orientation::Integer)
  if orientation == 1
    return "\$x\$"
  elseif orientation == 2
    return "\$y\$"
  elseif orientation == 3
    return "\$z\$"
  else
    return ""
  end
end
