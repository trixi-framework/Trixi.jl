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
                     slice_axis=:z, slice_axis_intercept=0)
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
        center_level_0) = unstructured_2d_to_3d(unstructured_data,
        coordinates, levels, length_level_0, center_level_0, slice_axis,
        slice_axis_intercept)
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
    mesh_vertices_x = Matrix{Float64}(undef, 0, 0)
    mesh_vertices_y = Matrix{Float64}(undef, 0, 0)
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
    max_nvisnodes = nvisnodes
  end

  interpolated_nodes = collect(range(original_nodes[1,1,1], original_nodes[1,end,end], length = max_nvisnodes*n_elements+1))
  interpolated_data = Array{Float64,3}(undef, n_vars, max_nvisnodes, n_elements)

  # Calculate vandermonde matrix for interpolation.
  vandermonde = polynomial_interpolation_matrix(original_nodes[1,:,1], interpolated_nodes[1:max_nvisnodes])

  # Iterate over all variables.
  for i in 1:n_vars
    reshaped_data =reshape(unstructured_data[:, :, i], 1, n_nodes, n_elements)

    # Interpolate data for each element.
    for j in 1:n_elements
      interpolated_data[i,:,j] = vec(multiply_dimensionwise(vandermonde, reshaped_data[:, :, j]))
    end
  end

  # Return results after data is reshaped and the last index is appended.
  return interpolated_nodes, convert(Array{Float64}, vcat(reshape(interpolated_data, n_vars,:)',unstructured_data[end,end,:]'))
end

# Change order of dimensions (variables are now last) and convert data to `solution_variables`
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function get_unstructured_data(u, semi, solution_variables)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  if solution_variables === cons2cons
    raw_data = u
    n_vars = size(raw_data, 1)
  else
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    raw_data = Array(reinterpret(eltype(u),
           solution_variables.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
                      Ref(equations))))
    n_vars = size(raw_data, 1)
  end

  unstructured_data = Array{Float64}(undef,
                                     ntuple((d) -> nnodes(solver), ndims(equations))...,
                                     nelements(solver, cache), n_vars)
  for variable in 1:n_vars
    @views unstructured_data[.., :, variable] .= raw_data[variable, .., :]
  end

  return unstructured_data
end
