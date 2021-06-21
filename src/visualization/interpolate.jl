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

function calc_arc_length(coordinates)
  arc_length = 0
  for i in 1:size(coordinates)[2]-1
    arc_length = arc_length + sqrt(sum((coordinates[:,i]-coordinates[:,i+1]).^2))
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
  nodes_on_curve = collect(range(0, arc_length, length = n_points_curve))

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

  return nodes_on_curve, data_on_curve, nothing
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
  nodes_on_curve = collect(range(0, arc_length, length = n_points_curve))

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

  return nodes_on_curve, data_on_curve, nothing
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
  @assert size(coordinates, 1) == 2 "only works in 2D"

  # Initialize output arrays
  n_elements = length(levels)
  ndim = 2
  x = Matrix{Float64}(undef, 2^ndim+1, n_elements)
  y = Matrix{Float64}(undef, 2^ndim+1, n_elements)

  # Calculate vertices for all coordinates at once
  for element_id in 1:n_elements
    length = length_level_0 / 2^levels[element_id]
    x[1, element_id] = coordinates[1, element_id] - 1/2 * length
    x[2, element_id] = coordinates[1, element_id] + 1/2 * length
    x[3, element_id] = coordinates[1, element_id] + 1/2 * length
    x[4, element_id] = coordinates[1, element_id] - 1/2 * length
    x[5, element_id] = coordinates[1, element_id] - 1/2 * length

    y[1, element_id] = coordinates[2, element_id] - 1/2 * length
    y[2, element_id] = coordinates[2, element_id] - 1/2 * length
    y[3, element_id] = coordinates[2, element_id] + 1/2 * length
    y[4, element_id] = coordinates[2, element_id] + 1/2 * length
    y[5, element_id] = coordinates[2, element_id] - 1/2 * length
  end

  return x, y
end


# Calculate the vertices to plot each grid line for CurvedMesh
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


# Calculate the vertices to plot each grid line for UnstructuredQuadMesh
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function calc_vertices(node_coordinates, mesh::UnstructuredQuadMesh)
  @unpack n_elements = mesh
  @assert size(node_coordinates, 1) == 2 "only works in 2D"

  # Initialize output arrays
  n_nodes = size(node_coordinates, 2)
  x = fill(NaN, 4*n_nodes, n_elements)
  y = fill(NaN, 4*n_nodes, n_elements)

  # Lines along the first local sides
  for element in 1:n_elements
#    for node in 1:n_nodes
      x[1:n_nodes, element] = node_coordinates[1, 1:n_nodes, 1, element]
      y[1:n_nodes, element] = node_coordinates[2, 1:n_nodes, 1, element]
#    end
  end

  # Lines along all the second local sides
  for element in 1:n_elements
    x[n_nodes+1:2*n_nodes, element] = node_coordinates[1, end, 1:n_nodes, element]
    y[n_nodes+1:2*n_nodes, element] = node_coordinates[2, end, 1:n_nodes, element]
  end

  # Lines along all the third local sides
  for element in 1:n_elements
    x[2*n_nodes+1:3*n_nodes, element] = node_coordinates[1, n_nodes:-1:1, end, element]
    y[2*n_nodes+1:3*n_nodes, element] = node_coordinates[2, n_nodes:-1:1, end, element]
  end

  # Lines along all the fourth local sides
  for element in 1:n_elements
    x[3*n_nodes+1:4*n_nodes, element] = node_coordinates[1, 1, n_nodes:-1:1, element]
    y[3*n_nodes+1:4*n_nodes, element] = node_coordinates[2, 1, n_nodes:-1:1, element]
  end

  return x, y
end
