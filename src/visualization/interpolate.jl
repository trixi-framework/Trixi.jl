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


# Convert 3d unstructured data to 2d slice.
# Additional to the new unstructured data updated coordinates, levels and
# center coordinates are returned.
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function unstructured_2d_to_3d(unstructured_data, coordinates, levels,
                               length_level_0, center_level_0, slice_axis,
                               slice_axis_intercept)

  if slice_axis === :x
    slice_axis_dimension = 1
    other_dimensions = [2, 3]
  elseif slice_axis === :y
    slice_axis_dimension = 2
    other_dimensions = [1, 3]
  elseif slice_axis === :z
    slice_axis_dimension = 3
    other_dimensions = [1, 2]
  else
    error("illegal dimension '$slice_axis', supported dimensions are :x, :y, and :z")
  end

  # Limits of domain in slice_axis dimension
  lower_limit = center_level_0[slice_axis_dimension] - length_level_0 / 2
  upper_limit = center_level_0[slice_axis_dimension] + length_level_0 / 2

  if slice_axis_intercept < lower_limit || slice_axis_intercept > upper_limit
    error(string("Slice plane $slice_axis = $slice_axis_intercept outside of domain. ",
        "$slice_axis must be between $lower_limit and $upper_limit"))
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

  # Permute dimensions such that the slice axis dimension is always the
  # third dimension of the array. Below we can always interpolate in the 
  # third dimension.
  if slice_axis === :x
    unstructured_data = permutedims(unstructured_data, [2, 3, 1, 4, 5])
  elseif slice_axis === :y
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
    if !((min_coordinate[slice_axis_dimension] <= slice_axis_intercept &&
          max_coordinate[slice_axis_dimension] > slice_axis_intercept) ||
        (slice_axis_intercept == upper_limit &&
          max_coordinate[slice_axis_dimension] == upper_limit))
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
        (slice_axis_intercept - min_coordinate[slice_axis_dimension]) /
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
function element2index(normalized_coordinates::AbstractArray{Float64}, levels::AbstractArray{Int},
                       resolution::Int, nvisnodes_per_level::AbstractArray{Int})
  n_elements = length(levels)
  ndim = 2 # FIXME

  # First, determine lower left coordinate for all cells
  dx = 2 / resolution
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


# Interpolate data for plotting
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function interpolate_data(data_in::AbstractArray, n_nodes_in::Integer, n_nodes_out::Integer)
  # Get node coordinates for input and output locations on reference element
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)
  dx = 2/n_nodes_out
  #=nodes_out = collect(range(-1 + dx/2, 1 - dx/2, length=n_nodes_out))=#
  nodes_out = collect(range(-1, 1, length=n_nodes_out))

  # Get interpolation matrix
  vandermonde = polynomial_interpolation_matrix(nodes_in, nodes_out)

  # Create output data structure
  ndim = 2 # FIXME
  n_elements = div(size(data_in, 1), n_nodes_in^ndim)
  n_variables = size(data_in, 2)
  data_out = Array{eltype(data_in)}(undef, n_nodes_out, n_nodes_out, n_elements, n_variables)

  # Interpolate each variable separately
  for v = 1:n_variables
    # Reshape data to fit expected format for interpolation function
    # FIXME: this "reshape here, reshape later" funny business should be implemented properly
    reshaped = reshape(data_in[:, v], 1, n_nodes_in, n_nodes_in, n_elements)

    # Interpolate data for each cell
    for element_id = 1:1#n_elements
      data_out[:, :, element_id, v] = multiply_dimensionwise(vandermonde, reshaped[:, :, :, element_id])
    end
  end

  return reshape(data_out, n_nodes_out^ndim * n_elements, n_variables)
end


# Calculate the vertices for each mesh cell such that it can be visualized as a closed box
#
# Note: This is a low-level function that is not considered as part of Trixi's interface and may
#       thus be changed in future releases.
function calc_vertices(coordinates, levels, length_level_0)
  ndim = 2 # FIXME
  @assert ndim == 2 "Algorithm currently only works in 2D"

  # Initialize output arrays
  n_elements = length(levels)
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
