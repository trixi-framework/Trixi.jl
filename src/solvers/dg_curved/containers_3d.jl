# Initialize data structures in element container
function init_elements!(elements, mesh::CurvedMesh{3}, basis::LobattoLegendreBasis)
  @unpack faces = mesh
  @unpack node_coordinates, left_neighbors, metric_terms, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate node coordinates, metric terms, and inverse Jacobian
  for cell_z in 1:size(mesh, 3), cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    element = linear_indices[cell_x, cell_y, cell_z]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, cell_z, mesh, basis)

    calc_metric_terms!(metric_terms, element, mesh, node_coordinates, basis)

    calc_inverse_jacobian!(inverse_jacobian, element, metric_terms)
    
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, cell_z,
                                mesh::CurvedMesh{3},
                                basis::LobattoLegendreBasis)
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  dy = 2 / size(mesh, 2)
  dz = 2 / size(mesh, 3)

  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  cell_y_offset = -1 + (cell_y-1) * dy + dy/2
  cell_z_offset = -1 + (cell_z-1) * dz + dz/2

  for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
    # node_coordinates are the mapped reference node_coordinates TODO: Needs to be adjusted for "full" curved
    node_coordinates[:, i, j, k, element] .= trilinear_mapping(cell_x_offset + dx/2 * nodes[i],
                                                               cell_y_offset + dy/2 * nodes[j],
                                                               cell_z_offset + dz/2 * nodes[k], mesh)
  end
end


# Calculate metric terms of the mapping from the reference element to the element in the physical domain
function calc_metric_terms!(metric_terms, element, mesh, node_coordinates::AbstractArray{<:Any, 5}, basis::LobattoLegendreBasis)
  @unpack faces = mesh
  @unpack nodes = basis

  dx, dy, dz = (faces[2](1, 1) .- faces[1](-1, -1)) ./ size(mesh)

  for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
    metric_terms[:, :, i, j, k, element] .= 0.5 * Diagonal([dx, dy, dz])
  end

  return metric_terms
end


# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 4}, element, metric_terms)
  @. @views inverse_jacobian[:, :, :, element] = inv(metric_terms[1, 1, 1, 1, 1, element] * metric_terms[2, 2, 1, 1, 1, element] * metric_terms[3, 3, 1, 1, 1, element])                                        
  return inverse_jacobian
end


# Save id of left neighbor of every element
function initialize_neighbor_connectivity!(left_neighbors, mesh::CurvedMesh{3}, linear_indices)
  # Neighbors in x-direction
  for cell_z in 1:size(mesh, 3), cell_y in 1:size(mesh, 2)
    # Inner elements
    for cell_x in 2:size(mesh, 1)
      element = linear_indices[cell_x, cell_y, cell_z]
      left_neighbors[1, element] = linear_indices[cell_x - 1, cell_y, cell_z]
    end

    # Periodic boundary
    left_neighbors[1, linear_indices[1, cell_y, cell_z]] = linear_indices[end, cell_y, cell_z]

  end

  # Neighbors in y-direction
  for  cell_z in 1:size(mesh, 3), cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y, cell_z]
      left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1, cell_z]
    end

    # Periodic boundary
    left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = linear_indices[cell_x, end, cell_z]
  end

  # Neighbors in z-direction
  for  cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_z in 2:size(mesh, 3)
      element = linear_indices[cell_x, cell_y, cell_z]
      left_neighbors[3, element] = linear_indices[cell_x, cell_y, cell_z - 1]
    end

    # Periodic boundary
    left_neighbors[3, linear_indices[cell_x, cell_y, 1]] = linear_indices[cell_x, cell_y, end]
  end


  return left_neighbors
end
