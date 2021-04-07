# Initialize data structures in element container
function init_elements!(elements, mesh::CurvedMesh{2}, basis::LobattoLegendreBasis)
  @unpack faces = mesh
  @unpack node_coordinates, left_neighbors, metric_terms, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate node coordinates, metric terms, and inverse Jacobian
  for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    element = linear_indices[cell_x, cell_y]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, mesh, basis)

    calc_metric_terms!(metric_terms, element, node_coordinates, basis)

    calc_inverse_jacobian!(inverse_jacobian, element, metric_terms)
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y,
                                mesh::CurvedMesh{2},
                                basis::LobattoLegendreBasis)
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  dy = 2 / size(mesh, 2)

  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  cell_y_offset = -1 + (cell_y-1) * dy + dy/2

  for j in eachindex(nodes), i in eachindex(nodes)
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, j, element] .= transfinite_mapping(cell_x_offset + dx/2 * nodes[i],
                                                              cell_y_offset + dy/2 * nodes[j], mesh)
  end
end


# Calculate metric terms of the mapping from the reference element to the element in the physical domain
function calc_metric_terms!(metric_terms, element, node_coordinates::AbstractArray{<:Any, 4}, basis::LobattoLegendreBasis)
  @views mul!(metric_terms[1, 1, :, :, element], basis.derivative_matrix, node_coordinates[1, :, :, element]) # x_ξ
  @views mul!(metric_terms[2, 1, :, :, element], basis.derivative_matrix, node_coordinates[2, :, :, element]) # y_ξ
  @views mul!(metric_terms[1, 2, :, :, element], node_coordinates[1, :, :, element], basis.derivative_matrix') # x_η
  @views mul!(metric_terms[2, 2, :, :, element], node_coordinates[2, :, :, element], basis.derivative_matrix') # y_η

  return metric_terms
end


# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 3}, element, metric_terms)
  @. @views inverse_jacobian[:, :, element] = inv(metric_terms[1, 1, :, :, element] * metric_terms[2, 2, :, :, element] -
                                                  metric_terms[1, 2, :, :, element] * metric_terms[2, 1, :, :, element])

  return inverse_jacobian
end


# Save id of left neighbor of every element
function initialize_neighbor_connectivity!(left_neighbors, mesh::CurvedMesh{2}, linear_indices)
  # Neighbors in x-direction
  for cell_y in 1:size(mesh, 2)
    # Inner elements
    for cell_x in 2:size(mesh, 1)
      element = linear_indices[cell_x, cell_y]
      left_neighbors[1, element] = linear_indices[cell_x - 1, cell_y]
    end

    if isperiodic(mesh, 1)
      # Periodic boundary
      left_neighbors[1, linear_indices[1, cell_y]] = linear_indices[end, cell_y]
    else
      # Use boundary conditions
      left_neighbors[1, linear_indices[1, cell_y]] = -1
    end
  end

  # Neighbors in y-direction
  for cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y]
      left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1]
    end

    if isperiodic(mesh, 2)
      # Periodic boundary
      left_neighbors[2, linear_indices[cell_x, 1]] = linear_indices[cell_x, end]
    else
      # Use boundary conditions
      left_neighbors[2, linear_indices[cell_x, 1]] = -1
    end
  end

  return left_neighbors
end
