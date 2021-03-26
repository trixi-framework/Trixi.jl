function init_elements!(elements, mesh::StructuredMesh{2}, basis::LobattoLegendreBasis{T, NNODES}) where {T, NNODES}
  @unpack faces = mesh
  @unpack node_coordinates, left_neighbors, metric_terms, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate inverse Jacobian and node coordinates
  for cell_x in 1:size(mesh, 1), cell_y in 1:size(mesh, 2)
    element = linear_indices[cell_x, cell_y]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, mesh, basis)

    calc_metric_terms!(metric_terms, element, node_coordinates, basis)

    calc_inverse_jacobian!(inverse_jacobian, element, metric_terms)
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

  return nothing
end


function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y,
                                mesh::StructuredMesh{2},
                                basis::LobattoLegendreBasis{T, NNODES}) where {T, NNODES}
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  dy = 2 / size(mesh, 2)

  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  cell_y_offset = -1 + (cell_y-1) * dy + dy/2

  for j in 1:NNODES, i in 1:NNODES
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, j, element] .= bilinear_mapping(cell_x_offset + dx/2 * nodes[i],
                                                           cell_y_offset + dy/2 * nodes[j], mesh)
  end
end


function calc_metric_terms!(metric_terms, element, node_coordinates::Array{RealT, 4}, 
                           basis::LobattoLegendreBasis{T, NNODES}) where {RealT, T, NNODES}
  @views metric_terms[1, 1, :, :, element] .= basis.derivative_matrix * node_coordinates[1, :, :, element] # x_ξ
  @views metric_terms[2, 1, :, :, element] .= basis.derivative_matrix * node_coordinates[2, :, :, element] # y_ξ
  @views metric_terms[1, 2, :, :, element] .= node_coordinates[1, :, :, element] * basis.derivative_matrix' # x_η
  @views metric_terms[2, 2, :, :, element] .= node_coordinates[2, :, :, element] * basis.derivative_matrix' # y_η

  return metric_terms
end


function calc_inverse_jacobian!(inverse_jacobian::Array{RealT, 3}, element, metric_terms) where {RealT}
  @views inverse_jacobian[:, :, element] .= inv.(metric_terms[1, 1, :, :, element] .* metric_terms[2, 2, :, :, element] .-
                                                 metric_terms[1, 2, :, :, element] .* metric_terms[2, 1, :, :, element])

  return inverse_jacobian
end


function initialize_neighbor_connectivity!(left_neighbors, mesh::StructuredMesh{2}, linear_indices)
  # Neighbors in x-direction
  for cell_y in 1:size(mesh, 2)
    # Inner elements
    for cell_x in 2:size(mesh, 1)
      element = linear_indices[cell_x, cell_y]
      left_neighbors[1, element] = linear_indices[cell_x - 1, cell_y]
    end

    # Periodic boundary
    left_neighbors[1, linear_indices[1, cell_y]] = linear_indices[end, cell_y]
  end

  # Neighbors in y-direction
  for cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y]
      left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1]
    end

    # Periodic boundary
    left_neighbors[2, linear_indices[cell_x, 1]] = linear_indices[cell_x, end]
  end

  return left_neighbors
end
