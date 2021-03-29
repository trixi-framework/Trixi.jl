function init_elements!(elements, mesh::CurvedMesh{1}, basis::LobattoLegendreBasis{T, NNODES}) where {T, NNODES}
  @unpack faces = mesh
  @unpack node_coordinates, left_neighbors, metric_terms, inverse_jacobian = elements


  # Calculate inverse Jacobian and node coordinates
  for cell_x in 1:size(mesh, 1)
    element = cell_x

    calc_node_coordinates!(node_coordinates, element, cell_x, mesh, basis)

    calc_metric_terms!(metric_terms, element, node_coordinates, basis)

    calc_inverse_jacobian!(inverse_jacobian, element, metric_terms)
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh)

  return nothing
end


function calc_node_coordinates!(node_coordinates, element,
                                cell_x, mesh::CurvedMesh{1},
                                basis::LobattoLegendreBasis{T, NNODES}) where {T, NNODES}
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  
  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  
  for i in 1:NNODES
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, element] .= bilinear_mapping(cell_x_offset + dx/2 * nodes[i], mesh)
  end
end


function calc_metric_terms!(metric_terms, element, node_coordinates::Array{RealT, 3}, 
                           basis::LobattoLegendreBasis{T, NNODES}) where {RealT, T, NNODES}
  @views metric_terms[1, 1, :, element] .= basis.derivative_matrix * node_coordinates[1, :, element] # x_ξ
  
  return metric_terms
end


function calc_inverse_jacobian!(inverse_jacobian::Array{RealT, 2}, element, metric_terms) where {RealT}
  @views inverse_jacobian[:, element] = inv.(metric_terms[1, 1, :, element])

  return inverse_jacobian
end


function initialize_neighbor_connectivity!(left_neighbors, mesh::CurvedMesh{1})
  # Neighbors in x-direction
  # Inner elements
  for cell_x in 2:size(mesh, 1)
   element = cell_x
    left_neighbors[1, element] = cell_x - 1
  end

  # Periodic boundary
  left_neighbors[1, 1] = size(mesh, 1)
  

  return left_neighbors
end
