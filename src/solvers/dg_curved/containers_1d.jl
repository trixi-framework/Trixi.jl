# Initialize data structures in element container
function init_elements!(elements, mesh::CurvedMesh{1}, basis::LobattoLegendreBasis)
  @unpack faces = mesh
  @unpack node_coordinates, left_neighbors, metric_terms, inverse_jacobian = elements

  # Calculate node coordinates, metric terms, and inverse Jacobian
  for cell_x in 1:size(mesh, 1)
    calc_node_coordinates!(node_coordinates, cell_x, mesh, basis)

    calc_metric_terms!(metric_terms, cell_x, node_coordinates, basis)

    calc_inverse_jacobian!(inverse_jacobian, cell_x, metric_terms)
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh)

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates, cell_x, mesh::CurvedMesh{1},
                                basis::LobattoLegendreBasis)
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  
  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  
  for i in eachindex(nodes)
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[1, i, cell_x] = linear_mapping(cell_x_offset + dx/2 * nodes[i], mesh)[1]
  end
end


# Calculate metric terms of the mapping from the reference element to the element in the physical domain
function calc_metric_terms!(metric_terms, element, node_coordinates::AbstractArray{<:Any, 3}, 
                            basis::LobattoLegendreBasis)
  @views mul!(metric_terms[1, 1, :, element], basis.derivative_matrix, node_coordinates[1, :, element]) # x_ξ
  
  return metric_terms
end


# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 2}, element, metric_terms)
  @views inverse_jacobian[:, element] .= inv.(metric_terms[1, 1, :, element])

  return inverse_jacobian
end


# Save id of left neighbor of every element
function initialize_neighbor_connectivity!(left_neighbors, mesh::CurvedMesh{1})
  # Neighbors in x-direction
  # Inner elements
  for cell_x in 2:size(mesh, 1)
    left_neighbors[1, cell_x] = cell_x - 1
  end

  if isperiodic(mesh)
    # Periodic boundary
    left_neighbors[1, 1] = size(mesh, 1)
  else
    # Use boundary conditions
    left_neighbors[1, 1] = -1
  end

  return left_neighbors
end
