# Initialize data structures in element container
function init_elements!(elements, mesh::CurvedMesh{2}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, left_neighbors,
          jacobian_matrix, contravariant_vectors, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
  for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    element = linear_indices[cell_x, cell_y]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, mesh.mapping, mesh, basis)

    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

    calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
# `mesh.mapping` is passed as an additional argument for type stability (function barrier)
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, mapping,
                                mesh::CurvedMesh{2},
                                basis::LobattoLegendreBasis)
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  dy = 2 / size(mesh, 2)

  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  cell_y_offset = -1 + (cell_y-1) * dy + dy/2

  for j in eachnode(basis), i in eachnode(basis)
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, j, element] .= mapping(cell_x_offset + dx/2 * nodes[i],
                                                  cell_y_offset + dy/2 * nodes[j])
  end
end


# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates::AbstractArray{<:Any, 4}, basis::LobattoLegendreBasis)
  @views mul!(jacobian_matrix[1, 1, :, :, element], basis.derivative_matrix, node_coordinates[1, :, :, element]) # x_ξ
  @views mul!(jacobian_matrix[2, 1, :, :, element], basis.derivative_matrix, node_coordinates[2, :, :, element]) # y_ξ
  @views mul!(jacobian_matrix[1, 2, :, :, element], node_coordinates[1, :, :, element], basis.derivative_matrix') # x_η
  @views mul!(jacobian_matrix[2, 2, :, :, element], node_coordinates[2, :, :, element], basis.derivative_matrix') # y_η

  return jacobian_matrix
end


# Calculate contravarant vectors, multiplied by the Jacobian determinant J of the transformation mapping.
# Those are called Ja^i in Kopriva's blue book.
function calc_contravariant_vectors!(contravariant_vectors::AbstractArray{<:Any,5}, element, jacobian_matrix)
  # First contravariant vector Ja^1
  @. @views contravariant_vectors[1, 1, :, :, element] =  jacobian_matrix[2, 2, :, :, element]
  @. @views contravariant_vectors[2, 1, :, :, element] = -jacobian_matrix[1, 2, :, :, element]
  # Second contravariant vector Ja^2
  @. @views contravariant_vectors[1, 2, :, :, element] = -jacobian_matrix[2, 1, :, :, element]
  @. @views contravariant_vectors[2, 2, :, :, element] =  jacobian_matrix[1, 1, :, :, element]

  return contravariant_vectors
end


# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any,3}, element, jacobian_matrix)
  @. @views inverse_jacobian[:, :, element] = inv(jacobian_matrix[1, 1, :, :, element] * jacobian_matrix[2, 2, :, :, element] -
                                                  jacobian_matrix[1, 2, :, :, element] * jacobian_matrix[2, 1, :, :, element])

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
      left_neighbors[1, linear_indices[1, cell_y]] = 0
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
      left_neighbors[2, linear_indices[cell_x, 1]] = 0
    end
  end

  return left_neighbors
end
