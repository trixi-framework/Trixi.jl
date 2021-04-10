# Initialize data structures in element container
function init_elements!(elements, mesh::CurvedMesh{3}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, left_neighbors, 
          jacobian_matrix, contravariant_vectors, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
  for cell_z in 1:size(mesh, 3), cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    element = linear_indices[cell_x, cell_y, cell_z]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, cell_z, mesh.mapping, mesh, basis)

    calc_jacobian_matrix!(jacobian_matrix, element, mesh, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

    calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix, basis)
    
  end

  initialize_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
# `mesh.mapping` is passed as an additional argument for type stability (function barrier)
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, cell_z,
                                mapping, mesh::CurvedMesh{3},
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
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, j, k, element] .= mapping(cell_x_offset + dx/2 * nodes[i],
                                                     cell_y_offset + dy/2 * nodes[j],
                                                     cell_z_offset + dz/2 * nodes[k])
  end
end


# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix!(jacobian_matrix, element, mesh, node_coordinates::AbstractArray{<:Any,5}, basis::LobattoLegendreBasis)
  @unpack mapping = mesh

  # TODO: Needs to be adjusted for actually curved meshes
  dx, dy, dz = (mapping(1, 1, 1) .- mapping(-1, -1, -1)) ./ size(mesh)

  for k in 1:nnodes(basis), j in 1:nnodes(basis), i in 1:nnodes(basis)
    jacobian_matrix[:, :, i, j, k, element] .= 0.5 * diagm([dx, dy, dz])
  end

  return jacobian_matrix
end


# Calculate contravarant vectors, multiplied by the Jacobian determinant J of the transformation mapping.
# Those are called Ja^i in Kopriva's blue book.
function calc_contravariant_vectors!(contravariant_vectors::AbstractArray{<:Any,6}, element, jacobian_matrix)
  # TODO: This needs to be adapted for actually curved meshes.
  # For rectangular meshes, the contravariant_vectors are just the scaled unit vectors
  fill!(view(contravariant_vectors, .., element), 0)

  @. @views contravariant_vectors[1, 1, :, :, :, element] =  jacobian_matrix[2, 2, :, :, :, element] * jacobian_matrix[3, 3, :, :, :, element]
  @. @views contravariant_vectors[2, 2, :, :, :, element] =  jacobian_matrix[1, 1, :, :, :, element] * jacobian_matrix[3, 3, :, :, :, element]
  @. @views contravariant_vectors[3, 3, :, :, :, element] =  jacobian_matrix[1, 1, :, :, :, element] * jacobian_matrix[2, 2, :, :, :, element]

  return contravariant_vectors
end


# # Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 4}, element, jacobian_matrix, basis)
  @unpack nodes = basis
  for k in eachindex(nodes), j in eachindex(nodes), i in eachindex(nodes)
    # Calculate Determinant by using Sarrus formula (about 100 times faster than LinearAlgebra.det())
    inverse_jacobian[i, j, k, element] = inv(
        jacobian_matrix[1, 1, i, j, k, element] * jacobian_matrix[2, 2, i, j, k, element] * jacobian_matrix[3, 3, i, j, k, element] +
        jacobian_matrix[1, 2, i, j, k, element] * jacobian_matrix[2, 3, i, j, k, element] * jacobian_matrix[3, 1, i, j, k, element] +
        jacobian_matrix[1, 3, i, j, k, element] * jacobian_matrix[2, 1, i, j, k, element] * jacobian_matrix[3, 2, i, j, k, element] -
        jacobian_matrix[3, 1, i, j, k, element] * jacobian_matrix[2, 2, i, j, k, element] * jacobian_matrix[1, 3, i, j, k, element] -
        jacobian_matrix[3, 2, i, j, k, element] * jacobian_matrix[2, 3, i, j, k, element] * jacobian_matrix[1, 1, i, j, k, element] -
        jacobian_matrix[3, 3, i, j, k, element] * jacobian_matrix[2, 1, i, j, k, element] * jacobian_matrix[1, 2, i, j, k, element] ) 
  end
        
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
  for cell_z in 1:size(mesh, 3), cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y, cell_z]
      left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1, cell_z]
    end

    # Periodic boundary
    left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = linear_indices[cell_x, end, cell_z]
  end

  # Neighbors in z-direction
  for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
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
