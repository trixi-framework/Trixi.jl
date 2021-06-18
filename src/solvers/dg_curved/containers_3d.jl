
# Initialize data structures in element container
function init_elements!(elements, mesh::CurvedMesh{3}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, left_neighbors,
          jacobian_matrix, contravariant_vectors, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
  for cell_z in 1:size(mesh, 3), cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    element = linear_indices[cell_x, cell_y, cell_z]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, cell_z, mesh.mapping, mesh, basis)

    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix, node_coordinates, basis)

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

  for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, j, k, element] .= mapping(cell_x_offset + dx/2 * nodes[i],
                                                     cell_y_offset + dy/2 * nodes[j],
                                                     cell_z_offset + dz/2 * nodes[k])
  end
end


# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix!(jacobian_matrix::AbstractArray{<:Any,6}, element, node_coordinates, basis)
  for dim in 1:3, j in eachnode(basis), i in eachnode(basis)
    # ∂/∂ξ
    @views mul!(jacobian_matrix[dim, 1, :, i, j, element], basis.derivative_matrix, node_coordinates[dim, :, i, j, element])
    # ∂/∂η
    @views mul!(jacobian_matrix[dim, 2, i, :, j, element], basis.derivative_matrix, node_coordinates[dim, i, :, j, element])
    # ∂/∂ζ
    @views mul!(jacobian_matrix[dim, 3, i, j, :, element], basis.derivative_matrix, node_coordinates[dim, i, j, :, element])
  end

  return jacobian_matrix
end


# Calculate contravariant vectors, multiplied by the Jacobian determinant J of the transformation mapping,
# using the invariant curl form.
# These are called Ja^i in Kopriva's blue book.
function calc_contravariant_vectors!(contravariant_vectors::AbstractArray{<:Any,6}, element,
                                     jacobian_matrix, node_coordinates, basis::LobattoLegendreBasis)
  # The general form is
  # Jaⁱₙ = 0.5 * ( ∇ × (Xₘ ∇ Xₗ - Xₗ ∇ Xₘ) )ᵢ  where (n, m, l) cyclic and ∇ = (∂/∂ξ, ∂/∂η, ∂/∂ζ)ᵀ

  # Calculate the first summand of the cross product in each dimension
  for n in 1:3, j in eachnode(basis), i in eachnode(basis)
    # (n, m, l) cyclic
    m = (n % 3) + 1
    l = ((n + 1) % 3) + 1

    # Calc only the first summand 0.5 * (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η of
    # Ja¹ₙ = 0.5 * [ (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η - (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ ]
    @views contravariant_vectors[n, 1, i, :, j, element] = 0.5 * basis.derivative_matrix * (
        node_coordinates[m, i, :, j, element] .* jacobian_matrix[l, 3, i, :, j, element] .-
        node_coordinates[l, i, :, j, element] .* jacobian_matrix[m, 3, i, :, j, element])

    # Calc only the first summand 0.5 * (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ of
    # Ja²ₙ = 0.5 * [ (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ - (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ ]
    @views contravariant_vectors[n, 2, i, j, :, element] = 0.5 * basis.derivative_matrix * (
        node_coordinates[m, i, j, :, element] .* jacobian_matrix[l, 1, i, j, :, element] .-
        node_coordinates[l, i, j, :, element] .* jacobian_matrix[m, 1, i, j, :, element])

    # Calc only the first summand 0.5 * (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ of
    # Ja³ₙ = 0.5 * [ (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ - (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η ]
    @views contravariant_vectors[n, 3, :, i, j, element] = 0.5 * basis.derivative_matrix * (
        node_coordinates[m, :, i, j, element] .* jacobian_matrix[l, 2, :, i, j, element] .-
        node_coordinates[l, :, i, j, element] .* jacobian_matrix[m, 2, :, i, j, element])
  end

  # Calculate the second summand of the cross product in each dimension
  for n in 1:3, j in eachnode(basis), i in eachnode(basis)
    # (n, m, l) cyclic
    m = (n % 3) + 1
    l = ((n + 1) % 3) + 1

    # Calc only the second summand -0.5 * (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ of
    # Ja¹ₙ = 0.5 * [ (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_η - (Xₘ Xₗ_η - Xₗ Xₘ_η)_ζ ]
    @views contravariant_vectors[n, 1, i, j, :, element] -= 0.5 * basis.derivative_matrix * (
        node_coordinates[m, i, j, :, element] .* jacobian_matrix[l, 2, i, j, :, element] .-
        node_coordinates[l, i, j, :, element] .* jacobian_matrix[m, 2, i, j, :, element])

    # Calc only the second summand -0.5 * (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ of
    # Ja²ₙ = 0.5 * [ (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_ζ - (Xₘ Xₗ_ζ - Xₗ Xₘ_ζ)_ξ ]
    @views contravariant_vectors[n, 2, :, i, j, element] -= 0.5 * basis.derivative_matrix * (
        node_coordinates[m, :, i, j, element] .* jacobian_matrix[l, 3, :, i, j, element] .-
        node_coordinates[l, :, i, j, element] .* jacobian_matrix[m, 3, :, i, j, element])

    # Calc only the second summand -0.5 * (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η of
    # Ja³ₙ = 0.5 * [ (Xₘ Xₗ_η - Xₗ Xₘ_η)_ξ - (Xₘ Xₗ_ξ - Xₗ Xₘ_ξ)_η ]
    @views contravariant_vectors[n, 3, i, :, j, element] -= 0.5 * basis.derivative_matrix * (
        node_coordinates[m, i, :, j, element] .* jacobian_matrix[l, 1, i, :, j, element] .-
        node_coordinates[l, i, :, j, element] .* jacobian_matrix[m, 1, i, :, j, element])
  end

  return contravariant_vectors
end


# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 4}, element, jacobian_matrix, basis)
  for k in eachnode(basis), j in eachnode(basis), i in eachnode(basis)
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

    if isperiodic(mesh, 1)
      # Periodic boundary
      left_neighbors[1, linear_indices[1, cell_y, cell_z]] = linear_indices[end, cell_y, cell_z]
    else
      left_neighbors[1, linear_indices[1, cell_y, cell_z]] = 0
    end
  end

  # Neighbors in y-direction
  for cell_z in 1:size(mesh, 3), cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y, cell_z]
      left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1, cell_z]
    end

    if isperiodic(mesh, 2)
      # Periodic boundary
      left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = linear_indices[cell_x, end, cell_z]
    else
      left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = 0
    end
  end

  # Neighbors in z-direction
  for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_z in 2:size(mesh, 3)
      element = linear_indices[cell_x, cell_y, cell_z]
      left_neighbors[3, element] = linear_indices[cell_x, cell_y, cell_z - 1]
    end

    if isperiodic(mesh, 3)
      # Periodic boundary
      left_neighbors[3, linear_indices[cell_x, cell_y, 1]] = linear_indices[cell_x, cell_y, end]
    else
      left_neighbors[3, linear_indices[cell_x, cell_y, 1]] = 0
    end
  end

  return left_neighbors
end
