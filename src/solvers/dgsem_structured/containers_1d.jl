# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize data structures in element container
function init_elements!(elements, mesh::StructuredMesh{1}, basis::LobattoLegendreBasis)
    @unpack node_coordinates, left_neighbors,
    jacobian_matrix, contravariant_vectors, inverse_jacobian = elements

    # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
    for cell_x in 1:size(mesh, 1)
        calc_node_coordinates!(node_coordinates, cell_x, mesh.mapping, mesh, basis)

        calc_jacobian_matrix!(jacobian_matrix, cell_x, node_coordinates, basis)

        calc_inverse_jacobian!(inverse_jacobian, cell_x, jacobian_matrix)
    end

    # Contravariant vectors don't make sense in 1D, they would be identical to inverse_jacobian
    fill!(contravariant_vectors, NaN)

    initialize_left_neighbor_connectivity!(left_neighbors, mesh)

    return nothing
end

# Calculate physical coordinates to which every node of the reference element is mapped
# `mesh.mapping` is passed as an additional argument for type stability (function barrier)
function calc_node_coordinates!(node_coordinates, cell_x, mapping,
                                mesh::StructuredMesh{1},
                                basis::LobattoLegendreBasis)
    @unpack nodes = basis

    # Get cell length in reference mesh
    dx = 2 / size(mesh, 1)

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2

    for i in eachnode(basis)
        # node_coordinates are the mapped reference node_coordinates
        node_coordinates[1, i, cell_x] = mapping(cell_x_offset + dx / 2 * nodes[i])[1]
    end
end

# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix!(jacobian_matrix, element,
                               node_coordinates::AbstractArray{<:Any, 3},
                               basis::LobattoLegendreBasis)
    @views mul!(jacobian_matrix[1, 1, :, element], basis.derivative_matrix,
                node_coordinates[1, :, element]) # x_ξ

    return jacobian_matrix
end

# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 2}, element,
                                jacobian_matrix)
    @views inverse_jacobian[:, element] .= inv.(jacobian_matrix[1, 1, :, element])

    return inverse_jacobian
end

# Save id of left neighbor of every element
function initialize_left_neighbor_connectivity!(left_neighbors, mesh::StructuredMesh{1})
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
        left_neighbors[1, 1] = 0
    end

    return left_neighbors
end
end # @muladd
