# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize data structures in element container
function init_elements!(elements, mesh::Union{StructuredMesh{2}, StructuredMeshView{2}},
                        basis::LobattoLegendreBasis)
    @unpack node_coordinates, left_neighbors,
    jacobian_matrix, contravariant_vectors, inverse_jacobian = elements

    linear_indices = LinearIndices(size(mesh))

    # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
    for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
        element = linear_indices[cell_x, cell_y]

        calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, mesh.mapping,
                               mesh, basis)

        calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

        calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

        calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
    end

    initialize_left_neighbor_connectivity!(left_neighbors, mesh, linear_indices)

    return nothing
end

# Calculate physical coordinates to which every node of the reference element is mapped
# `mesh.mapping` is passed as an additional argument for type stability (function barrier)
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, mapping,
                                mesh::StructuredMesh{2},
                                basis::LobattoLegendreBasis)
    @unpack nodes = basis

    # Get cell length in reference mesh
    dx = 2 / size(mesh, 1)
    dy = 2 / size(mesh, 2)

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x - 1) * dx + dx / 2
    cell_y_offset = -1 + (cell_y - 1) * dy + dy / 2

    for j in eachnode(basis), i in eachnode(basis)
        # node_coordinates are the mapped reference node_coordinates
        node_coordinates[:, i, j, element] .= mapping(cell_x_offset + dx / 2 * nodes[i],
                                                      cell_y_offset + dy / 2 * nodes[j])
    end

    return nothing
end

# Calculate Jacobian matrix of the mapping from the reference element to the element in the physical domain
function calc_jacobian_matrix!(jacobian_matrix, element,
                               node_coordinates::AbstractArray{<:Any, 4},
                               basis::LobattoLegendreBasis)
    @unpack derivative_matrix = basis

    # The code below is equivalent to the following matrix multiplications, which
    # seem to end up calling generic linear algebra code from Julia. Thus, the
    # optimized code below using `@turbo` is much faster.
    # jacobian_matrix[1, 1, :, :, element] = derivative_matrix * node_coordinates[1, :, :, element]  # x_ξ
    # jacobian_matrix[2, 1, :, :, element] = derivative_matrix * node_coordinates[2, :, :, element]  # y_ξ
    # jacobian_matrix[1, 2, :, :, element] = node_coordinates[1, :, :, element] * derivative_matrix' # x_η
    # jacobian_matrix[2, 2, :, :, element] = node_coordinates[2, :, :, element] * derivative_matrix' # y_η

    # x_ξ, y_ξ
    @turbo for xy in indices((jacobian_matrix, node_coordinates), (1, 1))
        for j in indices((jacobian_matrix, node_coordinates), (4, 3)),
            i in indices((jacobian_matrix, derivative_matrix), (3, 1))

            result = zero(eltype(jacobian_matrix))
            for ii in indices((node_coordinates, derivative_matrix), (2, 2))
                result += derivative_matrix[i, ii] *
                          node_coordinates[xy, ii, j, element]
            end
            jacobian_matrix[xy, 1, i, j, element] = result
        end
    end

    # x_η, y_η
    @turbo for xy in indices((jacobian_matrix, node_coordinates), (1, 1))
        for j in indices((jacobian_matrix, derivative_matrix), (4, 1)),
            i in indices((jacobian_matrix, node_coordinates), (3, 2))

            result = zero(eltype(jacobian_matrix))
            for jj in indices((node_coordinates, derivative_matrix), (3, 2))
                result += derivative_matrix[j, jj] *
                          node_coordinates[xy, i, jj, element]
            end
            jacobian_matrix[xy, 2, i, j, element] = result
        end
    end

    return jacobian_matrix
end

# Calculate contravariant vectors, multiplied by the Jacobian determinant J of the transformation mapping.
# Those are called Ja^i in Kopriva's blue book.
function calc_contravariant_vectors!(contravariant_vectors::AbstractArray{<:Any, 5},
                                     element, jacobian_matrix)
    # The code below is equivalent to the following using broadcasting but much faster.
    # # First contravariant vector Ja^1
    # contravariant_vectors[1, 1, :, :, element] =  jacobian_matrix[2, 2, :, :, element]
    # contravariant_vectors[2, 1, :, :, element] = -jacobian_matrix[1, 2, :, :, element]
    # # Second contravariant vector Ja^2
    # contravariant_vectors[1, 2, :, :, element] = -jacobian_matrix[2, 1, :, :, element]
    # contravariant_vectors[2, 2, :, :, element] =  jacobian_matrix[1, 1, :, :, element]

    @turbo for j in indices((contravariant_vectors, jacobian_matrix), (4, 4)),
               i in indices((contravariant_vectors, jacobian_matrix), (3, 3))
        # First contravariant vector Ja^1
        contravariant_vectors[1, 1, i, j, element] = jacobian_matrix[2, 2, i, j,
                                                                     element]
        contravariant_vectors[2, 1, i, j, element] = -jacobian_matrix[1, 2, i, j,
                                                                      element]

        # Second contravariant vector Ja^2
        contravariant_vectors[1, 2, i, j, element] = -jacobian_matrix[2, 1, i, j,
                                                                      element]
        contravariant_vectors[2, 2, i, j, element] = jacobian_matrix[1, 1, i, j,
                                                                     element]
    end

    return contravariant_vectors
end

# Calculate inverse Jacobian (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian!(inverse_jacobian::AbstractArray{<:Any, 3}, element,
                                jacobian_matrix)
    # The code below is equivalent to the following high-level code but much faster.
    # inverse_jacobian[i, j, element] = inv(det(jacobian_matrix[:, :, i, j, element])

    @turbo for j in indices((inverse_jacobian, jacobian_matrix), (2, 4)),
               i in indices((inverse_jacobian, jacobian_matrix), (1, 3))

        inverse_jacobian[i, j, element] = inv(jacobian_matrix[1, 1, i, j, element] *
                                              jacobian_matrix[2, 2, i, j, element] -
                                              jacobian_matrix[1, 2, i, j, element] *
                                              jacobian_matrix[2, 1, i, j, element])
    end

    return inverse_jacobian
end

# Save id of left neighbor of every element
function initialize_left_neighbor_connectivity!(left_neighbors,
                                                mesh::Union{StructuredMesh{2},
                                                            StructuredMeshView{2}},
                                                linear_indices)
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

# Fixed size, i.e., not AMR ready. Sufficient for `StructuredMesh` and `UnstructuredMesh2D`
struct FixedNormalVectorContainer2D{RealT <: Real} <: AbstractNormalVectorContainer
    # For normal vectors computed from first contravariant vectors
    normal_vectors_1::Array{RealT, 4}
    # For normal vectors computed from second contravariant vectors
    normal_vectors_2::Array{RealT, 4}
end

# Compute the normal vectors for freestream-preserving FV method on curvilinear subcells, see
# equation (B.53) in:
# - Hennemann, Rueda-Ramírez, Hindenlang, Gassner (2020)
#   A provably entropy stable subcell shock capturing approach for high order split form DG for the compressible Euler equations
#   [arXiv: 2008.12044v2](https://arxiv.org/pdf/2008.12044)
function calc_normalvectors_subcell_fv!(normal_vectors_1, normal_vectors_2,
                                        mesh::Union{StructuredMesh{2},
                                                    UnstructuredMesh2D,
                                                    P4estMesh{2}, T8codeMesh{2}},
                                        dg, cache_containers)
    @unpack contravariant_vectors = cache_containers.elements
    @unpack weights, derivative_matrix = dg.basis

    for element in eachelement(dg, cache_containers)
        for i in eachnode(dg)
            # j = 1
            # Optimize indexing: j to second position, i to third
            normal_vectors_1[:, 1, i, element] = get_contravariant_vector(1,
                                                                          contravariant_vectors,
                                                                          1, i, element)
            normal_vectors_2[:, 1, i, element] = get_contravariant_vector(2,
                                                                          contravariant_vectors,
                                                                          i, 1, element)
            for j in 2:nnodes(dg)
                normal_vectors_1[:, j, i, element] = normal_vectors_1[:, j - 1, i,
                                                                      element]
                normal_vectors_2[:, j, i, element] = normal_vectors_2[:, j - 1, i,
                                                                      element]
                for m in eachnode(dg)
                    wD_jm = weights[j - 1] * derivative_matrix[j - 1, m]
                    normal_vectors_1[:, j, i, element] += wD_jm *
                                                          get_contravariant_vector(1,
                                                                                   contravariant_vectors,
                                                                                   m, i,
                                                                                   element)

                    normal_vectors_2[:, j, i, element] += wD_jm *
                                                          get_contravariant_vector(2,
                                                                                   contravariant_vectors,
                                                                                   i, m,
                                                                                   element)
                end
            end
        end
    end

    return (normal_vectors_1, normal_vectors_2)
end

function FixedNormalVectorContainer2D(mesh::Union{StructuredMesh{2},
                                                  UnstructuredMesh2D},
                                      dg, cache_containers)
    @unpack contravariant_vectors = cache_containers.elements
    RealT = eltype(contravariant_vectors)
    n_elements = nelements(dg, cache_containers)
    n_nodes = nnodes(dg.basis)

    # For first contravariant vector
    normal_vectors_1 = Array{RealT, 4}(undef, 2, n_nodes, n_nodes,
                                       n_elements)
    # For second contravariant vector
    normal_vectors_2 = Array{RealT, 4}(undef, 2, n_nodes, n_nodes,
                                       n_elements)

    calc_normalvectors_subcell_fv!(normal_vectors_1, normal_vectors_2,
                                   mesh, dg, cache_containers)

    return FixedNormalVectorContainer2D{RealT}(normal_vectors_1, normal_vectors_2)
end

# Similar to `get_contravariant_vector`
@inline function get_normal_vector(normal_vectors, indices...)
    return SVector(ntuple(@inline(dim->normal_vectors[dim, indices...]),
                          Val(ndims(normal_vectors) - 2)))
end
end # @muladd
