# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize data structures in element container
function init_elements!(elements, mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                        basis::LobattoLegendreBasis)
    @unpack node_coordinates, jacobian_matrix,
    contravariant_vectors, inverse_jacobian = elements

    calc_node_coordinates!(node_coordinates, mesh, basis)

    if size(node_coordinates,1) == 3 # 2D cubed sphere
        for element in 1:ncells(mesh)
            calc_jacobian_matrix_cubed_sphere!(jacobian_matrix, element, node_coordinates, basis)

            #calc_contravariant_vectors_cubed_sphere!(contravariant_vectors, element, jacobian_matrix)

            calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

            #calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
            calc_inverse_jacobian_cubed_sphere!(inverse_jacobian, element, jacobian_matrix, basis)
        end
    else
        for element in 1:ncells(mesh)
            calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

            calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

            calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
        end
    end

    

    return nothing
end

"""
    calc_jacobian_matrix_cubed_sphere!(jacobian_matrix, element,
                                       node_coordinates::AbstractArray{<:Any, 4},
                                       basis::LobattoLegendreBasis)
Compute Jacobian matrix for cubed sphere. We compute the Jacobian components in ξ and η
direction as usual, and then compute third component (dx⃗/dζ) analytically as (dx⃗/dr). See, e.g.

*   Giraldo, F. X., Hesthaven, J. S., & Warburton, T. (2002). Nodal high-order discontinuous 
    Galerkin methods for the spherical shallow water equations. Journal of Computational Physics, 181(2), 499-525.
"""
function calc_jacobian_matrix_cubed_sphere!(jacobian_matrix, element,
                                            node_coordinates::AbstractArray{<:Any, 4},
                                            basis::LobattoLegendreBasis)
    # Compute 2D Jacobian matrix as usual
    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    # Compute third component (dx⃗/dζ) analytically as (dx⃗/dr). See, e.g.
    for j in indices((jacobian_matrix, node_coordinates), (4, 3)),
        i in indices((jacobian_matrix, node_coordinates), (3, 2))

        x = node_coordinates[1, i, j, element]
        y = node_coordinates[2, i, j, element]
        z = node_coordinates[3, i, j, element]
        theta = acos(z / sqrt(x^2 + y^2 + z^2))
        phi = sign(y) * acos(x / sqrt(x^2 + y^2))

        jacobian_matrix[1, 3, i, j, element] = sin(theta) * cos(phi)
        jacobian_matrix[2, 3, i, j, element] = sin(theta) * sin(phi)
        jacobian_matrix[3, 3, i, j, element] = cos(theta)
        #= jacobian_matrix[1, 3, i, j, element] = theta
        jacobian_matrix[2, 3, i, j, element] = phi
        jacobian_matrix[3, 3, i, j, element] = sqrt(sum(node_coordinates[:, i, j, element].^2)) =#
    end
end

# Calculate inverse Jacobian for the cubed sphere in 2D (determinant of Jacobian matrix of the mapping) in each node
function calc_inverse_jacobian_cubed_sphere!(inverse_jacobian::AbstractArray{<:Any, 3}, element,
                                             jacobian_matrix, basis)
    @turbo for j in eachnode(basis), i in eachnode(basis)
        # Calculate Determinant by using Sarrus formula (about 100 times faster than LinearAlgebra.det())
        inverse_jacobian[i, j, element] = inv(jacobian_matrix[1, 1, i, j,
                                                                 element] *
                                                 jacobian_matrix[2, 2, i, j,
                                                                 element] *
                                                 jacobian_matrix[3, 3, i, j, element] +
                                                 jacobian_matrix[1, 2, i, j,
                                                                 element] *
                                                 jacobian_matrix[2, 3, i, j,
                                                                 element] *
                                                 jacobian_matrix[3, 1, i, j, element] +
                                                 jacobian_matrix[1, 3, i, j,
                                                                 element] *
                                                 jacobian_matrix[2, 1, i, j,
                                                                 element] *
                                                 jacobian_matrix[3, 2, i, j, element] -
                                                 jacobian_matrix[3, 1, i, j,
                                                                 element] *
                                                 jacobian_matrix[2, 2, i, j,
                                                                 element] *
                                                 jacobian_matrix[1, 3, i, j, element] -
                                                 jacobian_matrix[3, 2, i, j,
                                                                 element] *
                                                 jacobian_matrix[2, 3, i, j,
                                                                 element] *
                                                 jacobian_matrix[1, 1, i, j, element] -
                                                 jacobian_matrix[3, 3, i, j,
                                                                 element] *
                                                 jacobian_matrix[2, 1, i, j,
                                                                 element] *
                                                 jacobian_matrix[1, 2, i, j, element])
    end

    return inverse_jacobian
end

# Interpolate tree_node_coordinates to each quadrant at the nodes of the specified basis
function calc_node_coordinates!(node_coordinates,
                                mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                basis::LobattoLegendreBasis)
    # Hanging nodes will cause holes in the mesh if its polydeg is higher
    # than the polydeg of the solver.
    @assert length(basis.nodes)>=length(mesh.nodes) "The solver can't have a lower polydeg than the mesh"

    calc_node_coordinates!(node_coordinates, mesh, basis.nodes)
end

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{2},
                                nodes::AbstractVector)
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(mesh),
                       StaticInt(size(mesh.tree_node_coordinates,1)), static_length(nodes), static_length(mesh.nodes))
    matrix1 = StrideArray(undef, real(mesh),
                          static_length(nodes), static_length(mesh.nodes))
    matrix2 = similar(matrix1)
    baryweights_in = barycentric_weights(mesh.nodes)

    # Macros from `p4est`
    p4est_root_len = 1 << P4EST_MAXLEVEL
    p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

    trees = unsafe_wrap_sc(p4est_tree_t, mesh.p4est.trees)

    for tree in eachindex(trees)
        offset = trees[tree].quadrants_offset
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree].quadrants)

        for i in eachindex(quadrants)
            element = offset + i
            quad = quadrants[i]

            quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

            nodes_out_x = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.x / p4est_root_len) .- 1
            nodes_out_y = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.y / p4est_root_len) .- 1
            polynomial_interpolation_matrix!(matrix1, mesh.nodes, nodes_out_x,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix2, mesh.nodes, nodes_out_y,
                                             baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, element),
                                    matrix1, matrix2,
                                    view(mesh.tree_node_coordinates, :, :, :, tree),
                                    tmp1)
        end
    end

    return node_coordinates
end

# Initialize node_indices of interface container
@inline function init_interface_node_indices!(interfaces::P4estInterfaceContainer{2},
                                              faces, orientation, interface_id)
    # Iterate over primary and secondary element
    for side in 1:2
        # Align interface in positive coordinate direction of primary element.
        # For orientation == 1, the secondary element needs to be indexed backwards
        # relative to the interface.
        if side == 1 || orientation == 0
            # Forward indexing
            i = :i_forward
        else
            # Backward indexing
            i = :i_backward
        end

        if faces[side] == 0
            # Index face in negative x-direction
            interfaces.node_indices[side, interface_id] = (:begin, i)
        elseif faces[side] == 1
            # Index face in positive x-direction
            interfaces.node_indices[side, interface_id] = (:end, i)
        elseif faces[side] == 2
            # Index face in negative y-direction
            interfaces.node_indices[side, interface_id] = (i, :begin)
        else # faces[side] == 3
            # Index face in positive y-direction
            interfaces.node_indices[side, interface_id] = (i, :end)
        end
    end

    return interfaces
end

# Initialize node_indices of boundary container
@inline function init_boundary_node_indices!(boundaries::P4estBoundaryContainer{2},
                                             face, boundary_id)
    if face == 0
        # Index face in negative x-direction
        boundaries.node_indices[boundary_id] = (:begin, :i_forward)
    elseif face == 1
        # Index face in positive x-direction
        boundaries.node_indices[boundary_id] = (:end, :i_forward)
    elseif face == 2
        # Index face in negative y-direction
        boundaries.node_indices[boundary_id] = (:i_forward, :begin)
    else # face == 3
        # Index face in positive y-direction
        boundaries.node_indices[boundary_id] = (:i_forward, :end)
    end

    return boundaries
end

# Initialize node_indices of mortar container
# faces[1] is expected to be the face of the small side.
@inline function init_mortar_node_indices!(mortars, faces, orientation, mortar_id)
    for side in 1:2
        # Align mortar in positive coordinate direction of small side.
        # For orientation == 1, the large side needs to be indexed backwards
        # relative to the mortar.
        if side == 1 || orientation == 0
            # Forward indexing for small side or orientation == 0
            i = :i_forward
        else
            # Backward indexing for large side with reversed orientation
            i = :i_backward
        end

        if faces[side] == 0
            # Index face in negative x-direction
            mortars.node_indices[side, mortar_id] = (:begin, i)
        elseif faces[side] == 1
            # Index face in positive x-direction
            mortars.node_indices[side, mortar_id] = (:end, i)
        elseif faces[side] == 2
            # Index face in negative y-direction
            mortars.node_indices[side, mortar_id] = (i, :begin)
        else # faces[side] == 3
            # Index face in positive y-direction
            mortars.node_indices[side, mortar_id] = (i, :end)
        end
    end

    return mortars
end
end # @muladd
