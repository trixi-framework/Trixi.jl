# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
function calc_node_coordinates!(node_coordinates,
                                mesh::T8codeMesh{3, RealT},
                                nodes::AbstractVector) where {RealT <: Real}
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(mesh),
                       StaticInt(3), static_length(nodes), static_length(mesh.nodes),
                       static_length(mesh.nodes))
    matrix1 = StrideArray(undef, real(mesh),
                          static_length(nodes), static_length(mesh.nodes))
    matrix2 = similar(matrix1)
    matrix3 = similar(matrix1)
    baryweights_in = barycentric_weights(mesh.nodes)

    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)

    scheme = t8_forest_get_scheme(mesh.forest)

    current_index = 0
    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(mesh.forest, itree)
        num_elements_in_tree = t8_forest_get_tree_num_leaf_elements(mesh.forest, itree)
        global_itree = t8_forest_global_tree_id(mesh.forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_leaf_element_in_tree(mesh.forest, itree, ielement)
            element_level = t8_element_get_level(scheme, tree_class, element)

            # Note, `t8_hex_len` is encoded as an integer (Morton encoding) in
            # relation to `t8_hex_root_len`. This line transforms the
            # "integer" length to a float in relation to the unit interval [0,1].
            element_length = t8_hex_len(element_level) / t8_hex_root_len

            element_coords = Vector{RealT}(undef, 3)
            t8_element_get_vertex_reference_coords(scheme, tree_class, element, 0,
                                                   pointer(element_coords))

            nodes_out_x = (2 *
                           (element_length * 0.5f0 * (nodes .+ 1) .+ element_coords[1]) .-
                           1)
            nodes_out_y = (2 *
                           (element_length * 0.5f0 * (nodes .+ 1) .+ element_coords[2]) .-
                           1)
            nodes_out_z = (2 *
                           (element_length * 0.5f0 * (nodes .+ 1) .+ element_coords[3]) .-
                           1)

            polynomial_interpolation_matrix!(matrix1, mesh.nodes, nodes_out_x,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix2, mesh.nodes, nodes_out_y,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix3, mesh.nodes, nodes_out_z,
                                             baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, :,
                                         current_index += 1),
                                    matrix1, matrix2, matrix3,
                                    view(mesh.tree_node_coordinates, :, :, :, :,
                                         global_itree + 1), tmp1)
        end
    end

    return node_coordinates
end

# This routine was originally copied and adapted from
# `src/dgsem_p4est/containers_3d.jl`: `orientation_to_indices_p4est`.
# It simplified a lot with t8code v4. However, it relies on the ordering given by t8code's
# space filling curve, and could become a point of failure.
# xref: https://github.com/DLR-AMR/t8code/issues/2190
@inline function init_mortar_neighbor_ids!(mortars::P4estMortarContainer{3}, my_face,
                                           other_face, orientation, neighbor_ielements,
                                           mortar_id)
    # In the following illustrations, the face corner numbering of `p4est` is shown.
    # ξ and η are the local coordinates of the respective face.
    # We're looking at both faces from the same side of the interface, so that "other side"
    # (in the illustrations on the left) has right-handed coordinates.
    #
    # Corner 0 of other side matches corner 0 of my side
    #   2┌──────┐3   2┌──────┐3
    #    │      │     │      │
    #    │      │     │      │
    #   0└──────┘1   0└──────┘1
    #     η            η
    #     ↑            ↑
    #     │            │
    #     └───> ξ      └───> ξ

    mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[1] + 1
    mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[2] + 1
    mortars.neighbor_ids[3, mortar_id] = neighbor_ielements[3] + 1
    mortars.neighbor_ids[4, mortar_id] = neighbor_ielements[4] + 1
end
end # @muladd
