# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Interpolate tree_node_coordinates to each quadrant at the specified nodes.
function calc_node_coordinates!(node_coordinates,
                                mesh::T8codeMesh{2},
                                nodes::AbstractVector)
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(mesh),
                       StaticInt(2), static_length(nodes), static_length(mesh.nodes))
    matrix1 = StrideArray(undef, real(mesh),
                          static_length(nodes), static_length(mesh.nodes))
    matrix2 = similar(matrix1)
    baryweights_in = barycentric_weights(mesh.nodes)

    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)

    current_index = 0
    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(mesh.forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(mesh.forest, tree_class)
        num_elements_in_tree = t8_forest_get_tree_num_elements(mesh.forest, itree)
        global_itree = t8_forest_global_tree_id(mesh.forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(mesh.forest, itree, ielement)
            element_level = t8_element_level(eclass_scheme, element)

            # Note, `t8_quad_len` is encoded as an integer (Morton encoding) in
            # relation to `t8_quad_root_len`. This line transforms the
            # "integer" length to a float in relation to the unit interval [0,1].
            element_length = t8_quad_len(element_level) / t8_quad_root_len

            element_coords = Array{Float64}(undef, 3)
            t8_element_vertex_reference_coords(eclass_scheme, element, 0,
                                               pointer(element_coords))

            nodes_out_x = 2 *
                          (element_length * 1 / 2 * (nodes .+ 1) .+ element_coords[1]) .-
                          1
            nodes_out_y = 2 *
                          (element_length * 1 / 2 * (nodes .+ 1) .+ element_coords[2]) .-
                          1

            polynomial_interpolation_matrix!(matrix1, mesh.nodes, nodes_out_x,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix2, mesh.nodes, nodes_out_y,
                                             baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, current_index += 1),
                                    matrix1, matrix2,
                                    view(mesh.tree_node_coordinates, :, :, :,
                                         global_itree + 1),
                                    tmp1)
        end
    end

    return node_coordinates
end

function init_mortar_neighbor_ids!(mortars::P4estMortarContainer{2}, my_face,
                                   other_face, orientation, neighbor_ielements,
                                   mortar_id)
    if orientation == 0
        mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[1] + 1
        mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[2] + 1
    else
        mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[2] + 1
        mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[1] + 1
    end
end
end # @muladd
