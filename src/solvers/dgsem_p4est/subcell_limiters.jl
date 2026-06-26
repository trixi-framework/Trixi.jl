# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# Auxiliary routine `get_boundary_outer_state` for non-periodic domains

"""
    get_boundary_outer_state(u_inner, t,
                             boundary_condition::BoundaryConditionDirichlet,
                             normal_direction
                             mesh, equations, dg, cache, indices...)
For subcell limiting, the calculation of local bounds for non-periodic domains requires the boundary
outer state. This function returns the boundary value  for [`BoundaryConditionDirichlet`](@ref) at
time `t` and for node with spatial indices `indices` at the boundary with `normal_direction`.

Should be used together with [`P4estMesh`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
@inline function get_boundary_outer_state(u_inner, t,
                                          boundary_condition::BoundaryConditionDirichlet,
                                          normal_direction,
                                          mesh::P4estMesh,
                                          equations, dg, cache, indices...)
    (; node_coordinates) = cache.elements

    x = get_node_coords(node_coordinates, equations, dg, indices...)
    u_outer = boundary_condition.boundary_value_function(x, t, equations)

    return u_outer
end

@inline function average_mortar_limiting_factor(limiting_factor, mesh::P4estMesh{2},
                                                dg, cache)
    (; weights) = dg.basis
    (; jacobian_matrix) = cache.elements
    (; neighbor_ids, node_indices) = cache.mortars

    avg_type = promote_type(eltype(limiting_factor), eltype(jacobian_matrix))
    weighted_sum = zero(avg_type)
    total_weight = zero(avg_type)
    index_range = eachnode(dg)

    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        large_indices = node_indices[2, mortar]

        i_large, i_large_step = index_to_start_step_2d(large_indices[1], index_range)
        j_large, j_large_step = index_to_start_step_2d(large_indices[2], index_range)
        tangent_direction = iszero(i_large_step) ? 2 : 1

        mortar_size = zero(avg_type)
        for node in eachnode(dg)
            tangent_x = jacobian_matrix[1, tangent_direction, i_large, j_large,
                                        large_element]
            tangent_y = jacobian_matrix[2, tangent_direction, i_large, j_large,
                                        large_element]
            mortar_size += weights[node] * sqrt(tangent_x^2 + tangent_y^2)

            i_large += i_large_step
            j_large += j_large_step
        end

        weighted_sum += limiting_factor[mortar] * mortar_size
        total_weight += mortar_size
    end

    return weighted_sum / total_weight
end
end # @muladd
