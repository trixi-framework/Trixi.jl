# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function analyze_coefficient(mesh::TreeMesh2D, equations, dg, cache,
                             limiter::SubcellLimiterIDP)
    @unpack weights = dg.basis
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = zero(eltype(alpha))
    total_volume = zero(eltype(alpha))
    for element in eachelement(dg, cache)
        jacobian = inv(cache.elements.inverse_jacobian[element])
        for j in eachnode(dg), i in eachnode(dg)
            alpha_avg += jacobian * weights[i] * weights[j] * alpha[i, j, element]
            total_volume += jacobian * weights[i] * weights[j]
        end
    end

    return alpha_avg / total_volume
end

function analyze_coefficient(mesh::Union{StructuredMesh{2}, P4estMesh{2}},
                             equations, dg, cache,
                             limiter::SubcellLimiterIDP)
    @unpack weights = dg.basis
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = zero(eltype(alpha))
    total_volume = zero(eltype(alpha))
    for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            jacobian = inv(cache.elements.inverse_jacobian[i, j, element])
            alpha_avg += jacobian * weights[i] * weights[j] * alpha[i, j, element]
            total_volume += jacobian * weights[i] * weights[j]
        end
    end

    return alpha_avg / total_volume
end

@inline function average_mortar_limiting_factor(limiting_factor, mesh::TreeMesh2D,
                                                dg, cache)
    (; neighbor_ids, large_sides, orientations) = cache.mortars
    (; node_coordinates) = cache.elements

    avg_type = promote_type(eltype(limiting_factor), eltype(node_coordinates))
    weighted_sum = zero(avg_type)
    total_weight = zero(avg_type)
    n_nodes = nnodes(dg)

    for mortar in eachindex(limiting_factor)
        large_element = neighbor_ids[3, mortar]

        if large_sides[mortar] == 1 # small elements on right side
            if orientations[mortar] == 1
                start_indices = (n_nodes, 1)
                end_indices = (n_nodes, n_nodes)
            else
                start_indices = (1, n_nodes)
                end_indices = (n_nodes, n_nodes)
            end
        else # large_sides[mortar] == 2, small elements on left side
            if orientations[mortar] == 1
                start_indices = (1, 1)
                end_indices = (1, n_nodes)
            else
                start_indices = (1, 1)
                end_indices = (n_nodes, 1)
            end
        end

        dx = (node_coordinates[1, end_indices..., large_element] -
              node_coordinates[1, start_indices..., large_element])
        dy = (node_coordinates[2, end_indices..., large_element] -
              node_coordinates[2, start_indices..., large_element])
        mortar_size = sqrt(dx^2 + dy^2)

        weighted_sum += limiting_factor[mortar] * mortar_size
        total_weight += mortar_size
    end

    return weighted_sum / total_weight
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
