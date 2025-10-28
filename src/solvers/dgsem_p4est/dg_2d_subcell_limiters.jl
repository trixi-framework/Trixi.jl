# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prolong2mortars!(cache, u, mesh::P4estMesh{2}, equations,
                          mortar_idp::LobattoLegendreMortarIDP, dg::DGSEM)
    prolong2mortars!(cache, u, mesh, equations, mortar_idp.mortar_l2, dg)

    (; neighbor_ids, node_indices, u_large) = cache.mortars
    index_range = eachnode(dg)

    # The data of both small elements were already copied to the mortar cache
    @threaded for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]

        # Copy solutions data from large element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                u_large[v, i, mortar] = u[v, i_large, j_large, large_element]
            end
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

function calc_mortar_flux_low_order!(surface_flux_values,
                                     mesh::P4estMesh{2},
                                     nonconservative_terms::False, equations,
                                     mortar_idp::LobattoLegendreMortarIDP,
                                     surface_integral, dg::DG, cache)
    @unpack surface_flux = surface_integral
    @unpack neighbor_ids, node_indices, u, u_large = cache.mortars
    @unpack contravariant_vectors = cache.elements
    (; mortar_weights, mortar_weights_sums) = mortar_idp
    index_range = eachnode(dg)

    @assert mortar_idp.local_factor "local factor should be active"

    @threaded for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        large_direction = indices2direction(large_indices)
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        surface_flux_values[:, :, small_direction, lower_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, small_direction, upper_element] .= zero(eltype(surface_flux_values))
        surface_flux_values[:, :, large_direction, large_element] .= zero(eltype(surface_flux_values))

        i_small = i_small_start
        j_small = j_small_start
        # Calculate fluxes
        for i in eachnode(dg)
            i_mortar = iszero(i_small_step) ? j_small : i_small

            # Get the normal direction on the small element.
            # Note, contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction_lower = get_normal_direction(small_direction,
                                                          contravariant_vectors,
                                                          i_small, j_small,
                                                          lower_element)
            normal_direction_upper = get_normal_direction(small_direction,
                                                          contravariant_vectors,
                                                          i_small, j_small,
                                                          upper_element)

            # Lower element
            u_lower_local, _ = get_surface_node_vars(u, equations, dg, 1, i, mortar)

            i_large = i_large_start
            j_large = j_large_start
            for j in eachnode(dg)
                j_mortar = iszero(i_large_step) ? j_large : i_large

                factor = mortar_weights[j_mortar, i_mortar, 1]
                if !isapprox(factor, zero(typeof(factor)))
                    u_large_local = get_node_vars(u_large, equations, dg, j, mortar)

                    normal_direction_large = get_normal_direction(large_direction,
                                                                  contravariant_vectors,
                                                                  i_large, j_large,
                                                                  large_element)
                    # TODO: What do I do with the normal_directions? Doesn't make sense right now. See theory.

                    flux = surface_flux(u_lower_local, u_large_local,
                                        normal_direction_lower, equations)

                    # Lower element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               factor /
                                               mortar_weights_sums[i_mortar, 1],
                                               flux, equations, dg,
                                               i, small_direction, lower_element)
                    # Large element
                    # The flux is calculated in the outward direction of the small elements,
                    # so the sign must be switched to get the flux in outward direction
                    # of the large element.
                    # The contravariant vectors of the large element (and therefore the normal
                    # vectors of the large element as well) are twice as large as the
                    # contravariant vectors of the small elements. Therefore, the flux needs
                    # to be scaled by a factor of 2 to obtain the flux of the large element.
                    multiply_add_to_node_vars!(surface_flux_values,
                                               -2 * factor /
                                               mortar_weights_sums[j_mortar, 2],
                                               flux, equations, dg,
                                               j, large_direction, large_element)
                end
                i_large += i_large_step
                j_large += j_large_step
            end

            # Upper element
            u_upper_local, _ = get_surface_node_vars(u, equations, dg, 2, i, mortar)

            i_large = i_large_start
            j_large = j_large_start
            for j in eachnode(dg)
                j_mortar = iszero(i_large_step) ? j_large : i_large

                factor = mortar_weights[j_mortar, i_mortar, 2]
                if !isapprox(factor, zero(typeof(factor)))
                    u_large_local = get_node_vars(u_large, equations, dg, j, mortar)

                    normal_direction_large = get_normal_direction(large_direction,
                                                                  contravariant_vectors,
                                                                  i_large, j_large,
                                                                  large_element)

                    flux = surface_flux(u_upper_local, u_large_local,
                                        normal_direction_upper, equations)

                    # Upper element
                    multiply_add_to_node_vars!(surface_flux_values,
                                               factor /
                                               mortar_weights_sums[i_mortar, 1],
                                               flux, equations, dg,
                                               i, small_direction, upper_element)
                    # Large element
                    # The flux is calculated in the outward direction of the small elements,
                    # so the sign must be switched to get the flux in outward direction
                    # of the large element.
                    # The contravariant vectors of the large element (and therefore the normal
                    # vectors of the large element as well) are twice as large as the
                    # contravariant vectors of the small elements. Therefore, the flux needs
                    # to be scaled by a factor of 2 to obtain the flux of the large element.
                    multiply_add_to_node_vars!(surface_flux_values,
                                               -2 * factor /
                                               mortar_weights_sums[j_mortar, 2],
                                               flux, equations, dg,
                                               j, large_direction, large_element)
                end
                i_large += i_large_step
                j_large += j_large_step
            end
            i_small += i_small_step
            j_small += j_small_step
        end
    end

    return nothing
end
end # @muladd
