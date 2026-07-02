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

@inline function calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions,
                                                    mesh::P4estMesh{2}, equations,
                                                    dg, cache; calc_bar_states = true)
    (; contravariant_vectors) = cache.elements
    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states

    if limiter.bar_states == false
        return nothing
    end

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    @threaded for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]
        secondary_direction = indices2direction(secondary_indices)

        # Create the local i,j indexing
        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)
        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_primary = i_primary_start - i_primary_step
        j_primary = j_primary_start - j_primary_step
        i_secondary = i_secondary_start - i_secondary_step
        j_secondary = j_secondary_start - j_secondary_step

        for node in eachnode(dg)
            # Increment primary element indices
            i_primary += i_primary_step
            j_primary += j_primary_step
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step

            # Get the normal direction on the primary element.
            # Contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(primary_direction,
                                                    contravariant_vectors, i_primary,
                                                    j_primary, primary_element)

            u_primary = get_node_vars(u, equations, dg, i_primary, j_primary,
                                      primary_element)
            u_secondary = get_node_vars(u, equations, dg, i_secondary, j_secondary,
                                        secondary_element)

            lambda = max_abs_speed_naive(u_primary, u_secondary, normal_direction,
                                         equations)
            if primary_direction == 1
                lambda1[i_primary, j_primary, primary_element] = lambda
                lambda1[i_secondary + 1, j_secondary, secondary_element] = lambda
            elseif primary_direction == 2
                lambda1[i_primary + 1, j_primary, primary_element] = lambda
                lambda1[i_secondary, j_secondary, secondary_element] = lambda
            elseif primary_direction == 3
                lambda2[i_primary, j_primary, primary_element] = lambda
                lambda2[i_secondary, j_secondary + 1, secondary_element] = lambda
            else # primary_direction == 4
                lambda2[i_primary, j_primary + 1, primary_element] = lambda
                lambda2[i_secondary, j_secondary, secondary_element] = lambda
            end

            calc_bar_states || continue

            flux_primary = flux(u_primary, normal_direction, equations)
            flux_secondary = flux(u_secondary, normal_direction, equations)

            bar_state = 0.5 * (u_primary + u_secondary) -
                        0.5 * (flux_secondary - flux_primary) / lambda
            if primary_direction == 1
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_primary, j_primary, primary_element)
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_secondary + 1, j_secondary, secondary_element)
            elseif primary_direction == 2
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_primary + 1, j_primary, primary_element)
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_secondary, j_secondary, secondary_element)
            elseif primary_direction == 3
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_primary, j_primary, primary_element)
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_secondary, j_secondary + 1, secondary_element)
            else # primary_direction == 4
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_primary, j_primary + 1, primary_element)
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_secondary, j_secondary, secondary_element)
            end
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states_mortar!(u, t, limiter, boundary_conditions,
                                                 mesh::P4estMesh{2}, equations,
                                                 dg, cache; calc_bar_states = true)
    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states
    if nmortars(dg, cache) == 0
        return nothing
    end

    (; mortar_weights, mortar_weights_sums) = dg.mortar
    (; neighbor_ids, node_indices) = cache.mortars
    (; contravariant_vectors) = cache.elements
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]

        # Get index information on small and large elements
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

        i_small = i_small_start - i_small_step
        j_small = j_small_start - j_small_step
        for node_small in eachnode(dg)
            i_small += i_small_step
            j_small += j_small_step

            i_mortar = iszero(i_small_step) ? j_small : i_small

            for small_element_index in 1:2
                small_element = neighbor_ids[small_element_index, mortar]
                u_small = get_node_vars(u, equations, dg, i_small, j_small,
                                        small_element)
                normal_direction_small = get_normal_direction(small_direction,
                                                              contravariant_vectors,
                                                              i_small, j_small,
                                                              small_element)

                i_large = i_large_start - i_large_step
                j_large = j_large_start - j_large_step
                for node_large in eachnode(dg)
                    i_large += i_large_step
                    j_large += j_large_step

                    j_mortar = iszero(i_large_step) ? j_large : i_large

                    weight = mortar_weights[j_mortar, i_mortar, small_element_index]
                    !iszero(weight) || continue

                    u_large = get_node_vars(u, equations, dg, i_large, j_large,
                                            large_element)

                    lambda = max_abs_speed_naive(u_small, u_large,
                                                 normal_direction_small,
                                                 equations)

                    if small_direction == 1
                        lambda1[i_small, j_small, small_element] += weight * lambda /
                                                                    mortar_weights_sums[i_mortar,
                                                                                        1]
                        lambda1[i_large + 1, j_large, large_element] += weight *
                                                                        lambda /
                                                                        mortar_weights_sums[j_mortar,
                                                                                            2]
                    elseif small_direction == 2
                        lambda1[i_small + 1, j_small, small_element] += weight *
                                                                        lambda /
                                                                        mortar_weights_sums[i_mortar,
                                                                                            1]
                        lambda1[i_large, j_large, large_element] += weight * lambda /
                                                                    mortar_weights_sums[j_mortar,
                                                                                        2]
                    elseif small_direction == 3
                        lambda2[i_small, j_small, small_element] += weight * lambda /
                                                                    mortar_weights_sums[i_mortar,
                                                                                        1]
                        lambda2[i_large, j_large + 1, large_element] += weight *
                                                                        lambda /
                                                                        mortar_weights_sums[j_mortar,
                                                                                            2]
                    else # small_direction == 4
                        lambda2[i_small, j_small + 1, small_element] += weight *
                                                                        lambda /
                                                                        mortar_weights_sums[i_mortar,
                                                                                            1]
                        lambda2[i_large, j_large, large_element] += weight * lambda /
                                                                    mortar_weights_sums[j_mortar,
                                                                                        2]
                    end

                    calc_bar_states || continue

                    flux_small = flux(u_small, normal_direction_small, equations)
                    flux_large = flux(u_large, normal_direction_small, equations)
                    bar_state = 0.5 * (u_small + u_large) -
                                0.5 * (flux_large - flux_small) / lambda

                    if small_direction == 1
                        for v in eachvariable(equations)
                            bar_states1[v, i_small, j_small, small_element] += weight *
                                                                               bar_state[v] /
                                                                               mortar_weights_sums[i_mortar,
                                                                                                   1]
                            bar_states1[v, i_large + 1, j_large, large_element] += weight *
                                                                                   bar_state[v] /
                                                                                   mortar_weights_sums[j_mortar,
                                                                                                       2]
                        end
                    elseif small_direction == 2
                        for v in eachvariable(equations)
                            bar_states1[v, i_small + 1, j_small, small_element] += weight *
                                                                                   bar_state[v] /
                                                                                   mortar_weights_sums[i_mortar,
                                                                                                       1]
                            bar_states1[v, i_large, j_large, large_element] += weight *
                                                                               bar_state[v] /
                                                                               mortar_weights_sums[j_mortar,
                                                                                                   2]
                        end
                    elseif small_direction == 3
                        for v in eachvariable(equations)
                            bar_states2[v, i_small, j_small, small_element] += weight *
                                                                               bar_state[v] /
                                                                               mortar_weights_sums[i_mortar,
                                                                                                   1]
                            bar_states2[v, i_large, j_large + 1, large_element] += weight *
                                                                                   bar_state[v] /
                                                                                   mortar_weights_sums[j_mortar,
                                                                                                       2]
                        end
                    else # small_direction == 4
                        for v in eachvariable(equations)
                            bar_states2[v, i_small, j_small + 1, small_element] += weight *
                                                                                   bar_state[v] /
                                                                                   mortar_weights_sums[i_mortar,
                                                                                                       1]
                            bar_states2[v, i_large, j_large, large_element] += weight *
                                                                               bar_state[v] /
                                                                               mortar_weights_sums[j_mortar,
                                                                                                   2]
                        end
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_lambdas_bar_states_boundary!(u, t, limiter,
                                                   boundary_conditions::BoundaryConditionPeriodic,
                                                   mesh::P4estMesh{2}, equations, dg,
                                                   cache; calc_bar_states = true)
    return nothing
end

@inline function calc_lambdas_bar_states_boundary!(u, t, limiter, boundary_conditions,
                                                   mesh::P4estMesh{2}, equations, dg,
                                                   cache; calc_bar_states = true)
    (; boundary_condition_types, boundary_indices) = boundary_conditions
    (; contravariant_vectors) = cache.elements

    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states

    (; boundaries) = cache
    index_range = eachnode(dg)

    # Allocation-free version of `foreach(enumerate((...))`
    foreach_enumerate(boundary_condition_types) do (i, boundary_condition)
        for boundary in boundary_indices[i]
            element = boundaries.neighbor_ids[boundary]

            node_indices = boundaries.node_indices[boundary]
            direction = indices2direction(node_indices)

            i_node_start, i_node_step = index_to_start_step_2d(node_indices[1],
                                                               index_range)
            j_node_start, j_node_step = index_to_start_step_2d(node_indices[2],
                                                               index_range)

            i_node = i_node_start - i_node_step
            j_node = j_node_start - j_node_step
            for node in eachnode(dg)
                i_node += i_node_step
                j_node += j_node_step

                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, element)

                u_inner = get_node_vars(u, equations, dg, i_node, j_node, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_condition, normal_direction,
                                                   mesh, equations, dg, cache,
                                                   i_node, j_node, element)

                lambda = max_abs_speed_naive(u_inner, u_outer, normal_direction,
                                             equations)
                if direction == 1
                    lambda1[i_node, j_node, element] = lambda
                elseif direction == 2
                    lambda1[i_node + 1, j_node, element] = lambda
                elseif direction == 3
                    lambda2[i_node, j_node, element] = lambda
                else # direction == 4
                    lambda2[i_node, j_node + 1, element] = lambda
                end

                calc_bar_states || continue

                flux_inner = flux(u_inner, normal_direction, equations)
                flux_outer = flux(u_outer, normal_direction, equations)

                bar_state = 0.5 * (u_inner + u_outer) -
                            0.5 * (flux_outer - flux_inner) / lambda
                if direction == 1
                    set_node_vars!(bar_states1, bar_state, equations, dg,
                                   i_node, j_node, element)
                elseif direction == 2
                    set_node_vars!(bar_states1, bar_state, equations, dg,
                                   i_node + 1, j_node, element)
                elseif direction == 3
                    set_node_vars!(bar_states2, bar_state, equations, dg,
                                   i_node, j_node, element)
                else # direction == 4
                    set_node_vars!(bar_states2, bar_state, equations, dg,
                                   i_node, j_node + 1, element)
                end
            end
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
    @unpack elements, mortars = cache
    @unpack neighbor_ids, node_indices, u_large = mortars
    @unpack contravariant_vectors = elements
    (; mortar_weights, mortar_weights_sums) = mortar_idp
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
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

        for small_element_index in 1:2
            small_element = neighbor_ids[small_element_index, mortar]
            surface_flux_values[:, :, small_direction, small_element] .= zero(eltype(surface_flux_values))
        end
        large_element = neighbor_ids[3, mortar]
        surface_flux_values[:, :, large_direction, large_element] .= zero(eltype(surface_flux_values))

        i_small = i_small_start
        j_small = j_small_start
        # Calculate fluxes
        for i in eachnode(dg)
            i_mortar = iszero(i_small_step) ? j_small : i_small

            for small_element_index in 1:2
                small_element = neighbor_ids[small_element_index, mortar]
                u_small_local, _ = get_surface_node_vars(mortars.u, equations, dg,
                                                         small_element_index, i, mortar)

                # Get the normal direction on the small element.
                # Note, contravariant vectors at interfaces in negative coordinate direction
                # are pointing inwards. This is handled by `get_normal_direction`.
                normal_direction_small = get_normal_direction(small_direction,
                                                              contravariant_vectors,
                                                              i_small, j_small,
                                                              small_element)

                i_large = i_large_start
                j_large = j_large_start
                for j in eachnode(dg)
                    j_mortar = iszero(i_large_step) ? j_large : i_large

                    factor = mortar_weights[j_mortar, i_mortar, small_element_index]
                    if !isapprox(factor, zero(typeof(factor)))
                        u_large_local = get_node_vars(u_large, equations, dg, j, mortar)

                        normal_direction_large = get_normal_direction(large_direction,
                                                                      contravariant_vectors,
                                                                      i_large, j_large,
                                                                      large_element)
                        # TODO: What do I do with the normal_directions? Doesn't make sense right now. See theory.

                        flux = surface_flux(u_small_local, u_large_local,
                                            normal_direction_small, equations)

                        # Add flux to small element
                        multiply_add_to_node_vars!(surface_flux_values,
                                                   factor /
                                                   mortar_weights_sums[i_mortar, 1],
                                                   flux, equations, dg,
                                                   i, small_direction, small_element)
                        # Add flux to large element
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
            end
            i_small += i_small_step
            j_small += j_small_step
        end
    end

    return nothing
end
end # @muladd
