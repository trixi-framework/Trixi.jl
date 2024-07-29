# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function calc_lambdas_bar_states_interface!(u, t, limiter, boundary_conditions,
                                                    mesh::P4estMesh{2}, equations,
                                                    dg, cache; calc_bar_states = true)
    (; contravariant_vectors) = cache.elements
    (; lambda1, lambda2, bar_states1, bar_states2) = limiter.cache.container_bar_states

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    # Calc lambdas and bar states at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
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
            elseif primary_direction == 2
                lambda1[i_primary + 1, j_primary, primary_element] = lambda
            elseif primary_direction == 3
                lambda2[i_primary, j_primary, primary_element] = lambda
            else # primary_direction == 4
                lambda2[i_primary, j_primary + 1, primary_element] = lambda
            end
            if secondary_direction == 1
                lambda1[i_secondary, j_secondary, secondary_element] = lambda
            elseif secondary_direction == 2
                lambda1[i_secondary + 1, j_secondary, secondary_element] = lambda
            elseif secondary_direction == 3
                lambda2[i_secondary, j_secondary, secondary_element] = lambda
            else # secondary_direction == 4
                lambda2[i_secondary, j_secondary + 1, secondary_element] = lambda
            end

            !calc_bar_states && continue

            flux_primary = flux(u_primary, normal_direction, equations)
            flux_secondary = flux(u_secondary, normal_direction, equations)

            bar_state = 0.5 * (u_primary + u_secondary) -
                        0.5 * (flux_secondary - flux_primary) / lambda
            if primary_direction == 1
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_primary, j_primary, primary_element)
            elseif primary_direction == 2
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_primary + 1, j_primary, primary_element)
            elseif primary_direction == 3
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_primary, j_primary, primary_element)
            else # primary_direction == 4
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_primary, j_primary + 1, primary_element)
            end
            if secondary_direction == 1
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_secondary, j_secondary, secondary_element)
            elseif secondary_direction == 2
                set_node_vars!(bar_states1, bar_state, equations, dg,
                               i_secondary + 1, j_secondary, secondary_element)
            elseif secondary_direction == 3
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_secondary, j_secondary, secondary_element)
            else # secondary_direction == 4
                set_node_vars!(bar_states2, bar_state, equations, dg,
                               i_secondary, j_secondary + 1, secondary_element)
            end
        end
    end

    calc_lambdas_bar_states_boundary!(u, t, limiter, boundary_conditions,
                                      mesh, equations, dg, cache;
                                      calc_bar_states = calc_bar_states)

    return nothing
end

@inline function calc_lambdas_bar_states_boundary!(u, t, limiter,
                                                   boundary_conditions::BoundaryConditionPeriodic,
                                                   mesh::P4estMesh{2}, equations, dg,
                                                   cache; calc_bar_states = true)
    return nothing
end

# Calc lambdas and bar states at physical boundaries
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

                !calc_bar_states && continue

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
end # @muladd
