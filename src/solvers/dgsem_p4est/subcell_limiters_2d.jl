# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function get_mortar_index(indices, i, j)
    if indices[1] === :i_forward || indices[1] === :i_backward
        return i
    else # indices[2] === :i_forward || indices[2] === :i_backward
        return j
    end
end

function calc_bounds_twosided_interface!(var_min, var_max, variable, u,
                                         semi, mesh::P4estMesh{2}, equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        # Create the local i,j indexing
        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)
        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        i_secondary = i_secondary_start
        j_secondary = j_secondary_start

        for node in eachnode(dg)
            var_primary = u[variable, i_primary, j_primary, primary_element]
            var_secondary = u[variable, i_secondary, j_secondary, secondary_element]

            var_min[i_primary, j_primary, primary_element] = min(var_min[i_primary,
                                                                         j_primary,
                                                                         primary_element],
                                                                 var_secondary)
            var_max[i_primary, j_primary, primary_element] = max(var_max[i_primary,
                                                                         j_primary,
                                                                         primary_element],
                                                                 var_secondary)

            var_min[i_secondary, j_secondary, secondary_element] = min(var_min[i_secondary,
                                                                               j_secondary,
                                                                               secondary_element],
                                                                       var_primary)
            var_max[i_secondary, j_secondary, secondary_element] = max(var_max[i_secondary,
                                                                               j_secondary,
                                                                               secondary_element],
                                                                       var_primary)

            # Increment primary element indices
            i_primary += i_primary_step
            j_primary += j_primary_step
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_mortar!(var_min, var_max, variable, u,
                                              semi, mesh::P4estMesh{2})
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    index_range = eachnode(dg)

    # `mortar_weights` is defined in mortar reference coordinates, so it has to be
    # indexed with the loop counters (i and j). Using the element-local face indices instead
    # would pair mirror-image subcells whenever the large side is traversed backwards,
    # i.e., for `:i_backward in large_indices`.

    # TODO: How to include values at mortar interfaces?
    # See comment above TreeMesh version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            var_lower = u[variable, i_small, j_small, lower_element]
            var_upper = u[variable, i_small, j_small, upper_element]
            var_large = u[variable, i_large, j_large, large_element]

            i_small_inner = i_small_start
            j_small_inner = j_small_start
            i_large_inner = i_large_start
            j_large_inner = j_large_start
            for j in eachnode(dg)
                # values of large element to lower element
                if l2_mortars || dg.mortar.mortar_weights[i, j, 1] > 0
                    var_min[i_small_inner, j_small_inner, lower_element] = min(var_min[i_small_inner,
                                                                                       j_small_inner,
                                                                                       lower_element],
                                                                               var_large)
                    var_max[i_small_inner, j_small_inner, lower_element] = max(var_max[i_small_inner,
                                                                                       j_small_inner,
                                                                                       lower_element],
                                                                               var_large)
                end
                # values of lower element to large element
                if l2_mortars || dg.mortar.mortar_weights[j, i, 1] > 0
                    var_min[i_large_inner, j_large_inner, large_element] = min(var_min[i_large_inner,
                                                                                       j_large_inner,
                                                                                       large_element],
                                                                               var_lower)
                    var_max[i_large_inner, j_large_inner, large_element] = max(var_max[i_large_inner,
                                                                                       j_large_inner,
                                                                                       large_element],
                                                                               var_lower)
                end
                # values of large element to upper element
                if l2_mortars || dg.mortar.mortar_weights[i, j, 2] > 0
                    var_min[i_small_inner, j_small_inner, upper_element] = min(var_min[i_small_inner,
                                                                                       j_small_inner,
                                                                                       upper_element],
                                                                               var_large)
                    var_max[i_small_inner, j_small_inner, upper_element] = max(var_max[i_small_inner,
                                                                                       j_small_inner,
                                                                                       upper_element],
                                                                               var_large)
                end
                # values of upper element to large element
                if l2_mortars || dg.mortar.mortar_weights[j, i, 2] > 0
                    var_min[i_large_inner, j_large_inner, large_element] = min(var_min[i_large_inner,
                                                                                       j_large_inner,
                                                                                       large_element],
                                                                               var_upper)
                    var_max[i_large_inner, j_large_inner, large_element] = max(var_max[i_large_inner,
                                                                                       j_large_inner,
                                                                                       large_element],
                                                                               var_upper)
                end
                i_small_inner += i_small_step
                j_small_inner += j_small_step
                i_large_inner += i_large_step
                j_large_inner += j_large_step
            end
            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                                boundary_conditions::BoundaryConditionPeriodic,
                                                mesh::P4estMesh{2},
                                                equations, dg, cache)
    return nothing
end

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                                boundary_conditions,
                                                mesh::P4estMesh{2},
                                                equations, dg, cache)
    (; boundary_condition_types, boundary_indices) = boundary_conditions
    (; contravariant_vectors) = cache.elements

    (; boundaries) = cache
    index_range = eachnode(dg)

    foreach_enumerate(boundary_condition_types) do (i, boundary_condition)
        for boundary in boundary_indices[i]
            element = boundaries.neighbor_ids[boundary]
            node_indices = boundaries.node_indices[boundary]
            direction = indices2direction(node_indices)

            i_node_start, i_node_step = index_to_start_step_2d(node_indices[1],
                                                               index_range)
            j_node_start, j_node_step = index_to_start_step_2d(node_indices[2],
                                                               index_range)

            i_node = i_node_start
            j_node = j_node_start
            for i in eachnode(dg)
                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, element)

                u_inner = get_node_vars(u, equations, dg, i_node, j_node, element)

                u_outer = get_boundary_outer_state(u_inner, t, boundary_condition,
                                                   normal_direction,
                                                   mesh, equations, dg, cache,
                                                   i_node, j_node, element)
                var_outer = u_outer[variable]

                var_min[i_node, j_node, element] = min(var_min[i_node, j_node, element],
                                                       var_outer)
                var_max[i_node, j_node, element] = max(var_max[i_node, j_node, element],
                                                       var_outer)

                i_node += i_node_step
                j_node += j_node_step
            end
        end
    end

    return nothing
end

function calc_bounds_onesided_interface!(var_minmax, minmax, variable, u,
                                         semi, mesh::P4estMesh{2})
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_values) = subcell_limiter_coefficients(dg.volume_integral)

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        # Create the local i,j indexing
        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)
        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        i_secondary = i_secondary_start
        j_secondary = j_secondary_start

        for node in eachnode(dg)
            var_primary = variable_values[i_primary, j_primary, primary_element]
            var_secondary = variable_values[i_secondary, j_secondary, secondary_element]

            var_minmax[i_primary, j_primary, primary_element] = minmax(var_minmax[i_primary,
                                                                                  j_primary,
                                                                                  primary_element],
                                                                       var_secondary)
            var_minmax[i_secondary, j_secondary, secondary_element] = minmax(var_minmax[i_secondary,
                                                                                        j_secondary,
                                                                                        secondary_element],
                                                                             var_primary)

            # Increment primary element indices
            i_primary += i_primary_step
            j_primary += j_primary_step
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_mortar!(var_minmax, minmax, variable, u,
                                              semi, mesh::P4estMesh{2})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    index_range = eachnode(dg)

    # `mortar_weights` is defined in mortar reference coordinates, so it has to be
    # indexed with the loop counters (i and j). Using the element-local face indices instead
    # would pair mirror-image subcells whenever the large side is traversed backwards,
    # i.e., for `:i_backward in large_indices`.

    # TODO: How to include values at mortar interfaces?
    # See comment above TreeMesh version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            u_lower = get_node_vars(u, equations, dg, i_small, j_small, lower_element)
            u_upper = get_node_vars(u, equations, dg, i_small, j_small, upper_element)
            u_large = get_node_vars(u, equations, dg, i_large, j_large, large_element)
            var_lower = variable(u_lower, equations)
            var_upper = variable(u_upper, equations)
            var_large = variable(u_large, equations)

            i_small_inner = i_small_start
            j_small_inner = j_small_start
            i_large_inner = i_large_start
            j_large_inner = j_large_start
            for j in eachnode(dg)
                # values of large element to lower element
                if l2_mortars || dg.mortar.mortar_weights[i, j, 1] > 0
                    var_minmax[i_small_inner, j_small_inner, lower_element] = minmax(var_minmax[i_small_inner,
                                                                                                j_small_inner,
                                                                                                lower_element],
                                                                                     var_large)
                end
                # values of lower element to large element
                if l2_mortars || dg.mortar.mortar_weights[j, i, 1] > 0
                    var_minmax[i_large_inner, j_large_inner, large_element] = minmax(var_minmax[i_large_inner,
                                                                                                j_large_inner,
                                                                                                large_element],
                                                                                     var_lower)
                end
                # values of large element to upper element
                if l2_mortars || dg.mortar.mortar_weights[i, j, 2] > 0
                    var_minmax[i_small_inner, j_small_inner, upper_element] = minmax(var_minmax[i_small_inner,
                                                                                                j_small_inner,
                                                                                                upper_element],
                                                                                     var_large)
                end
                # values of upper element to large element
                if l2_mortars || dg.mortar.mortar_weights[j, i, 2] > 0
                    var_minmax[i_large_inner, j_large_inner, large_element] = minmax(var_minmax[i_large_inner,
                                                                                                j_large_inner,
                                                                                                large_element],
                                                                                     var_upper)
                end
                i_small_inner += i_small_step
                j_small_inner += j_small_step
                i_large_inner += i_large_step
                j_large_inner += j_large_step
            end
            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_boundary!(var_minmax, minmax, variable, u, t,
                                                boundary_conditions::BoundaryConditionPeriodic,
                                                mesh::P4estMesh{2},
                                                equations, dg, cache)
    return nothing
end

@inline function calc_bounds_onesided_boundary!(var_minmax, minmax, variable, u, t,
                                                boundary_conditions,
                                                mesh::P4estMesh{2},
                                                equations, dg, cache)
    (; boundary_condition_types, boundary_indices) = boundary_conditions
    (; contravariant_vectors) = cache.elements

    (; boundaries) = cache
    index_range = eachnode(dg)

    foreach_enumerate(boundary_condition_types) do (i, boundary_condition)
        for boundary in boundary_indices[i]
            element = boundaries.neighbor_ids[boundary]
            node_indices = boundaries.node_indices[boundary]
            direction = indices2direction(node_indices)

            i_node_start, i_node_step = index_to_start_step_2d(node_indices[1],
                                                               index_range)
            j_node_start, j_node_step = index_to_start_step_2d(node_indices[2],
                                                               index_range)

            i_node = i_node_start
            j_node = j_node_start
            for node in eachnode(dg)
                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, element)

                u_inner = get_node_vars(u, equations, dg, i_node, j_node, element)

                u_outer = get_boundary_outer_state(u_inner, t, boundary_condition,
                                                   normal_direction,
                                                   mesh, equations, dg, cache,
                                                   i_node, j_node, element)
                var_outer = variable(u_outer, equations)

                var_minmax[i_node, j_node, element] = minmax(var_minmax[i_node, j_node,
                                                                        element],
                                                             var_outer)

                i_node += i_node_step
                j_node += j_node_step
            end
        end
    end

    return nothing
end

###############################################################################
# IDP mortar limiting
###############################################################################

@inline function precompute_n_mortars_per_nodes!(volume_integral::VolumeIntegralSubcellLimiting,
                                                 dg, cache,
                                                 mesh::P4estMesh{2})
    if !(dg.mortar isa LobattoLegendreMortarIDP)
        return nothing
    end

    (; n_mortars_per_node) = subcell_limiter_coefficients(volume_integral)
    (; neighbor_ids, node_indices) = cache.mortars
    index_range = eachnode(dg)

    n_mortars_per_node .= zero(eltype(n_mortars_per_node))

    for mortar in eachmortar(dg, cache)
        lower_element = neighbor_ids[1, mortar]
        upper_element = neighbor_ids[2, mortar]
        large_element = neighbor_ids[3, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for node in eachnode(dg)
            n_mortars_per_node[i_small, j_small, lower_element] += 1
            n_mortars_per_node[i_small, j_small, upper_element] += 1
            n_mortars_per_node[i_large, j_large, large_element] += 1

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

###############################################################################
# Local two-sided limiting of conservative variables
@inline function idp_mortar_local_twosided!(limiting_factor, u, dt, semi,
                                            mesh::P4estMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; inverse_weights) = dg.basis

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = subcell_limiter_coefficients(dg.volume_integral)
    variable_string = string(var_index)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

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

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            # Large element
            # Map the mortar node to the large-element face since its orientation may be flipped.
            # The small-element face needs no mapping because it is always traversed forward.
            large_node = get_mortar_index(large_indices, i_large, j_large)
            Q = zalesak_limiting_twosided(u, var_index, i_large, j_large, large_element,
                                          large_node, large_direction, factor, dt,
                                          var_min, var_max, n_mortars_per_node,
                                          mesh, cache)

            # Small elements
            for small_element_index in 1:2
                iszero(Q) && break # Skip if Q is zero, i.e., the limiting factor will be 1

                small_element = neighbor_ids[small_element_index, mortar]
                Q = min(Q,
                        zalesak_limiting_twosided(u, var_index, i_small, j_small,
                                                  small_element, i, small_direction,
                                                  factor, dt,
                                                  var_min, var_max, n_mortars_per_node,
                                                  mesh, cache))
            end

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

##############################################################################
# Local one-sided limiting of nonlinear variables
@inline function idp_mortar_local_onesided!(limiting_factor, u, dt, semi,
                                            mesh::P4estMesh{2}, variable,
                                            min_or_max)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars

    (; inverse_weights) = dg.basis
    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

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

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1 (no limiting needed)

            # Large element
            # Map the mortar node to the large-element face since its orientation may be flipped.
            # The small-element face needs no mapping because it is always traversed forward.
            large_node = get_mortar_index(large_indices, i_large, j_large)
            newton_loop_mortar!(limiting_factor, mortar, u,
                                i_large, j_large, large_element,
                                large_node, large_direction, factor, dt,
                                var_minmax, variable, min_or_max,
                                initial_check_local_onesided_newton_idp,
                                final_check_local_onesided_newton_idp,
                                mesh, equations, dg, cache)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            # Small elements
            for small_element_index in 1:2
                small_element = neighbor_ids[small_element_index, mortar]

                newton_loop_mortar!(limiting_factor, mortar, u,
                                    i_small, j_small, small_element,
                                    i, small_direction, factor, dt,
                                    var_minmax, variable, min_or_max,
                                    initial_check_local_onesided_newton_idp,
                                    final_check_local_onesided_newton_idp,
                                    mesh, equations, dg, cache)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            end

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables
@inline function idp_mortar_positivity_conservative!(limiting_factor, u, dt, semi,
                                                     mesh::P4estMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; inverse_weights) = dg.basis

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; n_mortars_per_node) = subcell_limiter_coefficients(dg.volume_integral)
    # The positivity bound follows from the current solution alone. It is deliberately not read
    # from `variable_bounds`, which holds the bounds of the *local* limiting: with a smoothness
    # indicator only the fraction `alpha_indicator` of the local limiting is applied, so
    # enforcing its bound here would bypass the indicator.
    (; positivity_correction_factor) = dg.mortar.limiter

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

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

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            # Large element
            # Map the mortar node to the large-element face since its orientation may be flipped.
            # The small-element face needs no mapping because it is always traversed forward.
            large_node = get_mortar_index(large_indices, i_large, j_large)
            Q = zalesak_limiting_onesided(u, var_index, i_large, j_large, large_element,
                                          large_node, large_direction, factor, dt,
                                          positivity_correction_factor,
                                          n_mortars_per_node, mesh, cache)

            # Small elements
            for small_element_index in 1:2
                iszero(Q) && break # Skip if Q is zero, i.e., the limiting factor will be 1

                small_element = neighbor_ids[small_element_index, mortar]
                Q = min(Q,
                        zalesak_limiting_onesided(u, var_index, i_small, j_small,
                                                  small_element, i, small_direction,
                                                  factor, dt,
                                                  positivity_correction_factor,
                                                  n_mortars_per_node, mesh, cache))
            end

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end

##############################################################################
# Local one-sided limiting of nonlinear variables
@inline function idp_mortar_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                  mesh::P4estMesh{2}, variable)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars

    (; inverse_weights) = dg.basis
    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    # The nonlinear positivity limiting is the only limiter writing this bound: the nonlinear
    # local limiting is only used for entropies, the nonlinear positivity limiting only for
    # the pressure. Therefore, `var_min` holds the positivity bound and can be reused here.
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

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

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1 (no limiting needed)

            # Large element
            # Map the mortar node to the large-element face since its orientation may be flipped.
            # The small-element face needs no mapping because it is always traversed forward.
            large_node = get_mortar_index(large_indices, i_large, j_large)
            newton_loop_mortar!(limiting_factor, mortar, u,
                                i_large, j_large, large_element,
                                large_node, large_direction, factor, dt,
                                var_min, variable, min,
                                initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp,
                                mesh, equations, dg, cache)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            # Small elements
            for small_element_index in 1:2
                small_element = neighbor_ids[small_element_index, mortar]

                newton_loop_mortar!(limiting_factor, mortar, u,
                                    i_small, j_small, small_element,
                                    i, small_direction, factor, dt,
                                    var_min, variable, min,
                                    initial_check_nonnegative_newton_idp,
                                    final_check_nonnegative_newton_idp,
                                    mesh, equations, dg, cache)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            end

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end
end # @muladd
