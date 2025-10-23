# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_bounds_twosided_interface!(var_min, var_max, variable,
                                         u, t, semi, mesh::P4estMesh{2}, equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    index_range = eachnode(dg)

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = cache.interfaces.neighbor_ids[1, interface]
        primary_indices = cache.interfaces.node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = cache.interfaces.neighbor_ids[2, interface]
        secondary_indices = cache.interfaces.node_indices[2, interface]

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

    # Calc bounds at mortars
    # TODO: How to include values at mortar interfaces?
    # See comment above TreeMesh version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Get index information on the small elements
        small_indices = cache.mortars.node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = cache.mortars.node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            i_mortar_s = iszero(i_small_step) ? j_small : i_small
            i_mortar_l = iszero(i_large_step) ? j_large : i_large

            var_lower = u[variable, i_small, j_small, lower_element]
            var_upper = u[variable, i_small, j_small, upper_element]
            var_large = u[variable, i_large, j_large, large_element]

            i_small_inner = i_small_start
            j_small_inner = j_small_start
            i_large_inner = i_large_start
            j_large_inner = j_large_start
            for j in eachnode(dg)
                j_mortar_s = iszero(i_small_step) ? j_small_inner : i_small_inner
                j_mortar_l = iszero(i_large_step) ? j_large_inner : i_large_inner

                # values of large element to lower element
                if l2_mortars || dg.mortar.mortar_weights[i_mortar_l, j_mortar_s, 1] > 0
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
                if l2_mortars || dg.mortar.mortar_weights[j_mortar_l, i_mortar_s, 1] > 0
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
                if l2_mortars || dg.mortar.mortar_weights[i_mortar_l, j_mortar_s, 2] > 0
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
                if l2_mortars || dg.mortar.mortar_weights[j_mortar_l, i_mortar_s, 2] > 0
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

    # Calc bounds at physical boundaries
    calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                   boundary_conditions,
                                   mesh, equations, dg, cache)

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

function calc_bounds_onesided_interface!(var_minmax, minmax, variable, u, t, semi,
                                         mesh::P4estMesh{2})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    index_range = eachnode(dg)

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = cache.interfaces.neighbor_ids[1, interface]
        primary_indices = cache.interfaces.node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = cache.interfaces.neighbor_ids[2, interface]
        secondary_indices = cache.interfaces.node_indices[2, interface]

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
            var_primary = variable(get_node_vars(u, equations, dg, i_primary, j_primary,
                                                 primary_element), equations)
            var_secondary = variable(get_node_vars(u, equations, dg, i_secondary,
                                                   j_secondary, secondary_element),
                                     equations)

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

    # Calc bounds at mortars
    # TODO: How to include values at mortar interfaces?
    # See comment above TreeMesh version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Get index information on the small elements
        small_indices = cache.mortars.node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = cache.mortars.node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            i_mortar_s = iszero(i_small_step) ? j_small : i_small
            i_mortar_l = iszero(i_large_step) ? j_large : i_large

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
                j_mortar_s = iszero(i_small_step) ? j_small_inner : i_small_inner
                j_mortar_l = iszero(i_large_step) ? j_large_inner : i_large_inner

                # values of large element to lower element
                if l2_mortars || dg.mortar.mortar_weights[i_mortar_l, j_mortar_s, 1] > 0
                    var_minmax[i_small_inner, j_small_inner, lower_element] = minmax(var_minmax[i_small_inner,
                                                                                                j_small_inner,
                                                                                                lower_element],
                                                                                     var_large)
                end
                # values of lower element to large element
                if l2_mortars || dg.mortar.mortar_weights[j_mortar_l, i_mortar_s, 1] > 0
                    var_minmax[i_large_inner, j_large_inner, large_element] = minmax(var_minmax[i_large_inner,
                                                                                                j_large_inner,
                                                                                                large_element],
                                                                                     var_lower)
                end
                # values of large element to upper element
                if l2_mortars || dg.mortar.mortar_weights[i_mortar_l, j_mortar_s, 2] > 0
                    var_minmax[i_small_inner, j_small_inner, upper_element] = minmax(var_minmax[i_small_inner,
                                                                                                j_small_inner,
                                                                                                upper_element],
                                                                                     var_large)
                end
                # values of upper element to large element
                if l2_mortars || dg.mortar.mortar_weights[j_mortar_l, i_mortar_s, 2] > 0
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

    # Calc bounds at physical boundaries
    calc_bounds_onesided_boundary!(var_minmax, minmax, variable, u, t,
                                   boundary_conditions,
                                   mesh, equations, dg, cache)

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

###############################################################################
# Local two-sided limiting of conservative variables
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   mesh::P4estMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; boundary_interpolation) = dg.basis

    (; positivity_correction_factor) = dg.volume_integral.limiter
    index_range = eachnode(dg)

    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

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

        if small_direction in (1, 3)
            factor_small = -boundary_interpolation[1, 1]
            factor_large = -boundary_interpolation[nnodes(dg), 2]
        else
            factor_large = -boundary_interpolation[1, 1]
            factor_small = -boundary_interpolation[nnodes(dg), 2]
        end
        # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
        # This sign switch is directly applied to the boundary interpolation factors here.

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start

        # Calc minimal low-order solution
        var_min_upper = typemax(eltype(surface_flux_values))
        var_min_lower = typemax(eltype(surface_flux_values))
        var_min_large = typemax(eltype(surface_flux_values))
        for i in eachnode(dg)
            var_upper = u[var_index, i_small, j_small, upper_element]
            var_lower = u[var_index, i_small, j_small, lower_element]
            var_large = u[var_index, i_large, j_large, large_element]
            var_min_upper = min(var_min_upper, var_upper)
            var_min_lower = min(var_min_lower, var_lower)
            var_min_large = min(var_min_large, var_large)

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
        var_min_upper = positivity_correction_factor * var_min_upper
        var_min_lower = positivity_correction_factor * var_min_lower
        var_min_large = positivity_correction_factor * var_min_large

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            var_upper = u[var_index, i_small, j_small, upper_element]
            var_lower = u[var_index, i_small, j_small, lower_element]
            var_large = u[var_index, i_large, j_large, large_element]

            if min(var_upper, var_lower, var_large) < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
            end

            inverse_jacobian_upper = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_small, j_small,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_small, j_small,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_large, j_large,
                                                          large_element)

            # Calculate Pm
            flux_lower_high_order = surface_flux_values_high_order[var_index, i,
                                                                   small_direction,
                                                                   lower_element]
            flux_lower_low_order = surface_flux_values[var_index, i, small_direction,
                                                       lower_element]
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order - flux_lower_low_order)

            flux_upper_high_order = surface_flux_values_high_order[var_index, i,
                                                                   small_direction,
                                                                   upper_element]
            flux_upper_low_order = surface_flux_values[var_index, i, small_direction,
                                                       upper_element]
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order - flux_upper_low_order)

            flux_large_high_order = surface_flux_values_high_order[var_index, i,
                                                                   large_direction,
                                                                   large_element]
            flux_large_low_order = surface_flux_values_high_order[var_index, i,
                                                                  large_direction,
                                                                  large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            # Check if high-order fluxes are finite. Otherwise, use pure low-order fluxes.
            if !all(isfinite.(flux_lower_high_order)) ||
               !all(isfinite(flux_upper_high_order)) ||
               !all(isfinite.(flux_large_high_order))
                limiting_factor[mortar] = 1
                continue
            end

            Qm_upper = min(0, var_min_upper - var_upper)
            Qm_lower = min(0, var_min_lower - var_lower)
            Qm_large = min(0, var_min_large - var_large)

            Pm_upper = min(0, flux_difference_upper)
            Pm_lower = min(0, flux_difference_lower)
            Pm_large = min(0, flux_difference_large)

            Pm_upper = dt * inverse_jacobian_upper * Pm_upper
            Pm_lower = dt * inverse_jacobian_lower * Pm_lower
            Pm_large = dt * inverse_jacobian_large * Pm_large

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm_upper = abs(Qm_upper) / (abs(Pm_upper) + eps(typeof(Qm_upper)) * 100)
            Qm_lower = abs(Qm_lower) / (abs(Pm_lower) + eps(typeof(Qm_lower)) * 100)
            Qm_large = abs(Qm_large) / (abs(Pm_large) + eps(typeof(Qm_large)) * 100)

            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Qm_upper,
                                          1 - Qm_lower, 1 - Qm_large)

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
@inline function limiting_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                mesh::P4estMesh{2}, variable)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, node_indices) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; boundary_interpolation) = dg.basis

    (; limiter) = dg.volume_integral
    (; positivity_correction_factor) = limiter

    index_range = eachnode(dg)

    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

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

        if small_direction in (1, 3)
            factor_small = -boundary_interpolation[1, 1]
            factor_large = -boundary_interpolation[nnodes(dg), 2]
        else
            factor_large = -boundary_interpolation[1, 1]
            factor_small = -boundary_interpolation[nnodes(dg), 2]
        end
        # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
        # This sign switch is directly applied to the boundary interpolation factors here.

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            inverse_jacobian_upper = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_small, j_small,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_small, j_small,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_large, j_large,
                                                          large_element)

            u_lower = get_node_vars(u, equations, dg, i_small, j_small, lower_element)
            u_upper = get_node_vars(u, equations, dg, i_small, j_small, upper_element)
            u_large = get_node_vars(u, equations, dg, i_large, j_large, large_element)
            var_lower = variable(u_lower, equations)
            var_upper = variable(u_upper, equations)
            var_large = variable(u_large, equations)
            if var_lower < 0 || var_upper < 0 || var_large < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end

            var_min_lower = positivity_correction_factor * var_lower
            var_min_upper = positivity_correction_factor * var_upper
            var_min_large = positivity_correction_factor * var_large

            # lower element
            flux_lower_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, small_direction, lower_element)
            flux_lower_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, small_direction, lower_element)
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order .- flux_lower_low_order)
            antidiffusive_flux_lower = inverse_jacobian_lower * flux_difference_lower

            newton_loop!(limiting_factor, var_min_lower, u_lower, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_lower)

            # upper element
            flux_upper_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, small_direction, upper_element)
            flux_upper_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, small_direction, upper_element)
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order .- flux_upper_low_order)
            antidiffusive_flux_upper = inverse_jacobian_upper * flux_difference_upper

            newton_loop!(limiting_factor, var_min_upper, u_upper, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_upper)

            # large element
            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, large_direction, large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, large_direction, large_element)
            flux_difference_large = factor_large *
                                    (flux_large_high_order .- flux_large_low_order)
            antidiffusive_flux_large = inverse_jacobian_large * flux_difference_large

            newton_loop!(limiting_factor, var_min_large, u_large, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end
end # @muladd
