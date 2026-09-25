# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# IDP Limiting
###############################################################################

###############################################################################
# Calculation of local bounds using low-order FV solution

@inline function calc_bounds_twosided!(var_min, var_max, variable,
                                       u::AbstractArray{<:Any, 4}, t,
                                       semi, equations)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        # Calculate bounds at Gauss-Lobatto nodes
        for j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, element]
            var_min[i, j, element] = var
            var_max[i, j, element] = var
        end

        # Apply values in x direction
        for j in eachnode(dg), i in 2:nnodes(dg)
            var = u[variable, i - 1, j, element]
            var_min[i, j, element] = min(var_min[i, j, element], var)
            var_max[i, j, element] = max(var_max[i, j, element], var)

            var = u[variable, i, j, element]
            var_min[i - 1, j, element] = min(var_min[i - 1, j, element], var)
            var_max[i - 1, j, element] = max(var_max[i - 1, j, element], var)
        end

        # Apply values in y direction
        for j in 2:nnodes(dg), i in eachnode(dg)
            var = u[variable, i, j - 1, element]
            var_min[i, j, element] = min(var_min[i, j, element], var)
            var_max[i, j, element] = max(var_max[i, j, element], var)

            var = u[variable, i, j, element]
            var_min[i, j - 1, element] = min(var_min[i, j - 1, element], var)
            var_max[i, j - 1, element] = max(var_max[i, j - 1, element], var)
        end
    end

    # Calc bounds at element interfaces and periodic boundaries
    calc_bounds_twosided_interface!(var_min, var_max, variable, u,
                                    semi, mesh, equations)

    # Calc bounds at mortars
    calc_bounds_twosided_mortar!(var_min, var_max, variable, u, semi, mesh)

    # Calc bounds at physical boundaries
    (; boundary_conditions) = semi
    calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                   boundary_conditions,
                                   mesh, equations, dg, cache)
    return nothing
end

@inline function calc_bounds_twosided_interface!(var_min, var_max, variable, u,
                                                 semi, mesh::TreeMesh2D, equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations) = cache.interfaces

    # Process x- and y-oriented interfaces separately. Interfaces with the
    # same orientation update disjoint faces of each element. The barrier
    # between these loops prevents races at element corners.
    for selected_orientation in 1:2
        @threaded for interface in eachinterface(dg, cache)
            orientations[interface] == selected_orientation || continue

            # Get neighboring element ids
            left_element = neighbor_ids[1, interface]
            right_element = neighbor_ids[2, interface]

            limit_left = perform_subcell_limiting(dg.volume_integral, left_element)
            limit_right = perform_subcell_limiting(dg.volume_integral, right_element)
            if limit_left || limit_right
                # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this interface
            else
                # Subcell limiting is not necessary for both elements => Skip this interface
                continue
            end

            for i in eachnode(dg)
                # Define node indices for left and right element based on the interface orientation
                if orientations[interface] == 1
                    index_left = (nnodes(dg), i)
                    index_right = (1, i)
                else # if orientation == 2
                    index_left = (i, nnodes(dg))
                    index_right = (i, 1)
                end

                if limit_right
                    var_left = u[variable, index_left..., left_element]
                    var_min[index_right..., right_element] = min(var_min[index_right...,
                                                                         right_element],
                                                                 var_left)
                    var_max[index_right..., right_element] = max(var_max[index_right...,
                                                                         right_element],
                                                                 var_left)
                end

                if limit_left
                    var_right = u[variable, index_right..., right_element]
                    var_min[index_left..., left_element] = min(var_min[index_left...,
                                                                       left_element],
                                                               var_right)
                    var_max[index_left..., left_element] = max(var_max[index_left...,
                                                                       left_element],
                                                               var_right)
                end
            end
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_mortar!(var_min, var_max, variable, u,
                                              semi, mesh::TreeMesh2D)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    # - For LobattoLegendreMortarIDP: include only values of nodes with nonnegative local weights
    # - For LobattoLegendreMortarL2: include all neighboring values
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            node_small = 1
            node_large = nnodes(dg)
        else # large_sides[mortar] == 2 -> small elements on left side
            node_small = nnodes(dg)
            node_large = 1
        end

        for i in eachnode(dg)
            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                # L2 mortars in y-direction
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end
            # Get solution data
            var_small = (u[variable, indices_small..., neighbor_ids[1, mortar]],
                         u[variable, indices_small..., neighbor_ids[2, mortar]])
            # Using the following version with `ntuple` creates allocations due to a type instability of `indices_small`.
            # var_small = index -> u[variable, indices_small..., neighbor_ids[index, mortar]]
            # Theoretically, that could be fixed with the following version:
            # f = let indices_small = indices_small
            #     index -> u[variable, indices_small..., neighbor_ids[index, mortar]]
            # end
            # var_small = ntuple(f, Val(2))
            var_large = u[variable, indices_large..., large_element]

            for j in eachnode(dg)
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small_inner = (node_small, j)
                    indices_large_inner = (node_large, j)
                else
                    # L2 mortars in y-direction
                    indices_small_inner = (j, node_small)
                    indices_large_inner = (j, node_large)
                end

                for small_element_index in 1:2
                    small_element = neighbor_ids[small_element_index, mortar]
                    # from large to small element
                    if l2_mortars ||
                       dg.mortar.mortar_weights[i, j, small_element_index] > 0
                        var_min[indices_small_inner..., small_element] = min(var_min[indices_small_inner...,
                                                                                     small_element],
                                                                             var_large)
                        var_max[indices_small_inner..., small_element] = max(var_max[indices_small_inner...,
                                                                                     small_element],
                                                                             var_large)
                    end
                    # from small to large element
                    if l2_mortars ||
                       dg.mortar.mortar_weights[j, i, small_element_index] > 0
                        var_min[indices_large_inner..., large_element] = min(var_min[indices_large_inner...,
                                                                                     large_element],
                                                                             var_small[small_element_index])
                        var_max[indices_large_inner..., large_element] = max(var_max[indices_large_inner...,
                                                                                     large_element],
                                                                             var_small[small_element_index])
                    end
                end
            end
        end
    end

    return nothing
end

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                                boundary_conditions,
                                                mesh::TreeMesh{2}, equations,
                                                dg, cache)
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                node_index = (1, i)
                boundary_index = 1
            else # Element is on the left, boundary on the right
                node_index = (nnodes(dg), i)
                boundary_index = 2
            end
            if orientation == 2
                node_index = reverse(node_index)
                boundary_index += 2
            end
            u_inner = get_node_vars(u, equations, dg, node_index..., element)
            u_outer = get_boundary_outer_state(u_inner, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg, cache,
                                               node_index..., element)
            var_outer = u_outer[variable]

            var_min[node_index..., element] = min(var_min[node_index..., element],
                                                  var_outer)
            var_max[node_index..., element] = max(var_max[node_index..., element],
                                                  var_outer)
        end
    end

    return nothing
end

@inline function calc_bounds_onesided!(var_minmax, min_or_max, variable,
                                       u::AbstractArray{<:Any, 4}, t, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_values) = subcell_limiter_coefficients(dg.volume_integral)

    # Cache the nonlinear variable once per node before constructing the bounds.
    # This avoids reevaluating the variable at interfaces.
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        # Calculate variable values at Gauss-Lobatto nodes
        for j in eachnode(dg), i in eachnode(dg)
            var = variable(get_node_vars(u, equations, dg, i, j, element), equations)
            variable_values[i, j, element] = var
            var_minmax[i, j, element] = var
        end

        # Apply neighboring values in the x direction
        for j in eachnode(dg), i in 2:nnodes(dg)
            var_minmax[i, j, element] = min_or_max(var_minmax[i, j, element],
                                                   variable_values[i - 1, j, element])

            var_minmax[i - 1, j, element] = min_or_max(var_minmax[i - 1, j, element],
                                                       variable_values[i, j, element])
        end

        # Apply neighboring values in the y direction
        for j in 2:nnodes(dg), i in eachnode(dg)
            var_minmax[i, j, element] = min_or_max(var_minmax[i, j, element],
                                                   variable_values[i, j - 1, element])

            var_minmax[i, j - 1, element] = min_or_max(var_minmax[i, j - 1, element],
                                                       variable_values[i, j, element])
        end
    end

    # Calc bounds at element interfaces and periodic boundaries
    calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u,
                                    semi, mesh)

    # Calc bounds at mortars
    calc_bounds_onesided_mortar!(var_minmax, min_or_max, variable, u, semi, mesh)

    # Calc bounds at physical boundaries
    (; boundary_conditions) = semi
    calc_bounds_onesided_boundary!(var_minmax, min_or_max, variable, u, t,
                                   boundary_conditions,
                                   mesh, equations, dg, cache)

    return nothing
end

@inline function calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u,
                                                 semi, mesh::TreeMesh2D)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_values) = subcell_limiter_coefficients(dg.volume_integral)
    n_nodes = nnodes(dg)

    (; neighbor_ids, orientations) = cache.interfaces

    # Process x- and y-oriented interfaces separately. Interfaces with the
    # same orientation update disjoint faces of each element. The barrier
    # between these loops prevents races at element corners.
    for selected_orientation in 1:2
        @threaded for interface in eachinterface(dg, cache)
            orientations[interface] == selected_orientation || continue

            # Get neighboring element ids
            left_element = neighbor_ids[1, interface]
            right_element = neighbor_ids[2, interface]

            limit_left = perform_subcell_limiting(dg.volume_integral, left_element)
            limit_right = perform_subcell_limiting(dg.volume_integral, right_element)
            if limit_left || limit_right
                # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this interface
            else
                # Subcell limiting is not necessary for both elements => Skip this interface
                continue
            end

            for i in eachnode(dg)
                # Define node indices for left and right element based on the interface orientation
                if orientations[interface] == 1
                    index_left = (n_nodes, i)
                    index_right = (1, i)
                else # if orientation == 2
                    index_left = (i, n_nodes)
                    index_right = (i, 1)
                end

                if limit_right
                    # Use cached value if available, otherwise compute it
                    var_left = if limit_left
                        variable_values[index_left..., left_element]
                    else
                        variable(get_node_vars(u, equations, dg, index_left...,
                                               left_element), equations)
                    end
                    var_minmax[index_right..., right_element] = min_or_max(var_minmax[index_right...,
                                                                                      right_element],
                                                                           var_left)
                end
                if limit_left
                    # Use cached value if available, otherwise compute it
                    var_right = if limit_right
                        variable_values[index_right..., right_element]
                    else
                        variable(get_node_vars(u, equations, dg, index_right...,
                                               right_element), equations)
                    end
                    var_minmax[index_left..., left_element] = min_or_max(var_minmax[index_left...,
                                                                                    left_element],
                                                                         var_right)
                end
            end
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_mortar!(var_minmax, min_or_max, variable, u,
                                              semi, mesh::TreeMesh2D)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    # See comment above two-sided version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            node_small = 1
            node_large = nnodes(dg)
        else # large_sides[mortar] == 2 -> small elements on left side
            node_small = nnodes(dg)
            node_large = 1
        end

        for i in eachnode(dg)
            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                # L2 mortars in y-direction
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end
            u_lower = get_node_vars(u, equations, dg, indices_small..., lower_element)
            u_upper = get_node_vars(u, equations, dg, indices_small..., upper_element)
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)
            var_lower = variable(u_lower, equations)
            var_upper = variable(u_upper, equations)
            var_large = variable(u_large, equations)

            for j in eachnode(dg)
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small_inner = (node_small, j)
                    indices_large_inner = (node_large, j)
                else
                    # L2 mortars in y-direction
                    indices_small_inner = (j, node_small)
                    indices_large_inner = (j, node_large)
                end

                # values of large element to lower element
                if l2_mortars || dg.mortar.mortar_weights[i, j, 1] > 0
                    var_minmax[indices_small_inner..., lower_element] = min_or_max(var_minmax[indices_small_inner...,
                                                                                              lower_element],
                                                                                   var_large)
                end
                # values of lower element to large element
                if l2_mortars || dg.mortar.mortar_weights[j, i, 1] > 0
                    var_minmax[indices_large_inner..., large_element] = min_or_max(var_minmax[indices_large_inner...,
                                                                                              large_element],
                                                                                   var_lower)
                end
                # values of large element to upper element
                if l2_mortars || dg.mortar.mortar_weights[i, j, 2] > 0
                    var_minmax[indices_small_inner..., upper_element] = min_or_max(var_minmax[indices_small_inner...,
                                                                                              upper_element],
                                                                                   var_large)
                end
                # values of upper element to large element
                if l2_mortars || dg.mortar.mortar_weights[j, i, 2] > 0
                    var_minmax[indices_large_inner..., large_element] = min_or_max(var_minmax[indices_large_inner...,
                                                                                              large_element],
                                                                                   var_upper)
                end
            end
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_boundary!(var_minmax, min_or_max, variable, u, t,
                                                boundary_conditions,
                                                mesh::TreeMesh{2}, equations,
                                                dg, cache)
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                node_index = (1, i)
                boundary_index = 1
            else # Element is on the left, boundary on the right
                node_index = (nnodes(dg), i)
                boundary_index = 2
            end
            if orientation == 2
                node_index = reverse(node_index)
                boundary_index += 2
            end
            u_inner = get_node_vars(u, equations, dg, node_index..., element)
            u_outer = get_boundary_outer_state(u_inner, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg, cache,
                                               node_index..., element)
            var_outer = variable(u_outer, equations)

            var_minmax[node_index..., element] = min_or_max(var_minmax[node_index...,
                                                                       element],
                                                            var_outer)
        end
    end

    return nothing
end

@inline function merge_alphas!(alpha::AbstractArray{<:Any, 3}, alpha_local,
                               alpha_indicator, dg, cache)
    # `alpha` holds the positivity limiting factor, `alpha_local` the local one. Positivity
    # has to be enforced completely, while the local limiting is only applied with the
    # fraction `alpha_indicator`. Blending the local limiting *on top of* the positivity one
    # (instead of taking a convex combination of both) makes sure that the merged factor
    # never falls below `alpha`.
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            alpha[i, j, element] = alpha[i, j, element] +
                                   alpha_indicator[element] *
                                   max(0,
                                       alpha_local[i, j, element] -
                                       alpha[i, j, element])
        end
    end

    return nothing
end

@inline function merge_alphas_mortar!(limiting_factor, limiting_factor_local,
                                      alpha_indicator, dg, mesh::AbstractMesh{2}, cache)
    # Same blending as `merge_alphas!` uses within the elements: the positivity limiting in
    # `limiting_factor` is enforced completely, while the local limiting is added on top with
    # the fraction `alpha_indicator`. This never lets the merged factor fall below the
    # positivity one. A mortar is assigned the largest `alpha_indicator` of its elements.
    (; neighbor_ids) = cache.mortars
    @threaded for mortar in eachmortar(dg, cache)
        alpha_element = max(alpha_indicator[neighbor_ids[1, mortar]],
                            alpha_indicator[neighbor_ids[2, mortar]],
                            alpha_indicator[neighbor_ids[3, mortar]])
        limiting_factor[mortar] = limiting_factor[mortar] +
                                  alpha_element *
                                  max(0,
                                      limiting_factor_local[mortar] -
                                      limiting_factor[mortar])
    end

    return nothing
end

###############################################################################
# Local minimum and maximum limiting of conservative variables

@inline function idp_local_twosided!(alpha, limiter, u::AbstractArray{<:Any, 4}, t, dt,
                                     semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis # Plays role of inverse DG-subcell sizes

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    variable_string = string(variable)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]
    if limiter.bar_states == false
        calc_bounds_twosided!(var_min, var_max, variable, u, t, semi, equations)
    end

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for j in eachnode(dg), i in eachnode(dg)
            isone(alpha[i, j, element]) && continue # Skip if alpha is already 1

            var = u[variable, i, j, element]
            # Real Zalesak type limiter
            #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #         for each interface, not each node

            Qp = max(0, (var_max[i, j, element] - var) / dt)
            Qm = min(0, (var_min[i, j, element] - var) / dt)

            # Calculate Pp and Pm
            # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, element]

            Pp = max(0, val_flux1_local) + max(0, val_flux1_local_ip1) +
                 max(0, val_flux2_local) + max(0, val_flux2_local_jp1)
            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1)

            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, element)
            Pp = inverse_jacobian * Pp
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            eps_ = eps(typeof(Qp)) * 100 * abs(var_max[i, j, element])
            Qp = abs(Qp) / (abs(Pp) + eps_)
            Qm = abs(Qm) / (abs(Pm) + eps_)

            # Calculate alpha at nodes
            alpha[i, j, element] = max(alpha[i, j, element], 1 - min(1, Qp, Qm))
        end
    end

    return nothing
end

##############################################################################
# Local minimum or maximum limiting of nonlinear variables

@inline function idp_local_onesided!(alpha, limiter, u::AbstractArray{<:Real, 4},
                                     t, dt, semi, variable, min_or_max)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]
    if limiter.bar_states == false
        calc_bounds_onesided!(var_minmax, min_or_max, variable, u, t, semi)
    end

    # Perform Newton's bisection method to find new alpha
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for j in eachnode(dg), i in eachnode(dg)
            isone(alpha[i, j, element]) && continue # Skip if alpha is already 1

            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, element)
            u_local = get_node_vars(u, equations, dg, i, j, element)
            newton_loops_alpha!(alpha, var_minmax[i, j, element], u_local,
                                i, j, element, variable, min_or_max,
                                initial_check_local_onesided_newton_idp,
                                final_check_local_onesided_newton_idp, inverse_jacobian,
                                dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables

@inline function idp_positivity_conservative!(alpha, limiter,
                                              u::AbstractArray{<:Real, 4},
                                              dt, semi, variable)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    # Check whether the local limiting already computed a bound for this variable in this stage.
    # The local limiting always runs before the positivity limiting, so `var_min` holds a valid
    # local bound if this is `true`.
    was_limited_locally = limiter.local_twosided &&
                          (variable in limiter.local_twosided_variables_cons)
    # Without a smoothness indicator, both limiters are enforced completely and `var_min`
    # holds the more restrictive of the two bounds. With a smoothness indicator, local bounds are
    # only enforced fractionally, while positivity limiting is enforced completely.
    # In that case, the local bound is stored in `var_min`, while the positivity bound is stored in
    # `var_min_positivity`.
    enabled_indicator = !isnothing(limiter.indicator)

    # Array the positivity bound was written to. Only with a smoothness indicator it is stored
    # separately; otherwise the more restrictive of the two bounds is kept in `var_min`.
    if was_limited_locally && enabled_indicator
        var_min_positivity = variable_bounds[Symbol(string(variable),
                                                    "_min_positivity")]
    else
        var_min_positivity = var_min
    end
    # Only when both limiters share `var_min`, the local bound has to be compared to the
    # positivity bound before it is overwritten.
    merge_bounds = was_limited_locally && !enabled_indicator

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, element]
            if var < 0
                error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
            end

            # Compute bound
            bound = positivity_correction_factor * var
            if merge_bounds && var_min[i, j, element] >= bound
                # Local limiting is more restrictive than positivity limiting and is
                # enforced completely (no smoothness indicator)
                # => Skip positivity limiting for this node
                continue
            end
            var_min_positivity[i, j, element] = bound

            isone(alpha[i, j, element]) && continue # Skip if alpha is already 1

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each interface, not each node
            # Note: Use `bound` and not `var_min`, which may hold the local bound. Enforcing
            #       that one here would bypass the smoothness indicator.
            Qm = min(0, (bound - var) / dt)

            # Calculate Pm
            # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, element]

            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1)

            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, element)
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            eps_ = eps(typeof(Qm)) * 100
            Qm = abs(Qm) / (abs(Pm) + eps_)

            # Calculate alpha
            alpha[i, j, element] = max(alpha[i, j, element], 1 - Qm)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of nonlinear variables

@inline function idp_positivity_nonlinear!(alpha, limiter,
                                           u::AbstractArray{<:Real, 4},
                                           dt, semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, element)

            # Compute bound
            u_local = get_node_vars(u, equations, dg, i, j, element)
            var = variable(u_local, equations)
            if var < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end
            var_min[i, j, element] = positivity_correction_factor * var

            # Perform Newton's bisection method to find new alpha
            newton_loops_alpha!(alpha, var_min[i, j, element], u_local, i, j, element,
                                variable, min, initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp, inverse_jacobian,
                                dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Auxiliary functions for Newton-bisection method

@inline function newton_loops_alpha!(alpha, bound, u, i, j, element,
                                     variable, min_or_max,
                                     initial_check, final_check,
                                     inverse_jacobian, dt,
                                     equations::AbstractEquations{2},
                                     dg, cache, limiter)
    (; inverse_weights) = dg.basis # Plays role of inverse DG-subcell sizes
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes

    indices = (i, j, element)
    isone(alpha[indices...]) && return nothing # Skip if alpha is already 1

    # The updated state is a convex combination of one provisional state per antidiffusive flux
    # contributing to this node. Scaling the fluxes with the number of these contributions is
    # sharper than using the uniform constant `2 * ndims` at nodes adjacent to an element boundary.
    # In 2D, the number of contributions is 4 for inner nodes, 3 for nodes at an element boundary,
    # and 2 for nodes at an element corner.
    gamma = min(limiter.gamma_constant_newton,
                n_antidiffusive_contributions(i, j, dg))

    # negative xi direction
    if i > 1
        antidiffusive_flux = gamma * inverse_jacobian *
                             inverse_weights[i] *
                             get_node_vars(antidiffusive_flux1_R, equations, dg,
                                           i, j, element)
        newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                     final_check, equations, dt, limiter, antidiffusive_flux)
        isone(alpha[indices...]) && return nothing # Skip if alpha is already 1
    end

    # positive xi direction
    if i < nnodes(dg)
        antidiffusive_flux = -gamma * inverse_jacobian *
                             inverse_weights[i] *
                             get_node_vars(antidiffusive_flux1_L, equations, dg,
                                           i + 1, j, element)
        newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                     final_check, equations, dt, limiter, antidiffusive_flux)
        isone(alpha[indices...]) && return nothing # Skip if alpha is already 1
    end

    # negative eta direction
    if j > 1
        antidiffusive_flux = gamma * inverse_jacobian *
                             inverse_weights[j] *
                             get_node_vars(antidiffusive_flux2_R, equations, dg,
                                           i, j, element)
        newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                     final_check, equations, dt, limiter, antidiffusive_flux)
        isone(alpha[indices...]) && return nothing # Skip if alpha is already 1
    end

    # positive eta direction
    if j < nnodes(dg)
        antidiffusive_flux = -gamma * inverse_jacobian *
                             inverse_weights[j] *
                             get_node_vars(antidiffusive_flux2_L, equations, dg,
                                           i, j + 1, element)
        newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                     final_check, equations, dt, limiter, antidiffusive_flux)
    end

    return nothing
end

# Number of antidiffusive flux contributions to the update of the node `(i, j)`, i.e., the
# number of provisional states whose convex combination gives the new state. Since the bound is
# imposed on every provisional state separately, this is the factor the antidiffusive fluxes have to
# be scaled with. Nodes at an element boundary get fewer contributions than inner nodes because the
# flux across that boundary is the surface flux, which is not limited - unless that boundary is a
# mortar, which is limited as well.
@inline function n_antidiffusive_contributions(i, j, dg)
    return (i > 1) + (i < nnodes(dg)) + (j > 1) + (j < nnodes(dg))
end

###############################################################################
# IDP mortar limiting
###############################################################################

@inline function precompute_n_mortars_per_nodes!(volume_integral::AbstractVolumeIntegral,
                                                 dg, cache, mesh)
    return nothing
end
@inline function precompute_n_mortars_per_nodes!(volume_integral::VolumeIntegralAdaptive,
                                                 dg, cache, mesh)
    return precompute_n_mortars_per_nodes!(volume_integral.volume_integral_stabilized,
                                           dg, cache, mesh)
end
@inline function precompute_n_mortars_per_nodes!(volume_integral::VolumeIntegralSubcellLimiting,
                                                 dg, cache, mesh::TreeMesh{2})
    if !(dg.mortar isa LobattoLegendreMortarIDP)
        return nothing
    end

    (; n_mortars_per_node) = subcell_limiter_coefficients(volume_integral)
    (; neighbor_ids, orientations, large_sides) = cache.mortars

    n_mortars_per_node .= zero(eltype(n_mortars_per_node))

    for mortar in eachmortar(dg, cache)
        lower_element = neighbor_ids[1, mortar]
        upper_element = neighbor_ids[2, mortar]
        large_element = neighbor_ids[3, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            node_small = 1
            node_large = nnodes(dg)
        else # large_sides[mortar] == 2 -> small elements on left side
            node_small = nnodes(dg)
            node_large = 1
        end

        for i in eachnode(dg)
            if orientation == 1
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end

            n_mortars_per_node[indices_small..., lower_element] += 1
            n_mortars_per_node[indices_small..., upper_element] += 1
            n_mortars_per_node[indices_large..., large_element] += 1
        end
    end

    return nothing
end

###############################################################################
# Local minimum and maximum limiting of conservative variables

@inline function idp_mortar_local_twosided!(limiting_factor, u, dt, semi,
                                            mesh::TreeMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; variable_bounds, n_mortars_per_node) = subcell_limiter_coefficients(dg.volume_integral)
    variable_string = string(var_index)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

        # Set up correct direction and factors
        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            direction_small = 2 * orientation - 1
            direction_large = 2 * orientation
            node_small = 1
            node_large = nnodes(dg)

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            direction_small = 2 * orientation
            direction_large = 2 * orientation - 1
            node_small = nnodes(dg)
            node_large = 1

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        # Compute limiting factor
        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                # L2 mortars in y-direction
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end

            # Large element
            Q = zalesak_limiting_twosided(u, var_index, indices_large..., large_element,
                                          i, direction_large, factor_large, dt,
                                          var_min, var_max, n_mortars_per_node,
                                          mesh, cache)

            # Small elements
            for small_element_index in 1:2
                iszero(Q) && break # Skip if Q is zero, i.e., the limiting factor will be 1

                small_element = neighbor_ids[small_element_index, mortar]
                Q = min(Q,
                        zalesak_limiting_twosided(u, var_index, indices_small...,
                                                  small_element, i, direction_small,
                                                  factor_small, dt,
                                                  var_min, var_max, n_mortars_per_node,
                                                  mesh, cache))
            end

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)
        end
    end

    return nothing
end

# Zalesak-type limiting factor for one face node of one element adjacent to a mortar.
# Returns the admissible fraction `Q` of the antidiffusive surface flux, which keeps the
# solution within the local two-sided bounds. If the high-order flux is not finite, zero is
# returned, which results in pure low-order fluxes at this mortar.
#   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
#   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
#   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
#         for each mortar, not each node
@inline function zalesak_limiting_twosided(u, var_index, i_node, j_node, element,
                                           surface_node, direction, factor, dt,
                                           var_min, var_max, n_mortars_per_node,
                                           mesh, cache)
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    var = u[var_index, i_node, j_node, element]

    # Two-sided local bounds
    var_min_node = var_min[i_node, j_node, element]
    var_max_node = var_max[i_node, j_node, element]

    Qp = max(0, (var_max_node - var) / dt)
    Qm = min(0, (var_min_node - var) / dt)

    # Compute flux difference
    flux_high_order = surface_flux_values_high_order[var_index, surface_node,
                                                     direction, element]
    # Check if high-order flux is finite. Otherwise, use pure low-order fluxes.
    isfinite(flux_high_order) || return zero(Qp)
    flux_low_order = surface_flux_values[var_index, surface_node, direction, element]
    flux_difference = factor * (flux_high_order - flux_low_order)

    Pp = max(0, flux_difference)
    Pm = min(0, flux_difference)

    inverse_jacobian_node = get_inverse_jacobian(inverse_jacobian, mesh,
                                                 i_node, j_node, element)
    Pp = inverse_jacobian_node * Pp
    Pm = inverse_jacobian_node * Pm

    # A node can be on multiple mortars. Scale the antidiffusive flux contribution
    # to account for this. Similar to scaling with `gamma_constant_newton`.
    n_mortars = n_mortars_per_node[i_node, j_node, element]
    Pp = n_mortars * Pp
    Pm = n_mortars * Pm

    # Compute blending coefficient avoiding division by zero
    # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
    eps_ = eps(typeof(Qp)) * 100 * abs(var_max_node)
    Qp = abs(Qp) / (abs(Pp) + eps_)
    Qm = abs(Qm) / (abs(Pm) + eps_)

    return min(one(Qp), Qp, Qm)
end

##############################################################################
# Local minimum or maximum limiting of nonlinear variables

@inline function idp_mortar_local_onesided!(limiting_factor, u, dt, semi,
                                            mesh::TreeMesh{2}, variable,
                                            min_or_max)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            direction_small = 2 * orientation - 1
            direction_large = 2 * orientation
            node_small = 1
            node_large = nnodes(dg)

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            direction_small = 2 * orientation
            direction_large = 2 * orientation - 1
            node_small = nnodes(dg)
            node_large = 1

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1 (no limiting needed)

            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                # L2 mortars in y-direction
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end

            # Large element
            newton_loop_mortar!(limiting_factor, mortar, u,
                                indices_large..., large_element,
                                i, direction_large, factor_large, dt,
                                var_minmax, variable, min_or_max,
                                initial_check_local_onesided_newton_idp,
                                final_check_local_onesided_newton_idp,
                                mesh, equations, dg, cache)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            # Small elements
            for small_element_index in 1:2
                small_element = neighbor_ids[small_element_index, mortar]

                newton_loop_mortar!(limiting_factor, mortar, u,
                                    indices_small..., small_element,
                                    i, direction_small, factor_small, dt,
                                    var_minmax, variable, min_or_max,
                                    initial_check_local_onesided_newton_idp,
                                    final_check_local_onesided_newton_idp,
                                    mesh, equations, dg, cache)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            end
        end
    end

    return nothing
end

# One-sided Newton-bisection limiting for one face node of one element adjacent to a mortar.
# Updates the limiting factor of the mortar such that the state at this node stays within the
# bound stored in `var_bound`. If the high-order flux is not finite, the limiting factor is set
# to one, which results in pure low-order fluxes at this mortar.
#   Note: The limiting factor has to be computed, even if the state is valid, because the
#         correction is for each mortar, not each node
@inline function newton_loop_mortar!(limiting_factor, mortar, u,
                                     i_node, j_node, element,
                                     surface_node, direction, factor, dt,
                                     var_bound, variable, min_or_max,
                                     initial_check, final_check,
                                     mesh, equations, dg, cache)
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; limiter) = dg.mortar
    (; n_mortars_per_node) = limiter.cache.subcell_limiter_coefficients

    # The correction of the volume integral is already applied to `u`. The remaining antidiffusive fluxes at the mortars need to be scaled only with the number of mortars adjacent to the node, i.e., `n_mortars_per_node[i_node, j_node, element]`.
    gamma = min(limiter.gamma_constant_newton,
                n_mortars_per_node[i_node, j_node, element])

    flux_high_order = get_node_vars(surface_flux_values_high_order, equations, dg,
                                    surface_node, direction, element)
    # Check if high-order flux is finite. Otherwise, use pure low-order fluxes.
    if !all(isfinite, flux_high_order)
        limiting_factor[mortar] = 1
        return nothing
    end
    flux_low_order = get_node_vars(surface_flux_values, equations, dg,
                                   surface_node, direction, element)

    inverse_jacobian_node = get_inverse_jacobian(inverse_jacobian, mesh,
                                                 i_node, j_node, element)
    antidiffusive_flux = gamma * factor * inverse_jacobian_node *
                         (flux_high_order .- flux_low_order)

    u_node = get_node_vars(u, equations, dg, i_node, j_node, element)
    bound = var_bound[i_node, j_node, element]

    newton_loop!(limiting_factor, bound, u_node, (mortar,), variable, min_or_max,
                 initial_check, final_check,
                 equations, dt, limiter, antidiffusive_flux)

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables
@inline function idp_mortar_positivity_conservative!(limiting_factor, u, dt, semi,
                                                     mesh::TreeMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    (; n_mortars_per_node, variable_bounds) = subcell_limiter_coefficients(dg.volume_integral)

    # Check whether the local limiting already computed a bound for this variable in this stage.
    was_limited_locally = limiter.local_twosided &&
                          (var_index in limiter.local_twosided_variables_cons)
    # Without a smoothness indicator, both limiters are enforced completely and `var_min`
    # holds the more restrictive of the two bounds. With a smoothness indicator, local bounds are
    # only enforced fractionally, while positivity limiting is enforced completely.
    # In that case, the local bound is stored in `var_min`, while the positivity bound is stored in
    # `var_min_positivity`.
    enabled_indicator = !isnothing(limiter.indicator)

    # Array the positivity bound was written to. Only with a smoothness indicator it is stored
    # separately; otherwise the more restrictive of the two bounds is kept in `var_min`.
    if was_limited_locally && !enabled_indicator
        # Positivity bound was merged into var_min and therefore already enforced during local limiting.
        # Skip positivity limiting for this variable.
        return nothing
    elseif was_limited_locally && enabled_indicator
        var_min = variable_bounds[Symbol(string(var_index), "_min_positivity")]
    else
        var_min = variable_bounds[Symbol(string(var_index), "_min")]
    end

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

        # Set up correct direction and factors
        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            direction_small = 2 * orientation - 1
            direction_large = 2 * orientation
            node_small = 1
            node_large = nnodes(dg)

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            direction_small = 2 * orientation
            direction_large = 2 * orientation - 1
            node_small = nnodes(dg)
            node_large = 1

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        # Compute limiting factor
        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                # L2 mortars in y-direction
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end

            # Large element
            Q = zalesak_limiting_onesided(u, var_index, indices_large..., large_element,
                                          i, direction_large, factor_large, dt,
                                          var_min, n_mortars_per_node,
                                          mesh, cache)

            # Small elements
            for small_element_index in 1:2
                iszero(Q) && break # Skip if Q is zero, i.e., the limiting factor will be 1

                small_element = neighbor_ids[small_element_index, mortar]
                Q = min(Q,
                        zalesak_limiting_onesided(u, var_index, indices_small...,
                                                  small_element,
                                                  i, direction_small, factor_small, dt,
                                                  var_min, n_mortars_per_node,
                                                  mesh, cache))
            end

            # Calculate limiting factor
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)
        end
    end

    return nothing
end

# One-sided Zalesak-type limiting factor for one face node of one element adjacent to a
# mortar. Returns the admissible fraction `Q` of the antidiffusive surface flux, which keeps
# the solution above the positivity bound. If the high-order flux is not finite, zero is
# returned, which results in pure low-order fluxes at this mortar.
#   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
#   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
#   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
#         for each mortar, not each node
@inline function zalesak_limiting_onesided(u, var_index, i_node, j_node, element,
                                           surface_node, direction, factor, dt,
                                           var_min, n_mortars_per_node,
                                           mesh, cache)
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    var = u[var_index, i_node, j_node, element]

    # Minimum bound
    var_min_node = var_min[i_node, j_node, element]

    Qm = min(0, (var_min_node - var) / dt)

    # Compute flux difference
    flux_high_order = surface_flux_values_high_order[var_index, surface_node,
                                                     direction, element]
    # Check if high-order flux is finite. Otherwise, use pure low-order fluxes.
    isfinite(flux_high_order) || return zero(Qm)
    flux_low_order = surface_flux_values[var_index, surface_node, direction, element]
    flux_difference = factor * (flux_high_order - flux_low_order)

    Pm = min(0, flux_difference)

    inverse_jacobian_node = get_inverse_jacobian(inverse_jacobian, mesh,
                                                 i_node, j_node, element)
    Pm = inverse_jacobian_node * Pm

    # A node can be on multiple mortars. Scale the antidiffusive flux contribution
    # to account for this. Similar to scaling with `gamma_constant_newton`.
    Pm = n_mortars_per_node[i_node, j_node, element] * Pm

    # Compute blending coefficient avoiding division by zero
    # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
    eps_ = eps(typeof(Qm)) * 100
    Qm = abs(Qm) / (abs(Pm) + eps_)

    return min(one(Qm), Qm)
end

##############################################################################
# Global positivity limiting of nonlinear variables
@inline function idp_mortar_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                  mesh::TreeMesh{2}, variable)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars
    (; inverse_weights) = dg.basis

    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.mortar
    # The nonlinear positivity limiting is the only limiter writing this bound: the nonlinear
    # local limiting is only used for entropies, the nonlinear positivity limiting only for
    # the pressure. Therefore, `var_min` holds the positivity bound and can be reused here.
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1

        large_element = neighbor_ids[3, mortar]

        orientation = orientations[mortar]
        if large_sides[mortar] == 1 # -> small elements on right side
            direction_small = 2 * orientation - 1
            direction_large = 2 * orientation
            node_small = 1
            node_large = nnodes(dg)

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            direction_small = 2 * orientation
            direction_large = 2 * orientation - 1
            node_small = nnodes(dg)
            node_large = 1

            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        for i in eachnode(dg)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1 (no limiting needed)

            if orientation == 1
                # L2 mortars in x-direction
                indices_small = (node_small, i)
                indices_large = (node_large, i)
            else
                # L2 mortars in y-direction
                indices_small = (i, node_small)
                indices_large = (i, node_large)
            end

            # Large element
            newton_loop_mortar!(limiting_factor, mortar, u,
                                indices_large..., large_element,
                                i, direction_large, factor_large, dt,
                                var_min, variable, min,
                                initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp,
                                mesh, equations, dg, cache)
            isone(limiting_factor[mortar]) && break # Skip if alpha is already 1

            # Small elements
            for small_element_index in 1:2
                small_element = neighbor_ids[small_element_index, mortar]

                newton_loop_mortar!(limiting_factor, mortar, u,
                                    indices_small..., small_element,
                                    i, direction_small, factor_small, dt,
                                    var_min, variable, min,
                                    initial_check_nonnegative_newton_idp,
                                    final_check_nonnegative_newton_idp,
                                    mesh, equations, dg, cache)
                isone(limiting_factor[mortar]) && break # Skip if alpha is already 1
            end
        end
    end

    return nothing
end
end # @muladd
