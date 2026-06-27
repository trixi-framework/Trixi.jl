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

    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

        if perform_subcell_limiting(dg.volume_integral, left_element) ||
           perform_subcell_limiting(dg.volume_integral, right_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this interface
        else
            # Subcell limiting is not necessary for both elements => Skip this interface
            continue
        end

        orientation = cache.interfaces.orientations[interface]

        for i in eachnode(dg)
            # Define node indices for left and right element based on the interface orientation
            if orientation == 1
                index_left = (nnodes(dg), i)
                index_right = (1, i)
            else # if orientation == 2
                index_left = (i, nnodes(dg))
                index_right = (i, 1)
            end

            if perform_subcell_limiting(dg.volume_integral, right_element)
                var_left = u[variable, index_left..., left_element]
                var_min[index_right..., right_element] = min(var_min[index_right...,
                                                                     right_element],
                                                             var_left)
                var_max[index_right..., right_element] = max(var_max[index_right...,
                                                                     right_element],
                                                             var_left)
            end

            if perform_subcell_limiting(dg.volume_integral, left_element)
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

    return nothing
end

@inline function calc_bounds_twosided_mortar!(var_min, var_max, variable, u,
                                              semi, mesh::TreeMesh2D)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    # TODO: How to include values at mortar interfaces?
    # - For LobattoLegendreMortarIDP: include only values of nodes with nonnegative local weights
    # - For LobattoLegendreMortarL2: include all neighboring values (TODO?)
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]

        orientation = orientations[mortar]

        for i in eachnode(dg)
            if large_sides[mortar] == 1 # -> small elements on right side
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
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
                if large_sides[mortar] == 1 # -> small elements on right side
                    if orientation == 1
                        # L2 mortars in x-direction
                        indices_small_inner = (1, j)
                        indices_large_inner = (nnodes(dg), j)
                    else
                        # L2 mortars in y-direction
                        indices_small_inner = (j, 1)
                        indices_large_inner = (j, nnodes(dg))
                    end
                else # large_sides[mortar] == 2 -> small elements on left side
                    if orientation == 1
                        # L2 mortars in x-direction
                        indices_small_inner = (nnodes(dg), j)
                        indices_large_inner = (1, j)
                    else
                        # L2 mortars in y-direction
                        indices_small_inner = (j, nnodes(dg))
                        indices_large_inner = (j, 1)
                    end
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
                                       u::AbstractArray{<:Any, 4}, t,
                                       semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    # The approach used in `calc_bounds_twosided!` is not used here because it requires more
    # evaluations of the variable and is therefore slower.

    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        # Reset bounds
        for j in eachnode(dg), i in eachnode(dg)
            if min_or_max === max
                var_minmax[i, j, element] = typemin(eltype(var_minmax))
            else
                var_minmax[i, j, element] = typemax(eltype(var_minmax))
            end
        end

        # Calculate bounds at Gauss-Lobatto nodes
        for j in eachnode(dg), i in eachnode(dg)
            var = variable(get_node_vars(u, equations, dg, i, j, element), equations)
            var_minmax[i, j, element] = min_or_max(var_minmax[i, j, element], var)

            if i > 1
                var_minmax[i - 1, j, element] = min_or_max(var_minmax[i - 1, j,
                                                                      element], var)
            end
            if i < nnodes(dg)
                var_minmax[i + 1, j, element] = min_or_max(var_minmax[i + 1, j,
                                                                      element], var)
            end
            if j > 1
                var_minmax[i, j - 1, element] = min_or_max(var_minmax[i, j - 1,
                                                                      element], var)
            end
            if j < nnodes(dg)
                var_minmax[i, j + 1, element] = min_or_max(var_minmax[i, j + 1,
                                                                      element], var)
            end
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

    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

        if perform_subcell_limiting(dg.volume_integral, left_element) ||
           perform_subcell_limiting(dg.volume_integral, right_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this interface
        else
            # Subcell limiting is not necessary for both elements => Skip this interface
            continue
        end

        orientation = cache.interfaces.orientations[interface]

        for i in eachnode(dg)
            # Define node indices for left and right element based on the interface orientation
            if orientation == 1
                index_left = (nnodes(dg), i)
                index_right = (1, i)
            else # if orientation == 2
                index_left = (i, nnodes(dg))
                index_right = (i, 1)
            end

            if perform_subcell_limiting(dg.volume_integral, right_element)
                u_left = get_node_vars(u, equations, dg, index_left..., left_element)
                var_left = variable(u_left, equations)
                var_minmax[index_right..., right_element] = min_or_max(var_minmax[index_right...,
                                                                                  right_element],
                                                                       var_left)
            end
            if perform_subcell_limiting(dg.volume_integral, left_element)
                u_right = get_node_vars(u, equations, dg, index_right..., right_element)
                var_right = variable(u_right, equations)
                var_minmax[index_left..., left_element] = min_or_max(var_minmax[index_left...,
                                                                                left_element],
                                                                     var_right)
            end
        end
    end

    return nothing
end

@inline function calc_bounds_onesided_mortar!(var_minmax, min_or_max, variable, u,
                                              semi, mesh::TreeMesh2D)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; neighbor_ids, orientations, large_sides) = cache.mortars

    # TODO: How to include values at mortar interfaces?
    # See comment above two-sided version
    l2_mortars = dg.mortar isa LobattoLegendreMortarL2
    for mortar in eachmortar(dg, cache)
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

        orientation = orientations[mortar]

        for i in eachnode(dg)
            if large_sides[mortar] == 1 # -> small elements on right side
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientation == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
            end
            u_lower = get_node_vars(u, equations, dg, indices_small..., lower_element)
            u_upper = get_node_vars(u, equations, dg, indices_small..., upper_element)
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)
            var_lower = variable(u_lower, equations)
            var_upper = variable(u_upper, equations)
            var_large = variable(u_large, equations)

            for j in eachnode(dg)
                if large_sides[mortar] == 1 # -> small elements on right side
                    if orientation == 1
                        # L2 mortars in x-direction
                        indices_small_inner = (1, j)
                        indices_large_inner = (nnodes(dg), j)
                    else
                        # L2 mortars in y-direction
                        indices_small_inner = (j, 1)
                        indices_large_inner = (j, nnodes(dg))
                    end
                else # large_sides[mortar] == 2 -> small elements on left side
                    if orientation == 1
                        # L2 mortars in x-direction
                        indices_small_inner = (nnodes(dg), j)
                        indices_large_inner = (1, j)
                    else
                        # L2 mortars in y-direction
                        indices_small_inner = (j, nnodes(dg))
                        indices_large_inner = (j, 1)
                    end
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
            isone(alpha[i, j, element]) && continue # Skip if alpha is already 1 (no limiting needed)

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
            Qp = abs(Qp) /
                 (abs(Pp) + eps(typeof(Qp)) * 100 * abs(var_max[i, j, element]))
            Qm = abs(Qm) /
                 (abs(Pm) + eps(typeof(Qm)) * 100 * abs(var_max[i, j, element]))

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
            isone(alpha[i, j, element]) && continue # Skip if alpha is already 1 (no limiting needed)

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

    @threaded for element in eachelement(dg, semi.cache)

        # detect if subcell limiting is necessary
        perform_subcell_limiting(dg.volume_integral, element) || continue

        for j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, element]
            if var < 0
                error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
            end

            # Compute bound
            if limiter.local_twosided &&
               (variable in limiter.local_twosided_variables_cons) &&
               (var_min[i, j, element] >= positivity_correction_factor * var)
                # Local limiting is more restrictive that positivity limiting
                # => Skip positivity limiting for this node
                continue
            end
            var_min[i, j, element] = positivity_correction_factor * var

            isone(alpha[i, j, element]) && continue # Skip if alpha is already 1 (no limiting needed)

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each interface, not each node
            Qm = min(0, (var_min[i, j, element] - var) / dt)

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
            Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

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

    (; gamma_constant_newton) = limiter

    indices = (i, j, element)
    isone(alpha[indices...]) && return # Skip if alpha is already 1 (no limiting needed)

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_R, equations, dg,
                                       i, j, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    # positive xi direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_L, equations, dg,
                                       i + 1, j, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    # negative eta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_R, equations, dg,
                                       i, j, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    # positive eta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_L, equations, dg,
                                       i, j + 1, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    return nothing
end

###############################################################################
# IDP mortar limiting
###############################################################################

@inline function precompute_n_mortars_per_nodes!(volume_integral::AbstractVolumeIntegral,
                                                 dg, cache, mesh)
    return nothing
end
@inline function precompute_n_mortars_per_nodes!(volume_integral::VolumeIntegralSubcellLimiting,
                                                 dg, cache, mesh::TreeMesh{2})
    if !(dg.mortar isa LobattoLegendreMortarIDP)
        return nothing
    end

    (; n_mortars_per_node) = volume_integral.limiter.cache.subcell_limiter_coefficients
    (; neighbor_ids, orientations, large_sides) = cache.mortars

    n_mortars_per_node .= zero(eltype(n_mortars_per_node))

    for mortar in eachmortar(dg, cache)
        lower_element = neighbor_ids[1, mortar]
        upper_element = neighbor_ids[2, mortar]
        large_element = neighbor_ids[3, mortar]

        for i in eachnode(dg)
            if large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
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

@inline function limiting_local_conservative!(limiting_factor, u, dt, semi,
                                              mesh::TreeMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.volume_integral
    if !(var_index in limiter.local_twosided_variables_cons)
        error("Conservative variable $var_index is not included in local_twosided_variables_cons in the volume integral. So, the bounds are not computed before.")
    end
    (; variable_bounds, n_mortars_per_node) = limiter.cache.subcell_limiter_coefficients
    variable_string = string(var_index)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]

    for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1 (no limiting needed)

        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, lower_element) ||
           perform_subcell_limiting(dg.volume_integral, upper_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

        # Set up correct direction and factors
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            else
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            else
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        # Compute limiting factor
        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
            end
            var_upper = u[var_index, indices_small..., upper_element]
            var_lower = u[var_index, indices_small..., lower_element]
            var_large = u[var_index, indices_large..., large_element]

            if min(var_upper, var_lower, var_large) < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
            end

            Qp_lower = max(0,
                           (var_max[indices_small..., lower_element] - var_lower) / dt)
            Qm_lower = min(0,
                           (var_min[indices_small..., lower_element] - var_lower) / dt)

            Qp_upper = max(0,
                           (var_max[indices_small..., upper_element] - var_upper) / dt)
            Qm_upper = min(0,
                           (var_min[indices_small..., upper_element] - var_upper) / dt)

            Qp_large = max(0,
                           (var_max[indices_large..., large_element] - var_large) / dt)
            Qm_large = min(0,
                           (var_min[indices_large..., large_element] - var_large) / dt)

            # Compute flux differences
            flux_lower_high_order = surface_flux_values_high_order[var_index, i,
                                                                   direction_small,
                                                                   lower_element]
            flux_lower_low_order = surface_flux_values[var_index, i, direction_small,
                                                       lower_element]
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order - flux_lower_low_order)

            flux_upper_high_order = surface_flux_values_high_order[var_index, i,
                                                                   direction_small,
                                                                   upper_element]
            flux_upper_low_order = surface_flux_values[var_index, i, direction_small,
                                                       upper_element]
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order - flux_upper_low_order)

            flux_large_high_order = surface_flux_values_high_order[var_index, i,
                                                                   direction_large,
                                                                   large_element]
            flux_large_low_order = surface_flux_values[var_index, i, direction_large,
                                                       large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            # Use pure low-order fluxes if high-order fluxes are not finite.
            if !isfinite(flux_lower_high_order) ||
               !isfinite(flux_upper_high_order) ||
               !isfinite(flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end

            inverse_jacobian_upper = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_small...,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_small...,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)

            Pp_upper = max(0, flux_difference_upper)
            Pm_upper = min(0, flux_difference_upper)
            Pp_upper = inverse_jacobian_upper * Pp_upper
            Pm_upper = inverse_jacobian_upper * Pm_upper

            Pp_lower = max(0, flux_difference_lower)
            Pm_lower = min(0, flux_difference_lower)
            Pp_lower = inverse_jacobian_lower * Pp_lower
            Pm_lower = inverse_jacobian_lower * Pm_lower

            Pp_large = max(0, flux_difference_large)
            Pm_large = min(0, flux_difference_large)
            Pp_large = inverse_jacobian_large * Pp_large
            Pm_large = inverse_jacobian_large * Pm_large

            # A node can be on multiple mortars. Scale the antidiffusive flux contribution
            # to account for this. Similar to scaling with `gamma_constant_newton`.
            n_mortars_upper = n_mortars_per_node[indices_small..., upper_element]
            n_mortars_lower = n_mortars_per_node[indices_small..., lower_element]
            n_mortars_large = n_mortars_per_node[indices_large..., large_element]
            Pp_upper *= n_mortars_upper
            Pm_upper *= n_mortars_upper
            Pp_lower *= n_mortars_lower
            Pm_lower *= n_mortars_lower
            Pp_large *= n_mortars_large
            Pm_large *= n_mortars_large

            Qp_upper = abs(Qp_upper) /
                       (abs(Pp_upper) +
                        eps(typeof(Qp_upper)) * 100 *
                        abs(var_max[indices_small..., upper_element]))
            Qm_upper = abs(Qm_upper) /
                       (abs(Pm_upper) +
                        eps(typeof(Qm_upper)) * 100 *
                        abs(var_max[indices_small..., upper_element]))

            Qp_lower = abs(Qp_lower) /
                       (abs(Pp_lower) +
                        eps(typeof(Qp_lower)) * 100 *
                        abs(var_max[indices_small..., lower_element]))
            Qm_lower = abs(Qm_lower) /
                       (abs(Pm_lower) +
                        eps(typeof(Qm_lower)) * 100 *
                        abs(var_max[indices_small..., lower_element]))

            Qp_large = abs(Qp_large) /
                       (abs(Pp_large) +
                        eps(typeof(Qp_large)) * 100 *
                        abs(var_max[indices_large..., large_element]))
            Qm_large = abs(Qm_large) /
                       (abs(Pm_large) +
                        eps(typeof(Qm_large)) * 100 *
                        abs(var_max[indices_large..., large_element]))

            # Calculate limiting factor
            Q = min(1, Qp_upper, Qm_upper, Qp_lower, Qm_lower, Qp_large, Qm_large)
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Q)
        end
    end

    return nothing
end

##############################################################################
# Local minimum or maximum limiting of nonlinear variables

@inline function limiting_local_nonlinear!(limiting_factor, u, dt, semi,
                                           mesh::TreeMesh{2}, variable,
                                           min_or_max)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.volume_integral
    if !((variable, min_or_max) in limiter.local_onesided_variables_nonlinear)
        error("Nonlinear variable $variable with bound $min_or_max is not included in local_onesided_variables_nonlinear in the volume integral. So, the bounds are not computed before.")
    end
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]

    (; gamma_constant_newton) = limiter

    for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1 (no limiting needed)

        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, lower_element) ||
           perform_subcell_limiting(dg.volume_integral, upper_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            else
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            else
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
            end

            inverse_jacobian_upper = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_small...,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_small...,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_large...,
                                                          large_element)

            u_lower = get_node_vars(u, equations, dg, indices_small..., lower_element)
            u_upper = get_node_vars(u, equations, dg, indices_small..., upper_element)
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)

            bound_lower = var_minmax[indices_small..., lower_element]
            bound_upper = var_minmax[indices_small..., upper_element]
            bound_large = var_minmax[indices_large..., large_element]

            # large element
            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, direction_large, large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, direction_large, large_element)
            if !all(isfinite, flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end
            antidiffusive_flux_large = gamma_constant_newton * factor_large *
                                       inverse_jacobian_large *
                                       (flux_large_high_order .- flux_large_low_order)

            newton_loop!(limiting_factor, bound_large, u_large, (mortar,), variable,
                         min_or_max, initial_check_local_onesided_newton_idp,
                         final_check_local_onesided_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)

            # lower element
            flux_lower_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, direction_small, lower_element)
            flux_lower_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, direction_small, lower_element)
            if !all(isfinite, flux_lower_high_order)
                limiting_factor[mortar] = 1
                break
            end
            antidiffusive_flux_lower = gamma_constant_newton * factor_small *
                                       inverse_jacobian_lower *
                                       (flux_lower_high_order .- flux_lower_low_order)

            newton_loop!(limiting_factor, bound_lower, u_lower, (mortar,), variable,
                         min_or_max, initial_check_local_onesided_newton_idp,
                         final_check_local_onesided_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_lower)

            # upper element
            flux_upper_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, direction_small, upper_element)
            flux_upper_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, direction_small, upper_element)
            if !all(isfinite, flux_upper_high_order)
                limiting_factor[mortar] = 1
                break
            end
            antidiffusive_flux_upper = gamma_constant_newton * factor_small *
                                       inverse_jacobian_upper *
                                       (flux_upper_high_order .- flux_upper_low_order)

            newton_loop!(limiting_factor, bound_upper, u_upper, (mortar,), variable,
                         min_or_max, initial_check_local_onesided_newton_idp,
                         final_check_local_onesided_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_upper)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   mesh::TreeMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values, inverse_jacobian) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.volume_integral
    if !(var_index in limiter.local_twosided_variables_cons ||
         var_index in limiter.positivity_variables_cons)
        error("Conservative variable $var_index is not included to the limiting in the volume integral. So, the bounds are not computed before.")
    end
    (; variable_bounds, n_mortars_per_node) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(var_index), "_min")]

    for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1 (no limiting needed)

        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, lower_element) ||
           perform_subcell_limiting(dg.volume_integral, upper_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

        # Set up correct direction and factors
        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            else
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            else
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        # Compute limiting factor
        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
            end
            var_upper = u[var_index, indices_small..., upper_element]
            var_lower = u[var_index, indices_small..., lower_element]
            var_large = u[var_index, indices_large..., large_element]

            if min(var_upper, var_lower, var_large) < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
            end

            # Minimum bound
            var_min_lower = var_min[indices_small..., lower_element]
            var_min_upper = var_min[indices_small..., upper_element]
            var_min_large = var_min[indices_large..., large_element]

            # Compute flux differences
            flux_lower_high_order = surface_flux_values_high_order[var_index, i,
                                                                   direction_small,
                                                                   lower_element]
            flux_lower_low_order = surface_flux_values[var_index, i, direction_small,
                                                       lower_element]
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order - flux_lower_low_order)

            flux_upper_high_order = surface_flux_values_high_order[var_index, i,
                                                                   direction_small,
                                                                   upper_element]
            flux_upper_low_order = surface_flux_values[var_index, i, direction_small,
                                                       upper_element]
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order - flux_upper_low_order)

            flux_large_high_order = surface_flux_values_high_order[var_index, i,
                                                                   direction_large,
                                                                   large_element]
            flux_large_low_order = surface_flux_values[var_index, i, direction_large,
                                                       large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            # Use pure low-order fluxes if high-order fluxes are not finite.
            if !isfinite(flux_lower_high_order) ||
               !isfinite(flux_upper_high_order) ||
               !isfinite(flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end

            inverse_jacobian_upper = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_small...,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_small...,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(inverse_jacobian, mesh,
                                                          indices_large...,
                                                          large_element)

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each mortar, not each node
            Qm_upper = min(0, var_min_upper - var_upper)
            Qm_lower = min(0, var_min_lower - var_lower)
            Qm_large = min(0, var_min_large - var_large)

            Pm_upper = min(0, flux_difference_upper)
            Pm_lower = min(0, flux_difference_lower)
            Pm_large = min(0, flux_difference_large)

            # A node can be on multiple mortars. Scale the antidiffusive flux contribution
            # to account for this. Similar to scaling with `gamma_constant_newton`.
            n_mortars_upper = n_mortars_per_node[indices_small..., upper_element]
            n_mortars_lower = n_mortars_per_node[indices_small..., lower_element]
            n_mortars_large = n_mortars_per_node[indices_large..., large_element]
            Pm_upper *= n_mortars_upper
            Pm_lower *= n_mortars_lower
            Pm_large *= n_mortars_large

            Pm_upper = dt * inverse_jacobian_upper * Pm_upper
            Pm_lower = dt * inverse_jacobian_lower * Pm_lower
            Pm_large = dt * inverse_jacobian_large * Pm_large

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm_upper = abs(Qm_upper) / (abs(Pm_upper) + eps(typeof(Qm_upper)) * 100)
            Qm_lower = abs(Qm_lower) / (abs(Pm_lower) + eps(typeof(Qm_lower)) * 100)
            Qm_large = abs(Qm_large) / (abs(Pm_large) + eps(typeof(Qm_large)) * 100)

            # Calculate limiting factor
            Qm = min(1, Qm_upper, Qm_lower, Qm_large)
            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Qm)
        end
    end

    return nothing
end

##############################################################################
# Global positivity limiting of nonlinear variables
@inline function limiting_positivity_nonlinear!(limiting_factor, u, dt, semi,
                                                mesh::TreeMesh{2}, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    (; limiter) = dg.volume_integral
    if !(variable in limiter.local_onesided_variables_nonlinear ||
         variable in limiter.positivity_variables_nonlinear)
        error("Variable $variable is not included to the limiting in the volume integral. So, the bounds are not computed before.")
    end
    (; variable_bounds) = dg.volume_integral.limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    (; gamma_constant_newton) = limiter

    for mortar in eachmortar(dg, cache)
        isone(limiting_factor[mortar]) && continue # Skip if alpha is already 1 (no limiting needed)

        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        if perform_subcell_limiting(dg.volume_integral, large_element) ||
           perform_subcell_limiting(dg.volume_integral, lower_element) ||
           perform_subcell_limiting(dg.volume_integral, upper_element)
            # Subcell limiting is necessary for at least one of the elements => Calculate bounds at this mortar
        else
            # Subcell limiting is not necessary for all elements => Skip this mortar
            continue
        end

        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 1
                direction_large = 2
            else
                # L2 mortars in y-direction
                direction_small = 3
                direction_large = 4
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                # L2 mortars in x-direction
                direction_small = 2
                direction_large = 1
            else
                # L2 mortars in y-direction
                direction_small = 4
                direction_large = 3
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
            end

            u_lower = get_node_vars(u, equations, dg, indices_small..., lower_element)
            u_upper = get_node_vars(u, equations, dg, indices_small..., upper_element)
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)
            var_lower = variable(u_lower, equations)
            var_upper = variable(u_upper, equations)
            var_large = variable(u_large, equations)
            if var_lower < 0 || var_upper < 0 || var_large < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end

            # Minimum bound
            var_min_lower = var_min[indices_small..., lower_element]
            var_min_upper = var_min[indices_small..., upper_element]
            var_min_large = var_min[indices_large..., large_element]

            inverse_jacobian_upper = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_small...,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_small...,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_large...,
                                                          large_element)

            # large element
            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, direction_large, large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, direction_large, large_element)
            if !all(isfinite, flux_large_high_order)
                limiting_factor[mortar] = 1
                break
            end
            antidiffusive_flux_large = gamma_constant_newton * factor_large *
                                       inverse_jacobian_large *
                                       (flux_large_high_order .- flux_large_low_order)

            newton_loop!(limiting_factor, var_min_large, u_large, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)

            # lower element
            flux_lower_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, direction_small, lower_element)
            flux_lower_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, direction_small, lower_element)
            # If high-order fluxes are non-finite, disable mortar correction.
            # This mirrors the conservative mortar limiter behavior.
            if !all(isfinite, flux_lower_high_order)
                limiting_factor[mortar] = 1
                break
            end
            antidiffusive_flux_lower = gamma_constant_newton * factor_small *
                                       inverse_jacobian_lower *
                                       (flux_lower_high_order .- flux_lower_low_order)

            newton_loop!(limiting_factor, var_min_lower, u_lower, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_lower)

            # upper element
            flux_upper_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, direction_small, upper_element)
            flux_upper_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, direction_small, upper_element)
            if !all(isfinite, flux_upper_high_order)
                limiting_factor[mortar] = 1
                break
            end
            antidiffusive_flux_upper = gamma_constant_newton * factor_small *
                                       inverse_jacobian_upper *
                                       (flux_upper_high_order .- flux_upper_low_order)

            newton_loop!(limiting_factor, var_min_upper, u_upper, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_upper)
        end
    end

    return nothing
end
end # @muladd
