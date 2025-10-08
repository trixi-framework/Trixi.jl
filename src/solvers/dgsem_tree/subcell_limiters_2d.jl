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
                                       u::AbstractArray{<:Any, 4}, t, semi, equations)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)
        var_min[:, :, element] .= typemax(eltype(var_min))
        var_max[:, :, element] .= typemin(eltype(var_max))
        # Calculate bounds at Gauss-Lobatto nodes using u
        for j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, element]
            var_min[i, j, element] = min(var_min[i, j, element], var)
            var_max[i, j, element] = max(var_max[i, j, element], var)

            if i > 1
                var_min[i - 1, j, element] = min(var_min[i - 1, j, element], var)
                var_max[i - 1, j, element] = max(var_max[i - 1, j, element], var)
            end
            if i < nnodes(dg)
                var_min[i + 1, j, element] = min(var_min[i + 1, j, element], var)
                var_max[i + 1, j, element] = max(var_max[i + 1, j, element], var)
            end
            if j > 1
                var_min[i, j - 1, element] = min(var_min[i, j - 1, element], var)
                var_max[i, j - 1, element] = max(var_max[i, j - 1, element], var)
            end
            if j < nnodes(dg)
                var_min[i, j + 1, element] = min(var_min[i, j + 1, element], var)
                var_max[i, j + 1, element] = max(var_max[i, j + 1, element], var)
            end
        end
    end

    # Values at element boundary
    calc_bounds_twosided_interface!(var_min, var_max, variable,
                                    u, t, semi, mesh, equations)
    return nothing
end

@inline function calc_bounds_twosided_interface!(var_min, var_max, variable,
                                                 u, t, semi, mesh::TreeMesh2D,
                                                 equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for i in eachnode(dg)
            index_left = (nnodes(dg), i)
            index_right = (1, i)
            if orientation == 2
                index_left = reverse(index_left)
                index_right = reverse(index_right)
            end
            var_left = u[variable, index_left..., left]
            var_right = u[variable, index_right..., right]

            var_min[index_right..., right] = min(var_min[index_right..., right],
                                                 var_left)
            var_max[index_right..., right] = max(var_max[index_right..., right],
                                                 var_left)

            var_min[index_left..., left] = min(var_min[index_left..., left], var_right)
            var_max[index_left..., left] = max(var_max[index_left..., left], var_right)
        end
    end

    # TODO: How to include values at mortar interfaces?
    # - For LobattoLegendreMortarIDP: include only values of nodes with nonnegative local weights
    # - For LobattoLegendreMortarL2: include all neighboring values (TODO?)
    for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        orientation = cache.mortars.orientations[mortar]

        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
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
            u_lower = get_node_vars(u, equations, dg, indices_small...,
                                    lower_element)
            u_upper = get_node_vars(u, equations, dg, indices_small...,
                                    upper_element)
            u_large = get_node_vars(u, equations, dg, indices_large...,
                                    large_element)
            var_lower = u_lower[variable]
            var_upper = u_upper[variable]
            var_large = u_large[variable]

            for j in eachnode(dg)
                if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                    if orientation == 1
                        # L2 mortars in x-direction
                        indices_small = (1, j)
                        indices_large = (nnodes(dg), j)
                    else
                        # L2 mortars in y-direction
                        indices_small = (j, 1)
                        indices_large = (j, nnodes(dg))
                    end
                else # large_sides[mortar] == 2 -> small elements on left side
                    if orientation == 1
                        # L2 mortars in x-direction
                        indices_small = (nnodes(dg), j)
                        indices_large = (1, j)
                    else
                        # L2 mortars in y-direction
                        indices_small = (j, nnodes(dg))
                        indices_large = (j, 1)
                    end
                end

                l2_mortars = dg.mortar isa LobattoLegendreMortarL2
                if l2_mortars || dg.mortar.mortar_weights[i, j] > 0
                    var_min[indices_small..., lower_element] = min(var_min[indices_small...,
                                                                           lower_element],
                                                                   var_large)
                    var_max[indices_small..., lower_element] = max(var_max[indices_small...,
                                                                           lower_element],
                                                                   var_large)
                end
                if l2_mortars || dg.mortar.mortar_weights[j, i] > 0
                    var_min[indices_large..., large_element] = min(var_min[indices_large...,
                                                                           large_element],
                                                                   var_lower)
                    var_max[indices_large..., large_element] = max(var_max[indices_large...,
                                                                           large_element],
                                                                   var_lower)
                end
                if l2_mortars || dg.mortar.mortar_weights[i, j + nnodes(dg)] > 0
                    var_min[indices_small..., upper_element] = min(var_min[indices_small...,
                                                                           upper_element],
                                                                   var_large)
                    var_max[indices_small..., upper_element] = max(var_max[indices_small...,
                                                                           upper_element],
                                                                   var_large)
                end
                if l2_mortars || dg.mortar.mortar_weights[j, i + nnodes(dg)] > 0
                    var_min[indices_large..., large_element] = min(var_min[indices_large...,
                                                                           large_element],
                                                                   var_upper)
                    var_max[indices_large..., large_element] = max(var_max[indices_large...,
                                                                           large_element],
                                                                   var_upper)
                end
            end
        end
    end

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                index = (1, i)
                boundary_index = 1
            else # Element is on the left, boundary on the right
                index = (nnodes(dg), i)
                boundary_index = 2
            end
            if orientation == 2
                index = reverse(index)
                boundary_index += 2
            end
            u_inner = get_node_vars(u, equations, dg, index..., element)
            u_outer = get_boundary_outer_state(u_inner, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg, cache,
                                               index..., element)
            var_outer = u_outer[variable]

            var_min[index..., element] = min(var_min[index..., element], var_outer)
            var_max[index..., element] = max(var_max[index..., element], var_outer)
        end
    end

    return nothing
end

@inline function calc_bounds_onesided!(var_minmax, min_or_max, variable,
                                       u::AbstractArray{<:Any, 4}, t, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)
        # Reset bounds
        for j in eachnode(dg), i in eachnode(dg)
            if min_or_max === max
                var_minmax[i, j, element] = typemin(eltype(var_minmax))
            else
                var_minmax[i, j, element] = typemax(eltype(var_minmax))
            end
        end

        # Calculate bounds at Gauss-Lobatto nodes using u
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

    # Values at element boundary
    calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u, t, semi, mesh)

    return nothing
end

@inline function calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u, t,
                                                 semi, mesh::TreeMesh2D)
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for i in eachnode(dg)
            index_left = (nnodes(dg), i)
            index_right = (1, i)
            if orientation == 2
                index_left = reverse(index_left)
                index_right = reverse(index_right)
            end
            var_left = variable(get_node_vars(u, equations, dg, index_left..., left),
                                equations)
            var_right = variable(get_node_vars(u, equations, dg, index_right..., right),
                                 equations)

            var_minmax[index_right..., right] = min_or_max(var_minmax[index_right...,
                                                                      right], var_left)
            var_minmax[index_left..., left] = min_or_max(var_minmax[index_left...,
                                                                    left], var_right)
        end
    end

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                index = (1, i)
                boundary_index = 1
            else # Element is on the left, boundary on the right
                index = (nnodes(dg), i)
                boundary_index = 2
            end
            if orientation == 2
                index = reverse(index)
                boundary_index += 2
            end
            u_inner = get_node_vars(u, equations, dg, index..., element)
            u_outer = get_boundary_outer_state(u_inner, t,
                                               boundary_conditions[boundary_index],
                                               orientation, boundary_index,
                                               mesh, equations, dg, cache,
                                               index..., element)
            var_outer = variable(u_outer, equations)

            var_minmax[index..., element] = min_or_max(var_minmax[index..., element],
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
    calc_bounds_twosided!(var_min, var_max, variable, u, t, semi, equations)

    @threaded for element in eachelement(dg, semi.cache)
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, element)
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

@inline function idp_local_onesided!(alpha, limiter, u::AbstractArray{<:Real, 4}, t, dt,
                                     semi, variable, min_or_max)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]
    calc_bounds_onesided!(var_minmax, min_or_max, variable, u, t, semi)

    # Perform Newton's bisection method to find new alpha
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
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
        for j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, element)
            var = u[variable, i, j, element]
            if var < 0
                error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
            end

            # Compute bound
            if limiter.local_twosided &&
               variable in limiter.local_twosided_variables_cons &&
               var_min[i, j, element] >= positivity_correction_factor * var
                # Local limiting is more restrictive that positivity limiting
                # => Skip positivity limiting for this node
                continue
            end
            var_min[i, j, element] = positivity_correction_factor * var

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
# IDP mortar limiting
###############################################################################

@inline function calc_mortar_limiting_factor!(u, semi, t, dt)
    mesh, _, solver, cache = mesh_equations_solver_cache(semi)
    (; positivity_variables_cons) = solver.mortar
    (; limiting_factor) = cache.mortars
    @trixi_timeit timer() "reset alpha" limiting_factor.=zeros(eltype(limiting_factor))

    @trixi_timeit timer() "conservative variables" for var_index in positivity_variables_cons
        limiting_positivity_conservative!(limiting_factor, u, dt, semi, mesh, var_index)
    end

    return nothing
end

###############################################################################
# Local two-sided limiting of conservative variables
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   mesh::TreeMesh{2}, var_index)
    _, _, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; boundary_interpolation) = dg.basis

    (; positivity_correction_factor) = dg.volume_integral.limiter

    for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Calc minimal low-order solution
        var_min_upper = typemax(eltype(surface_flux_values))
        var_min_lower = typemax(eltype(surface_flux_values))
        var_min_large = typemax(eltype(surface_flux_values))
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
            var_min_upper = min(var_min_upper, var_upper)
            var_min_lower = min(var_min_lower, var_lower)
            var_min_large = min(var_min_large, var_large)
        end
        var_min_upper = positivity_correction_factor * var_min_upper
        var_min_lower = positivity_correction_factor * var_min_lower
        var_min_large = positivity_correction_factor * var_min_large

        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                    direction_small = 1
                    direction_large = 2
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                    direction_small = 3
                    direction_large = 4
                end
                factor_small = boundary_interpolation[1, 1]
                factor_large = -boundary_interpolation[nnodes(dg), 2]
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                    direction_small = 2
                    direction_large = 1
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                    direction_small = 4
                    direction_large = 3
                end
                factor_large = boundary_interpolation[1, 1]
                factor_small = -boundary_interpolation[nnodes(dg), 2]
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.

            var_upper = u[var_index, indices_small..., upper_element]
            var_lower = u[var_index, indices_small..., lower_element]
            var_large = u[var_index, indices_large..., large_element]

            if min(var_upper, var_lower, var_large) < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
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

            # Calculate Pm
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
            flux_large_low_order = surface_flux_values_high_order[var_index, i,
                                                                  direction_large,
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
        end
    end

    return nothing
end

###############################################################################
# Newton-bisection method

# 2D version
@inline function newton_loops_alpha!(alpha, bound, u, i, j, element,
                                     variable, min_or_max,
                                     initial_check, final_check,
                                     inverse_jacobian, dt,
                                     equations::AbstractEquations{2},
                                     dg, cache, limiter)
    (; inverse_weights) = dg.basis
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    indices = (i, j, element)

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
end # @muladd
