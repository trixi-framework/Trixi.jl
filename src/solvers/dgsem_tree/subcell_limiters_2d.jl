# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# IDP Limiting
###############################################################################

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function create_cache(limiter::Type{SubcellLimiterIDP}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis, bound_keys)
    subcell_limiter_coefficients = Trixi.ContainerSubcellLimiterIDP2D{real(basis)}(0,
                                                                                   nnodes(basis),
                                                                                   bound_keys)

    # Memory for bounds checking routine with `BoundsCheckCallback`.
    # Local variable contains the maximum deviation since the last export.
    idp_bounds_delta_local = Dict{Symbol, real(basis)}()
    # Global variable contains the total maximum deviation.
    idp_bounds_delta_global = Dict{Symbol, real(basis)}()
    for key in bound_keys
        idp_bounds_delta_local[key] = zero(real(basis))
        idp_bounds_delta_global[key] = zero(real(basis))
    end

    return (; subcell_limiter_coefficients, idp_bounds_delta_local,
            idp_bounds_delta_global)
end

function (limiter::SubcellLimiterIDP)(u::AbstractArray{<:Any, 4},
                                      semi, equations, dg::DGSEM,
                                      t, dt;
                                      kwargs...)
    @unpack alpha = limiter.cache.subcell_limiter_coefficients
    # TODO: Do not abuse `reset_du!` but maybe implement a generic `set_zero!`
    @trixi_timeit timer() "reset alpha" reset_du!(alpha, dg, semi.cache)

    if limiter.local_twosided
        @trixi_timeit timer() "local twosided" idp_local_twosided!(alpha, limiter,
                                                                   u, t, dt, semi)
    end
    if limiter.positivity
        @trixi_timeit timer() "positivity" idp_positivity!(alpha, limiter, u, dt, semi)
    end
    if limiter.local_onesided
        @trixi_timeit timer() "local onesided" idp_local_onesided!(alpha, limiter,
                                                                   u, t, dt, semi)
    end

    # Calculate alpha1 and alpha2
    @unpack alpha1, alpha2 = limiter.cache.subcell_limiter_coefficients
    @threaded for element in eachelement(dg, semi.cache)
        for j in eachnode(dg), i in 2:nnodes(dg)
            alpha1[i, j, element] = max(alpha[i - 1, j, element], alpha[i, j, element])
        end
        for j in 2:nnodes(dg), i in eachnode(dg)
            alpha2[i, j, element] = max(alpha[i, j - 1, element], alpha[i, j, element])
        end
        alpha1[1, :, element] .= zero(eltype(alpha1))
        alpha1[nnodes(dg) + 1, :, element] .= zero(eltype(alpha1))
        alpha2[:, 1, element] .= zero(eltype(alpha2))
        alpha2[:, nnodes(dg) + 1, element] .= zero(eltype(alpha2))
    end

    return nothing
end

###############################################################################
# Calculation of local bounds using low-order FV solution

@inline function calc_bounds_twosided!(var_min, var_max, variable,
                                       u, t, semi, equations)
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
    # - For "alternative implementation" and "global factors" include all neighboring values
    # - For "local factors" include only values with nonnegative local weights
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
                alternative = dg.mortar isa LobattoLegendreMortarIDPAlternative
                include_all_values = l2_mortars || alternative ||
                                     !(dg.mortar.local_factor)
                if include_all_values || dg.mortar.local_mortar_weights[i, j] > 0
                    var_min[indices_small..., lower_element] = min(var_min[indices_small...,
                                                                           lower_element],
                                                                   var_large)
                    var_max[indices_small..., lower_element] = max(var_max[indices_small...,
                                                                           lower_element],
                                                                   var_large)
                end
                if include_all_values || dg.mortar.local_mortar_weights[j, i] > 0
                    var_min[indices_large..., large_element] = min(var_min[indices_large...,
                                                                           large_element],
                                                                   var_lower)
                    var_max[indices_large..., large_element] = max(var_max[indices_large...,
                                                                           large_element],
                                                                   var_lower)
                end
                if include_all_values ||
                   dg.mortar.local_mortar_weights[i, j + nnodes(dg)] > 0
                    var_min[indices_small..., upper_element] = min(var_min[indices_small...,
                                                                           upper_element],
                                                                   var_large)
                    var_max[indices_small..., upper_element] = max(var_max[indices_small...,
                                                                           upper_element],
                                                                   var_large)
                end
                if include_all_values ||
                   dg.mortar.local_mortar_weights[j, i + nnodes(dg)] > 0
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

@inline function calc_bounds_onesided!(var_minmax, min_or_max, variable, u, t, semi)
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
# Local two-sided limiting of conservative variables

@inline function idp_local_twosided!(alpha, limiter, u, t, dt, semi)
    for variable in limiter.local_twosided_variables_cons
        idp_local_twosided!(alpha, limiter, u, t, dt, semi, variable)
    end

    return nothing
end

@inline function idp_local_twosided!(alpha, limiter, u, t, dt, semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

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
# Local one-sided limiting of nonlinear variables

@inline function idp_local_onesided!(alpha, limiter, u, t, dt, semi)
    for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
        idp_local_onesided!(alpha, limiter, u, t, dt, semi, variable, min_or_max)
    end

    return nothing
end

@inline function idp_local_onesided!(alpha, limiter, u, t, dt, semi,
                                     variable, min_or_max)
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
# Global positivity limiting

@inline function idp_positivity!(alpha, limiter, u, dt, semi)
    # Conservative variables
    for variable in limiter.positivity_variables_cons
        @trixi_timeit timer() "conservative variables" idp_positivity_conservative!(alpha,
                                                                                    limiter,
                                                                                    u,
                                                                                    dt,
                                                                                    semi,
                                                                                    variable)
    end

    # Nonlinear variables
    for variable in limiter.positivity_variables_nonlinear
        @trixi_timeit timer() "nonlinear variables" idp_positivity_nonlinear!(alpha,
                                                                              limiter,
                                                                              u, dt,
                                                                              semi,
                                                                              variable)
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables

@inline function idp_positivity_conservative!(alpha, limiter, u, dt, semi, variable)
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

@inline function idp_positivity_nonlinear!(alpha, limiter, u, dt, semi, variable)
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
    (; positivity_variables_cons, positivity_variables_nonlinear) = semi.solver.mortar
    (; limiting_factor) = semi.cache.mortars
    limiting_factor .= zeros(eltype(limiting_factor))

    for var_index in positivity_variables_cons
        limiting_positivity_conservative!(limiting_factor, u, dt, semi, var_index)
    end

    for variable in positivity_variables_nonlinear
        limiting_positivity_nonlinear!(limiting_factor, u, dt, semi, variable)
    end

    # Provisional analysis of limiting factor (TODO)
    (; output_directory) = semi.solver.mortar
    if length(limiting_factor) > 0
        open(joinpath(output_directory, "mortar_limiting_factor.txt"), "a") do f
            print(f, t)
            print(f, ", ", minimum(limiting_factor), ", ", maximum(limiting_factor),
                  ", ", sum(limiting_factor) / length(limiting_factor))
            println(f)
        end
    else
        open(joinpath(output_directory, "mortar_limiting_factor.txt"), "a") do f
            print(f, t)
            print(f, ", ", 0.0, ", ", 0.0, ", ", 0.0)
            println(f)
        end
    end

    return nothing
end

###############################################################################
# Local two-sided limiting of conservative variables
@inline function limiting_positivity_conservative!(limiting_factor, u, dt, semi,
                                                   var_index)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

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

##############################################################################
# Local one-sided limiting of nonlinear variables
@inline function limiting_positivity_nonlinear!(limiting_factor, u, dt, semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; orientations) = cache.mortars
    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; boundary_interpolation) = dg.basis

    (; limiter) = dg.volume_integral
    (; positivity_correction_factor) = limiter

    for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

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
            var_lower = variable(u_lower, equations)
            u_upper = get_node_vars(u, equations, dg, indices_small..., upper_element)
            var_upper = variable(u_upper, equations)
            u_large = get_node_vars(u, equations, dg, indices_large..., large_element)
            var_large = variable(u_large, equations)
            if var_lower < 0 || var_upper < 0 || var_large < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end

            var_min_lower = positivity_correction_factor * var_lower
            var_min_upper = positivity_correction_factor * var_upper
            var_min_large = positivity_correction_factor * var_large

            # lower element
            flux_lower_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, direction_small,
                                                  lower_element)
            flux_lower_low_order = get_node_vars(surface_flux_values, equations, dg, i,
                                                 direction_small, lower_element)
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order .- flux_lower_low_order)
            antidiffusive_flux_lower = inverse_jacobian_lower * flux_difference_lower

            newton_loop!(limiting_factor, var_min_lower, u_lower, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_lower)

            # upper element
            flux_upper_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, direction_small,
                                                  upper_element)
            flux_upper_low_order = get_node_vars(surface_flux_values, equations, dg, i,
                                                 direction_small, upper_element)
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order .- flux_upper_low_order)
            antidiffusive_flux_upper = inverse_jacobian_upper * flux_difference_upper

            newton_loop!(limiting_factor, var_min_upper, u_upper, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_upper)

            # large element
            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, direction_large,
                                                  large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg, i,
                                                 direction_large, large_element)
            flux_difference_large = factor_large *
                                    (flux_large_high_order .- flux_large_low_order)
            antidiffusive_flux_large = inverse_jacobian_large * flux_difference_large

            newton_loop!(limiting_factor, var_min_large, u_large, (mortar,), variable,
                         min, initial_check_nonnegative_newton_idp,
                         final_check_nonnegative_newton_idp,
                         equations, dt, limiter, antidiffusive_flux_large)
        end
    end

    return nothing
end

###############################################################################
# Newton-bisection method

@inline function newton_loops_alpha!(alpha, bound, u, i, j, element, variable,
                                     min_or_max, initial_check, final_check,
                                     inverse_jacobian, dt,
                                     equations::AbstractEquations{2},
                                     dg, cache, limiter)
    (; inverse_weights) = dg.basis
    (; antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_R, equations, dg, i, j,
                                       element)
    newton_loop!(alpha, bound, u, (i, j, element), variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    # positive xi direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_L, equations, dg, i + 1, j,
                                       element)
    newton_loop!(alpha, bound, u, (i, j, element), variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    # negative eta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_R, equations, dg, i, j,
                                       element)
    newton_loop!(alpha, bound, u, (i, j, element), variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    # positive eta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_L, equations, dg, i, j + 1,
                                       element)
    newton_loop!(alpha, bound, u, (i, j, element), variable, min_or_max, initial_check,
                 final_check, equations, dt, limiter, antidiffusive_flux)

    return nothing
end

# TODO: Move to `subcell_limiters.jl` since it is dimension independent
@inline function newton_loop!(alpha, bound, u, alpha_indices, variable, min_or_max,
                              initial_check, final_check, equations, dt, limiter,
                              antidiffusive_flux)
    newton_reltol, newton_abstol = limiter.newton_tolerances

    beta = 1 - alpha[alpha_indices...]

    beta_L = 0 # alpha = 1
    beta_R = beta # No higher beta (lower alpha) than the current one

    u_curr = u + beta * dt * antidiffusive_flux

    # If state is valid, perform initial check and return if correction is not needed
    if isvalid(u_curr, equations)
        goal = goal_function_newton_idp(variable, bound, u_curr, equations)

        initial_check(min_or_max, bound, goal, newton_abstol) && return nothing
    end

    # Newton iterations
    for iter in 1:(limiter.max_iterations_newton)
        beta_old = beta

        # If the state is valid, evaluate d(goal)/d(beta)
        if isvalid(u_curr, equations)
            dgoal_dbeta = dgoal_function_newton_idp(variable, u_curr, dt,
                                                    antidiffusive_flux, equations)
        else # Otherwise, perform a bisection step
            dgoal_dbeta = 0
        end

        if dgoal_dbeta != 0
            # Update beta with Newton's method
            beta = beta - goal / dgoal_dbeta
        end

        # Check bounds
        if (beta < beta_L) || (beta > beta_R) || (dgoal_dbeta == 0) || isnan(beta)
            # Out of bounds, do a bisection step
            beta = 0.5f0 * (beta_L + beta_R)
            # Get new u
            u_curr = u + beta * dt * antidiffusive_flux

            # If the state is invalid, finish bisection step without checking tolerance and iterate further
            if !isvalid(u_curr, equations)
                beta_R = beta
                continue
            end

            # Check new beta for condition and update bounds
            goal = goal_function_newton_idp(variable, bound, u_curr, equations)
            if initial_check(min_or_max, bound, goal, newton_abstol)
                # New beta fulfills condition
                beta_L = beta
            else
                # New beta does not fulfill condition
                beta_R = beta
            end
        else
            # Get new u
            u_curr = u + beta * dt * antidiffusive_flux

            # If the state is invalid, redefine right bound without checking tolerance and iterate further
            if !isvalid(u_curr, equations)
                beta_R = beta
                continue
            end

            # Evaluate goal function
            goal = goal_function_newton_idp(variable, bound, u_curr, equations)
        end

        # Check relative tolerance
        if abs(beta_old - beta) <= newton_reltol
            break
        end

        # Check absolute tolerance
        if final_check(bound, goal, newton_abstol)
            break
        end
    end

    new_alpha = 1 - beta
    alpha[alpha_indices...] = new_alpha

    return nothing
end

### Auxiliary routines for Newton's bisection method ###
# Initial checks
@inline function initial_check_local_onesided_newton_idp(::typeof(min), bound,
                                                         goal, newton_abstol)
    goal <= max(newton_abstol, abs(bound) * newton_abstol)
end

@inline function initial_check_local_onesided_newton_idp(::typeof(max), bound,
                                                         goal, newton_abstol)
    goal >= -max(newton_abstol, abs(bound) * newton_abstol)
end

@inline initial_check_nonnegative_newton_idp(min_or_max, bound, goal, newton_abstol) = goal <=
                                                                                       0

# Goal and d(Goal)d(u) function
@inline goal_function_newton_idp(variable, bound, u, equations) = bound -
                                                                  variable(u, equations)
@inline function dgoal_function_newton_idp(variable, u, dt, antidiffusive_flux,
                                           equations)
    -dot(gradient_conservative(variable, u, equations), dt * antidiffusive_flux)
end

# Final checks
# final check for one-sided local limiting
@inline function final_check_local_onesided_newton_idp(bound, goal, newton_abstol)
    abs(goal) < max(newton_abstol, abs(bound) * newton_abstol)
end

# final check for nonnegativity limiting
@inline function final_check_nonnegative_newton_idp(bound, goal, newton_abstol)
    (goal <= eps()) && (goal > -max(newton_abstol, abs(bound) * newton_abstol))
end
end # @muladd
