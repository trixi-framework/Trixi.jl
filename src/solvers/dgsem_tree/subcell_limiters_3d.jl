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
                                       u::AbstractArray{<:Any, 5}, t, semi, equations)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)
        # Calculate bounds at Gauss-Lobatto nodes
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, k, element]
            var_min[i, j, k, element] = var
            var_max[i, j, k, element] = var
        end

        # Apply values in x direction
        for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
            var = u[variable, i - 1, j, k, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            var = u[variable, i, j, k, element]
            var_min[i - 1, j, k, element] = min(var_min[i - 1, j, k, element], var)
            var_max[i - 1, j, k, element] = max(var_max[i - 1, j, k, element], var)
        end

        # Apply values in y direction
        for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
            var = u[variable, i, j - 1, k, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            var = u[variable, i, j, k, element]
            var_min[i, j - 1, k, element] = min(var_min[i, j - 1, k, element], var)
            var_max[i, j - 1, k, element] = max(var_max[i, j - 1, k, element], var)
        end

        # Apply values in z direction
        for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, k - 1, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            var = u[variable, i, j, k, element]
            var_min[i, j, k - 1, element] = min(var_min[i, j, k - 1, element], var)
            var_max[i, j, k - 1, element] = max(var_max[i, j, k - 1, element], var)
        end
    end

    # Values at element boundary
    calc_bounds_twosided_interface!(var_min, var_max, variable,
                                    u, t, semi, mesh, equations)
    return nothing
end

@inline function calc_bounds_twosided_interface!(var_min, var_max, variable,
                                                 u, t, semi, mesh::TreeMesh3D,
                                                 equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices for left and right element based on the interface orientation
            if orientation == 1
                # interface in x-direction
                index_left = (nnodes(dg), i, j)
                index_right = (1, i, j)
            elseif orientation == 2
                # interface in y-direction
                index_left = (i, nnodes(dg), j)
                index_right = (i, 1, j)
            else # if orientation == 3
                # interface in z-direction
                index_left = (i, j, nnodes(dg))
                index_right = (i, j, 1)
            end
            var_left = u[variable, index_left..., left_element]
            var_right = u[variable, index_right..., right_element]

            var_min[index_right..., right_element] = min(var_min[index_right...,
                                                                 right_element],
                                                         var_left)
            var_max[index_right..., right_element] = max(var_max[index_right...,
                                                                 right_element],
                                                         var_left)

            var_min[index_left..., left_element] = min(var_min[index_left...,
                                                               left_element], var_right)
            var_max[index_left..., left_element] = max(var_max[index_left...,
                                                               left_element], var_right)
        end
    end

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices and boundary index based on the orientation and neighbor_side
            if neighbor_side == 2 # Element is on the right, boundary on the left
                if orientation == 1 # boundary in x-direction
                    node_index = (1, i, j)
                    boundary_index = 1
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, 1, j)
                    boundary_index = 3
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, 1)
                    boundary_index = 5
                end
            else # Element is on the left, boundary on the right
                if orientation == 1 # boundary in x-direction
                    node_index = (nnodes(dg), i, j)
                    boundary_index = 2
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, nnodes(dg), j)
                    boundary_index = 4
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, nnodes(dg))
                    boundary_index = 6
                end
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
                                       u::AbstractArray{<:Any, 5}, t, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements

    # The approach used in `calc_bounds_twosided!` is not used here because it requires more
    # evaluations of the variable and is therefore slower.

    @threaded for element in eachelement(dg, cache)
        # Reset bounds
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            if min_or_max === max
                var_minmax[i, j, k, element] = typemin(eltype(var_minmax))
            else
                var_minmax[i, j, k, element] = typemax(eltype(var_minmax))
            end
        end

        # Calculate bounds at Gauss-Lobatto nodes
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            var = variable(get_node_vars(u, equations, dg, i, j, k, element), equations)
            var_minmax[i, j, k, element] = min_or_max(var_minmax[i, j, k, element], var)

            if i > 1
                var_minmax[i - 1, j, k, element] = min_or_max(var_minmax[i - 1, j, k,
                                                                         element], var)
            end
            if i < nnodes(dg)
                var_minmax[i + 1, j, k, element] = min_or_max(var_minmax[i + 1, j, k,
                                                                         element], var)
            end
            if j > 1
                var_minmax[i, j - 1, k, element] = min_or_max(var_minmax[i, j - 1, k,
                                                                         element], var)
            end
            if j < nnodes(dg)
                var_minmax[i, j + 1, k, element] = min_or_max(var_minmax[i, j + 1, k,
                                                                         element], var)
            end
            if k > 1
                var_minmax[i, j, k - 1, element] = min_or_max(var_minmax[i, j, k - 1,
                                                                         element], var)
            end
            if k < nnodes(dg)
                var_minmax[i, j, k + 1, element] = min_or_max(var_minmax[i, j, k + 1,
                                                                         element], var)
            end
        end
    end

    # Values at element boundary
    calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u, t, semi, mesh)

    return nothing
end

@inline function calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u, t,
                                                 semi, mesh::TreeMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left_element = cache.interfaces.neighbor_ids[1, interface]
        right_element = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices for left and right element based on the interface orientation
            if orientation == 1
                # interface in x-direction
                index_left = (nnodes(dg), i, j)
                index_right = (1, i, j)
            elseif orientation == 2
                # interface in y-direction
                index_left = (i, nnodes(dg), j)
                index_right = (i, 1, j)
            else # if orientation == 3
                # interface in z-direction
                index_left = (i, j, nnodes(dg))
                index_right = (i, j, 1)
            end
            var_left = variable(get_node_vars(u, equations, dg, index_left...,
                                              left_element),
                                equations)
            var_right = variable(get_node_vars(u, equations, dg, index_right...,
                                               right_element),
                                 equations)

            var_minmax[index_right..., right_element] = min_or_max(var_minmax[index_right...,
                                                                              right_element],
                                                                   var_left)
            var_minmax[index_left..., left_element] = min_or_max(var_minmax[index_left...,
                                                                            left_element],
                                                                 var_right)
        end
    end

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for j in eachnode(dg), i in eachnode(dg)
            # Define node indices and boundary index based on the orientation and neighbor_side
            if neighbor_side == 2 # Element is on the right, boundary on the left
                if orientation == 1 # boundary in x-direction
                    node_index = (1, i, j)
                    boundary_index = 1
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, 1, j)
                    boundary_index = 3
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, 1)
                    boundary_index = 5
                end
            else # Element is on the left, boundary on the right
                if orientation == 1 # boundary in x-direction
                    node_index = (nnodes(dg), i, j)
                    boundary_index = 2
                elseif orientation == 2 # boundary in y-direction
                    node_index = (i, nnodes(dg), j)
                    boundary_index = 4
                else # orientation == 3 # boundary in z-direction
                    node_index = (i, j, nnodes(dg))
                    boundary_index = 6
                end
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

@inline function idp_local_twosided!(alpha, limiter, u::AbstractArray{<:Any, 5},
                                     t, dt, semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    variable_string = string(variable)
    var_min = variable_bounds[Symbol(variable_string, "_min")]
    var_max = variable_bounds[Symbol(variable_string, "_max")]
    calc_bounds_twosided!(var_min, var_max, variable, u, t, semi, equations)

    @threaded for element in eachelement(dg, semi.cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            var = u[variable, i, j, k, element]
            # Real Zalesak type limiter
            #   * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            #   * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            #   Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #         for each interface, not each node

            Qp = max(0, (var_max[i, j, k, element] - var) / dt)
            Qm = min(0, (var_min[i, j, k, element] - var) / dt)

            # Calculate Pp and Pm
            # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, k, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, k, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, k, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, k, element]
            val_flux3_local = inverse_weights[k] *
                              antidiffusive_flux3_R[variable, i, j, k, element]
            val_flux3_local_jp1 = -inverse_weights[k] *
                                  antidiffusive_flux3_L[variable, i, j, k + 1, element]

            Pp = max(0, val_flux1_local) + max(0, val_flux1_local_ip1) +
                 max(0, val_flux2_local) + max(0, val_flux2_local_jp1) +
                 max(0, val_flux3_local) + max(0, val_flux3_local_jp1)
            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1) +
                 min(0, val_flux3_local) + min(0, val_flux3_local_jp1)

            Pp = inverse_jacobian * Pp
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qp = abs(Qp) /
                 (abs(Pp) + eps(typeof(Qp)) * 100 * abs(var_max[i, j, k, element]))
            Qm = abs(Qm) /
                 (abs(Pm) + eps(typeof(Qm)) * 100 * abs(var_max[i, j, k, element]))

            # Calculate alpha at nodes
            alpha[i, j, k, element] = max(alpha[i, j, k, element], 1 - min(1, Qp, Qm))
        end
    end

    return nothing
end

##############################################################################
# Local one-sided limiting of nonlinear variables

@inline function idp_local_onesided!(alpha, limiter, u::AbstractArray{<:Real, 5},
                                     t, dt, semi,
                                     variable, min_or_max)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_minmax = variable_bounds[Symbol(string(variable), "_", string(min_or_max))]
    calc_bounds_onesided!(var_minmax, min_or_max, variable, u, t, semi)

    # Perform Newton's bisection method to find new alpha
    @threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            u_local = get_node_vars(u, equations, dg, i, j, k, element)
            newton_loops_alpha!(alpha, var_minmax[i, j, k, element],
                                u_local, i, j, k, element,
                                variable, min_or_max,
                                initial_check_local_onesided_newton_idp,
                                final_check_local_onesided_newton_idp,
                                inverse_jacobian, dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of conservative variables

@inline function idp_positivity_conservative!(alpha, limiter,
                                              u::AbstractArray{<:Real, 5}, dt, semi,
                                              variable)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis # Plays role of DG subcell sizes
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in eachelement(dg, semi.cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            var = u[variable, i, j, k, element]
            if var < 0
                error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
            end

            # Compute bound
            if limiter.local_twosided &&
               (variable in limiter.local_twosided_variables_cons) &&
               (var_min[i, j, k, element] >= positivity_correction_factor * var)
                # Local limiting is more restrictive that positivity limiting
                # => Skip positivity limiting for this node
                continue
            end
            var_min[i, j, k, element] = positivity_correction_factor * var

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each interface, not each node
            Qm = min(0, (var_min[i, j, k, element] - var) / dt)

            # Calculate Pm
            # Note: Boundaries of antidiffusive_flux1/2/3 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, k, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, k, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, k, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, k, element]
            val_flux3_local = inverse_weights[k] *
                              antidiffusive_flux3_R[variable, i, j, k, element]
            val_flux3_local_jp1 = -inverse_weights[k] *
                                  antidiffusive_flux3_L[variable, i, j, k + 1, element]

            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1) +
                 min(0, val_flux3_local) + min(0, val_flux3_local_jp1)
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

            # Calculate alpha
            alpha[i, j, k, element] = max(alpha[i, j, k, element], 1 - Qm)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of nonlinear variables

@inline function idp_positivity_nonlinear!(alpha, limiter,
                                           u::AbstractArray{<:Real, 5}, dt, semi,
                                           variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in eachelement(dg, semi.cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)

            # Compute bound
            u_local = get_node_vars(u, equations, dg, i, j, k, element)
            var = variable(u_local, equations)
            if var < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end
            var_min[i, j, k, element] = positivity_correction_factor * var

            # Perform Newton's bisection method to find new alpha
            newton_loops_alpha!(alpha, var_min[i, j, k, element],
                                u_local, i, j, k, element,
                                variable, min,
                                initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp,
                                inverse_jacobian, dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Newton-bisection method

@inline function newton_loops_alpha!(alpha, bound, u, i, j, k, element,
                                     variable, min_or_max,
                                     initial_check, final_check,
                                     inverse_jacobian, dt,
                                     equations::AbstractEquations{3},
                                     dg, cache, limiter)
    (; inverse_weights) = dg.basis # Plays role of inverse DG-subcell sizes
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    indices = (i, j, k, element)

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive xi direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_L, equations, dg,
                                       i + 1, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # negative eta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive eta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_L, equations, dg,
                                       i, j + 1, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # negative zeta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[k] *
                         get_node_vars(antidiffusive_flux3_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive zeta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[k] *
                         get_node_vars(antidiffusive_flux3_L, equations, dg,
                                       i, j, k + 1, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    return nothing
end
end # @muladd
