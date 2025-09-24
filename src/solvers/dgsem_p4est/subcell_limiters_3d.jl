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
function create_cache(limiter::Type{SubcellLimiterIDP}, equations::AbstractEquations{3},
                      basis::LobattoLegendreBasis, bound_keys)
    subcell_limiter_coefficients = Trixi.ContainerSubcellLimiterIDP3D{real(basis)}(0,
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

###############################################################################
# Calculation of local bounds using low-order FV solution

@inline function calc_bounds_twosided!(var_min, var_max, variable,
                                       u::AbstractArray{<:Any, 5}, t, semi, equations)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements
    @threaded for element in eachelement(dg, cache)
        var_min[:, :, :, element] .= typemax(eltype(var_min))
        var_max[:, :, :, element] .= typemin(eltype(var_max))
        # Calculate bounds at Gauss-Lobatto nodes using u
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, k, element]
            var_min[i, j, k, element] = min(var_min[i, j, k, element], var)
            var_max[i, j, k, element] = max(var_max[i, j, k, element], var)

            if i > 1
                var_min[i - 1, j, k, element] = min(var_min[i - 1, j, k, element], var)
                var_max[i - 1, j, k, element] = max(var_max[i - 1, j, k, element], var)
            end
            if i < nnodes(dg)
                var_min[i + 1, j, k, element] = min(var_min[i + 1, j, k, element], var)
                var_max[i + 1, j, k, element] = max(var_max[i + 1, j, k, element], var)
            end
            if j > 1
                var_min[i, j - 1, k, element] = min(var_min[i, j - 1, k, element], var)
                var_max[i, j - 1, k, element] = max(var_max[i, j - 1, k, element], var)
            end
            if j < nnodes(dg)
                var_min[i, j + 1, k, element] = min(var_min[i, j + 1, k, element], var)
                var_max[i, j + 1, k, element] = max(var_max[i, j + 1, k, element], var)
            end
            if k > 1
                var_min[i, j, k - 1, element] = min(var_min[i, j, k - 1, element], var)
                var_max[i, j, k - 1, element] = max(var_max[i, j, k - 1, element], var)
            end
            if k < nnodes(dg)
                var_min[i, j, k + 1, element] = min(var_min[i, j, k + 1, element], var)
                var_max[i, j, k + 1, element] = max(var_max[i, j, k + 1, element], var)
            end
        end
    end

    # Values at element boundary
    calc_bounds_twosided_interface!(var_min, var_max, variable,
                                    u, t, semi, mesh, equations)
    return nothing
end

function calc_bounds_twosided_interface!(var_min, var_max, variable,
                                         u, t, semi, mesh::P4estMesh{3}, equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        # Create the local i,j,k indexing
        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                var_primary = u[variable, i_primary, j_primary, k_primary,
                                primary_element]
                var_secondary = u[variable, i_secondary, j_secondary, k_secondary,
                                  secondary_element]

                var_min[i_primary, j_primary, k_primary, primary_element] = min(var_min[i_primary,
                                                                                        j_primary,
                                                                                        k_primary,
                                                                                        primary_element],
                                                                                var_secondary)
                var_max[i_primary, j_primary, k_primary, primary_element] = max(var_max[i_primary,
                                                                                        j_primary,
                                                                                        k_primary,
                                                                                        primary_element],
                                                                                var_secondary)

                var_min[i_secondary, j_secondary, k_secondary, secondary_element] = min(var_min[i_secondary,
                                                                                                j_secondary,
                                                                                                k_secondary,
                                                                                                secondary_element],
                                                                                        var_primary)
                var_max[i_secondary, j_secondary, k_secondary, secondary_element] = max(var_max[i_secondary,
                                                                                                j_secondary,
                                                                                                k_secondary,
                                                                                                secondary_element],
                                                                                        var_primary)

                # Increment the primary element indices
                i_primary += i_primary_step_i
                j_primary += j_primary_step_i
                k_primary += k_primary_step_i
                # Increment the secondary element surface indices
                i_secondary += i_secondary_step_i
                j_secondary += j_secondary_step_i
                k_secondary += k_secondary_step_i
            end
            # Increment the primary element indices
            i_primary += i_primary_step_j
            j_primary += j_primary_step_j
            k_primary += k_primary_step_j
            # Increment the secondary element surface indices
            i_secondary += i_secondary_step_j
            j_secondary += j_secondary_step_j
            k_secondary += k_secondary_step_j
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
                                                mesh::Union{TreeMesh{3}, P4estMesh{3}},
                                                equations, dg, cache)
    return nothing
end

@inline function calc_bounds_twosided_boundary!(var_min, var_max, variable, u, t,
                                                boundary_conditions,
                                                mesh::P4estMesh{3},
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

            i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                                index_range)
            j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                                index_range)
            k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                                index_range)

            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(direction,
                                                            contravariant_vectors,
                                                            i_node, j_node, k_node,
                                                            element)

                    u_inner = get_node_vars(u, equations, dg, i_node, j_node, k_node,
                                            element)

                    u_outer = get_boundary_outer_state(u_inner, t, boundary_condition,
                                                       normal_direction,
                                                       mesh, equations, dg, cache,
                                                       i_node, j_node, k_node, element)
                    var_outer = u_outer[variable]

                    var_min[i_node, j_node, k_node, element] = min(var_min[i_node,
                                                                           j_node,
                                                                           k_node,
                                                                           element],
                                                                   var_outer)
                    var_max[i_node, j_node, k_node, element] = max(var_max[i_node,
                                                                           j_node,
                                                                           k_node,
                                                                           element],
                                                                   var_outer)

                    i_node += i_node_step_i
                    j_node += j_node_step_i
                    k_node += k_node_step_i
                end
                i_node += i_node_step_j
                j_node += j_node_step_j
                k_node += k_node_step_j
            end
        end
    end

    return nothing
end

@inline function calc_bounds_onesided!(var_minmax, min_or_max, variable,
                                       u::AbstractArray{<:Any, 5}, t, semi)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    # Calc bounds inside elements

    @threaded for element in eachelement(dg, cache)
        # Reset bounds
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            if min_or_max === max
                var_minmax[i, j, k, element] = typemin(eltype(var_minmax))
            else
                var_minmax[i, j, k, element] = typemax(eltype(var_minmax))
            end
        end

        # Calculate bounds at Gauss-Lobatto nodes using u
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

function calc_bounds_onesided_interface!(var_minmax, minmax, variable, u, t, semi,
                                         mesh::P4estMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        # Create the local i,j,k indexing
        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                var_primary = variable(get_node_vars(u, equations, dg, i_primary,
                                                     j_primary, k_primary,
                                                     primary_element), equations)
                var_secondary = variable(get_node_vars(u, equations, dg, i_secondary,
                                                       j_secondary, k_secondary,
                                                       secondary_element),
                                         equations)

                var_minmax[i_primary, j_primary, k_primary, primary_element] = minmax(var_minmax[i_primary,
                                                                                                 j_primary,
                                                                                                 k_primary,
                                                                                                 primary_element],
                                                                                      var_secondary)
                var_minmax[i_secondary, j_secondary, k_secondary, secondary_element] = minmax(var_minmax[i_secondary,
                                                                                                         j_secondary,
                                                                                                         k_secondary,
                                                                                                         secondary_element],
                                                                                              var_primary)

                # Increment the primary element indices
                i_primary += i_primary_step_i
                j_primary += j_primary_step_i
                k_primary += k_primary_step_i
                # Increment the secondary element surface indices
                i_secondary += i_secondary_step_i
                j_secondary += j_secondary_step_i
                k_secondary += k_secondary_step_i
            end
            # Increment the primary element indices
            i_primary += i_primary_step_j
            j_primary += j_primary_step_j
            k_primary += k_primary_step_j
            # Increment the secondary element surface indices
            i_secondary += i_secondary_step_j
            j_secondary += j_secondary_step_j
            k_secondary += k_secondary_step_j
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
                                                mesh::P4estMesh{3},
                                                equations, dg, cache)
    return nothing
end

@inline function calc_bounds_onesided_boundary!(var_minmax, minmax, variable, u, t,
                                                boundary_conditions,
                                                mesh::P4estMesh{3},
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

            i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                                index_range)
            j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                                index_range)
            k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                                index_range)

            i_node = i_node_start
            j_node = j_node_start
            k_node = k_node_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(direction,
                                                            contravariant_vectors,
                                                            i_node, j_node, k_node,
                                                            element)

                    u_inner = get_node_vars(u, equations, dg, i_node, j_node, k_node,
                                            element)

                    u_outer = get_boundary_outer_state(u_inner, t, boundary_condition,
                                                       normal_direction,
                                                       mesh, equations, dg, cache,
                                                       i_node, j_node, k_node, element)
                    var_outer = variable(u_outer, equations)

                    var_minmax[i_node, j_node, k_node, element] = minmax(var_minmax[i_node,
                                                                                    j_node,
                                                                                    k_node,
                                                                                    element],
                                                                         var_outer)

                    i_node += i_node_step_i
                    j_node += j_node_step_i
                    k_node += k_node_step_i
                end
                i_node += i_node_step_j
                j_node += j_node_step_j
                k_node += k_node_step_j
            end
        end
    end

    return nothing
end

###############################################################################
# Local two-sided limiting of conservative variables

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
            newton_loops_alpha!(alpha, var_minmax[i, j, k, element], u_local,
                                i, j, k, element, variable, min_or_max,
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

@inline function idp_positivity_nonlinear!(alpha, limiter, u::AbstractArray{<:Real, 5},
                                           dt, semi, variable)
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
            newton_loops_alpha!(alpha, var_min[i, j, k, element], u_local, i, j, k,
                                element,
                                variable, min, initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp, inverse_jacobian,
                                dt, equations, dg, cache, limiter)
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
                                     equations, dg, cache, limiter)
    (; inverse_weights) = dg.basis
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    indices = (i, j, k, element)

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[i] *
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
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[j] *
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
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian * inverse_weights[j] *
                         get_node_vars(antidiffusive_flux3_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive zeta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux3_L, equations, dg,
                                       i, j, k + 1, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)
    return nothing
end
end # @muladd
