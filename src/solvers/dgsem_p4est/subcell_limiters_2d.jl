# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_bounds_twosided_interface!(var_min, var_max, variable, u, t, semi,
                                         mesh::P4estMesh{2})
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
                                                   equations, dg, cache,
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

    (; neighbor_ids, node_indices) = cache.interfaces
    index_range = eachnode(dg)
    index_end = last(index_range)

    # Calc bounds at interfaces and periodic boundaries
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
                                                   equations, dg, cache,
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
end # @muladd
