# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# IDP Limiting
###############################################################################

@inline function calc_bounds_twosided_interface!(var_min, var_max, variable,
                                                 u, t, semi, mesh::TreeMesh3D,
                                                 equations)
    _, _, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for j in eachnode(dg), i in eachnode(dg)
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

    # Calc bounds at physical boundaries
    for boundary in eachboundary(dg, cache)
        element = cache.boundaries.neighbor_ids[boundary]
        orientation = cache.boundaries.orientations[boundary]
        neighbor_side = cache.boundaries.neighbor_sides[boundary]

        for j in eachnode(dg), i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                if orientation == 1 # boundary in x-direction
                    index = (1, i, j)
                    boundary_index = 1
                elseif orientation == 2 # boundary in y-direction
                    index = (i, 1, j)
                    boundary_index = 3
                else # orientation == 3 # boundary in z-direction
                    index = (i, j, 1)
                    boundary_index = 5
                end
            else # Element is on the left, boundary on the right
                if orientation == 1 # boundary in x-direction
                    index = (nnodes(dg), i, j)
                    boundary_index = 2
                elseif orientation == 2 # boundary in y-direction
                    index = (i, nnodes(dg), j)
                    boundary_index = 4
                else # orientation == 3 # boundary in z-direction
                    index = (i, j, nnodes(dg))
                    boundary_index = 6
                end
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

@inline function calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u, t,
                                                 semi, mesh::TreeMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi

    # Calc bounds at interfaces and periodic boundaries
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.neighbor_ids[1, interface]
        right = cache.interfaces.neighbor_ids[2, interface]

        orientation = cache.interfaces.orientations[interface]

        for j in eachnode(dg), i in eachnode(dg)
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

        for j in eachnode(dg), i in eachnode(dg)
            if neighbor_side == 2 # Element is on the right, boundary on the left
                if orientation == 1 # boundary in x-direction
                    index = (1, i, j)
                    boundary_index = 1
                elseif orientation == 2 # boundary in y-direction
                    index = (i, 1, j)
                    boundary_index = 3
                else # orientation == 3 # boundary in z-direction
                    index = (i, j, 1)
                    boundary_index = 5
                end
            else # Element is on the left, boundary on the right
                if orientation == 1 # boundary in x-direction
                    index = (nnodes(dg), i, j)
                    boundary_index = 2
                elseif orientation == 2 # boundary in y-direction
                    index = (i, nnodes(dg), j)
                    boundary_index = 4
                else # orientation == 3 # boundary in z-direction
                    index = (i, j, nnodes(dg))
                    boundary_index = 6
                end
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
end # @muladd
