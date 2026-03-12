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

    return nothing
end

@inline function calc_bounds_onesided_interface!(var_minmax, min_or_max, variable, u, t,
                                                 semi, mesh::TreeMesh{3})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)

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

    return nothing
end
end # @muladd
