# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_bounds_twosided_interface!(var_min, var_max, variable, u, t, semi,
                                         mesh::StructuredMesh{2})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    (; contravariant_vectors) = cache.elements

    # Calc bounds at interfaces and periodic boundaries
    for element in eachelement(dg, cache)
        # Get neighboring element ids
        left = cache.elements.left_neighbors[1, element]
        lower = cache.elements.left_neighbors[2, element]

        if left != 0
            for j in eachnode(dg)
                var_left = u[variable, nnodes(dg), j, left]
                var_element = u[variable, 1, j, element]

                var_min[1, j, element] = min(var_min[1, j, element], var_left)
                var_max[1, j, element] = max(var_max[1, j, element], var_left)

                var_min[nnodes(dg), j, left] = min(var_min[nnodes(dg), j, left],
                                                   var_element)
                var_max[nnodes(dg), j, left] = max(var_max[nnodes(dg), j, left],
                                                   var_element)
            end
        end
        if lower != 0
            for i in eachnode(dg)
                var_lower = u[variable, i, nnodes(dg), lower]
                var_element = u[variable, i, 1, element]

                var_min[i, 1, element] = min(var_min[i, 1, element], var_lower)
                var_max[i, 1, element] = max(var_max[i, 1, element], var_lower)

                var_min[i, nnodes(dg), lower] = min(var_min[i, nnodes(dg), lower],
                                                    var_element)
                var_max[i, nnodes(dg), lower] = max(var_max[i, nnodes(dg), lower],
                                                    var_element)
            end
        end
    end

    # Calc bounds at physical boundaries
    if isperiodic(mesh)
        return nothing
    end
    linear_indices = LinearIndices(size(mesh))
    if !isperiodic(mesh, 1)
        # - xi direction
        for cell_y in axes(mesh, 2)
            element = linear_indices[begin, cell_y]
            for j in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, 1, j, element)
                u_inner = get_node_vars(u, equations, dg, 1, j, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[1], Ja1, 1,
                                                   equations, dg, cache,
                                                   1, j, element)
                var_outer = u_outer[variable]

                var_min[1, j, element] = min(var_min[1, j, element], var_outer)
                var_max[1, j, element] = max(var_max[1, j, element], var_outer)
            end
        end
        # + xi direction
        for cell_y in axes(mesh, 2)
            element = linear_indices[end, cell_y]
            for j in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, nnodes(dg), j,
                                               element)
                u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[2], Ja1, 2,
                                                   equations, dg, cache,
                                                   nnodes(dg), j, element)
                var_outer = u_outer[variable]

                var_min[nnodes(dg), j, element] = min(var_min[nnodes(dg), j, element],
                                                      var_outer)
                var_max[nnodes(dg), j, element] = max(var_max[nnodes(dg), j, element],
                                                      var_outer)
            end
        end
    end
    if !isperiodic(mesh, 2)
        # - eta direction
        for cell_x in axes(mesh, 1)
            element = linear_indices[cell_x, begin]
            for i in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, 1, element)
                u_inner = get_node_vars(u, equations, dg, i, 1, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[3], Ja2, 3,
                                                   equations, dg, cache,
                                                   i, 1, element)
                var_outer = u_outer[variable]

                var_min[i, 1, element] = min(var_min[i, 1, element], var_outer)
                var_max[i, 1, element] = max(var_max[i, 1, element], var_outer)
            end
        end
        # - eta direction
        for cell_x in axes(mesh, 1)
            element = linear_indices[cell_x, end]
            for i in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, nnodes(dg),
                                               element)
                u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[4], Ja2, 4,
                                                   equations, dg, cache,
                                                   i, nnodes(dg), element)
                var_outer = u_outer[variable]

                var_min[i, nnodes(dg), element] = min(var_min[i, nnodes(dg), element],
                                                      var_outer)
                var_max[i, nnodes(dg), element] = max(var_max[i, nnodes(dg), element],
                                                      var_outer)
            end
        end
    end

    return nothing
end

function calc_bounds_onesided_interface!(var_minmax, minmax, variable, u, t, semi,
                                         mesh::StructuredMesh{2})
    _, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; boundary_conditions) = semi
    (; contravariant_vectors) = cache.elements

    # Calc bounds at interfaces and periodic boundaries
    for element in eachelement(dg, cache)
        # Get neighboring element ids
        left = cache.elements.left_neighbors[1, element]
        lower = cache.elements.left_neighbors[2, element]

        if left != 0
            for j in eachnode(dg)
                var_left = variable(get_node_vars(u, equations, dg, nnodes(dg), j,
                                                  left), equations)
                var_element = variable(get_node_vars(u, equations, dg, 1, j, element),
                                       equations)

                var_minmax[1, j, element] = minmax(var_minmax[1, j, element], var_left)
                var_minmax[nnodes(dg), j, left] = minmax(var_minmax[nnodes(dg), j,
                                                                    left], var_element)
            end
        end
        if lower != 0
            for i in eachnode(dg)
                var_lower = variable(get_node_vars(u, equations, dg, i, nnodes(dg),
                                                   lower), equations)
                var_element = variable(get_node_vars(u, equations, dg, i, 1, element),
                                       equations)

                var_minmax[i, 1, element] = minmax(var_minmax[i, 1, element], var_lower)
                var_minmax[i, nnodes(dg), lower] = minmax(var_minmax[i, nnodes(dg),
                                                                     lower],
                                                          var_element)
            end
        end
    end

    # Calc bounds at physical boundaries
    if isperiodic(mesh)
        return nothing
    end
    linear_indices = LinearIndices(size(mesh))
    if !isperiodic(mesh, 1)
        # - xi direction
        for cell_y in axes(mesh, 2)
            element = linear_indices[begin, cell_y]
            for j in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, 1, j, element)
                u_inner = get_node_vars(u, equations, dg, 1, j, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[1], Ja1, 1,
                                                   equations, dg, cache,
                                                   1, j, element)
                var_outer = variable(u_outer, equations)

                var_minmax[1, j, element] = minmax(var_minmax[1, j, element], var_outer)
            end
        end
        # + xi direction
        for cell_y in axes(mesh, 2)
            element = linear_indices[end, cell_y]
            for j in eachnode(dg)
                Ja1 = get_contravariant_vector(1, contravariant_vectors, nnodes(dg), j,
                                               element)
                u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[2], Ja1, 2,
                                                   equations, dg, cache,
                                                   nnodes(dg), j, element)
                var_outer = variable(u_outer, equations)

                var_minmax[nnodes(dg), j, element] = minmax(var_minmax[nnodes(dg), j,
                                                                       element],
                                                            var_outer)
            end
        end
    end
    if !isperiodic(mesh, 2)
        # - eta direction
        for cell_x in axes(mesh, 1)
            element = linear_indices[cell_x, begin]
            for i in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, 1, element)
                u_inner = get_node_vars(u, equations, dg, i, 1, element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[3], Ja2, 3,
                                                   equations, dg, cache,
                                                   i, 1, element)
                var_outer = variable(u_outer, equations)

                var_minmax[i, 1, element] = minmax(var_minmax[i, 1, element], var_outer)
            end
        end
        # + eta direction
        for cell_x in axes(mesh, 1)
            element = linear_indices[cell_x, end]
            for i in eachnode(dg)
                Ja2 = get_contravariant_vector(2, contravariant_vectors, i, nnodes(dg),
                                               element)
                u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
                u_outer = get_boundary_outer_state(u_inner, t,
                                                   boundary_conditions[4], Ja2, 4,
                                                   equations, dg, cache,
                                                   i, nnodes(dg), element)
                var_outer = variable(u_outer, equations)

                var_minmax[i, nnodes(dg), element] = minmax(var_minmax[i, nnodes(dg),
                                                                       element],
                                                            var_outer)
            end
        end
    end

    return nothing
end
end # @muladd
