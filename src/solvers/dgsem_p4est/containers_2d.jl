# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize data structures in element container
function init_elements!(elements,
                        mesh::Union{P4estMesh{2}, P4estMeshView{2}, T8codeMesh{2}},
                        basis::LobattoLegendreBasis)
    @unpack node_coordinates, jacobian_matrix,
    contravariant_vectors, inverse_jacobian = elements

    calc_node_coordinates!(node_coordinates, mesh, basis)

    for element in 1:ncells(mesh)
        calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

        calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

        calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
    end

    return nothing
end

# Interpolate tree_node_coordinates to each quadrant at the nodes of the specified basis
function calc_node_coordinates!(node_coordinates,
                                mesh::Union{P4estMesh{2}, P4estMeshView{2},
                                            T8codeMesh{2}},
                                basis::LobattoLegendreBasis)
    # Hanging nodes will cause holes in the mesh if its polydeg is higher
    # than the polydeg of the solver.
    @assert length(basis.nodes)>=length(mesh.nodes) "The solver can't have a lower polydeg than the mesh"

    calc_node_coordinates!(node_coordinates, mesh, basis.nodes)
end

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{2, NDIMS_AMBIENT},
                                nodes::AbstractVector) where {NDIMS_AMBIENT}
    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(mesh),
                       StaticInt(NDIMS_AMBIENT), static_length(nodes),
                       static_length(mesh.nodes))
    matrix1 = StrideArray(undef, real(mesh),
                          static_length(nodes), static_length(mesh.nodes))
    matrix2 = similar(matrix1)
    baryweights_in = barycentric_weights(mesh.nodes)

    # Macros from `p4est`
    p4est_root_len = 1 << P4EST_MAXLEVEL
    p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

    trees = unsafe_wrap_sc(p4est_tree_t, mesh.p4est.trees)

    for tree in eachindex(trees)
        offset = trees[tree].quadrants_offset
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree].quadrants)

        for i in eachindex(quadrants)
            element = offset + i
            quad = quadrants[i]

            quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

            nodes_out_x = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.x / p4est_root_len) .- 1
            nodes_out_y = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
                           quad.y / p4est_root_len) .- 1
            polynomial_interpolation_matrix!(matrix1, mesh.nodes, nodes_out_x,
                                             baryweights_in)
            polynomial_interpolation_matrix!(matrix2, mesh.nodes, nodes_out_y,
                                             baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, element),
                                    matrix1, matrix2,
                                    view(mesh.tree_node_coordinates, :, :, :, tree),
                                    tmp1)
        end
    end

    return node_coordinates
end

# Initialize node_indices of interface container
@inline function init_interface_node_indices!(interfaces::P4estInterfaceContainer{2},
                                              faces, orientation, interface_id)
    # Iterate over primary and secondary element
    for side in 1:2
        # Align interface in positive coordinate direction of primary element.
        # For orientation == 1, the secondary element needs to be indexed backwards
        # relative to the interface.
        if side == 1 || orientation == 0
            # Forward indexing
            i = :i_forward
        else
            # Backward indexing
            i = :i_backward
        end

        if faces[side] == 0
            # Index face in negative x-direction
            interfaces.node_indices[side, interface_id] = (:begin, i)
        elseif faces[side] == 1
            # Index face in positive x-direction
            interfaces.node_indices[side, interface_id] = (:end, i)
        elseif faces[side] == 2
            # Index face in negative y-direction
            interfaces.node_indices[side, interface_id] = (i, :begin)
        else # faces[side] == 3
            # Index face in positive y-direction
            interfaces.node_indices[side, interface_id] = (i, :end)
        end
    end

    return interfaces
end

# Initialize node_indices of boundary container
@inline function init_boundary_node_indices!(boundaries::P4estBoundaryContainer{2},
                                             face, boundary_id)
    if face == 0
        # Index face in negative x-direction
        boundaries.node_indices[boundary_id] = (:begin, :i_forward)
    elseif face == 1
        # Index face in positive x-direction
        boundaries.node_indices[boundary_id] = (:end, :i_forward)
    elseif face == 2
        # Index face in negative y-direction
        boundaries.node_indices[boundary_id] = (:i_forward, :begin)
    else # face == 3
        # Index face in positive y-direction
        boundaries.node_indices[boundary_id] = (:i_forward, :end)
    end

    return boundaries
end

# Initialize node_indices of mortar container
# faces[1] is expected to be the face of the small side.
@inline function init_mortar_node_indices!(mortars, faces, orientation, mortar_id)
    for side in 1:2
        # Align mortar in positive coordinate direction of small side.
        # For orientation == 1, the large side needs to be indexed backwards
        # relative to the mortar.
        if side == 1 || orientation == 0
            # Forward indexing for small side or orientation == 0
            i = :i_forward
        else
            # Backward indexing for large side with reversed orientation
            i = :i_backward
        end

        if faces[side] == 0
            # Index face in negative x-direction
            mortars.node_indices[side, mortar_id] = (:begin, i)
        elseif faces[side] == 1
            # Index face in positive x-direction
            mortars.node_indices[side, mortar_id] = (:end, i)
        elseif faces[side] == 2
            # Index face in negative y-direction
            mortars.node_indices[side, mortar_id] = (i, :begin)
        else # faces[side] == 3
            # Index face in positive y-direction
            mortars.node_indices[side, mortar_id] = (i, :end)
        end
    end

    return mortars
end

# Initialize auxiliary surface node variables (2D implementation)
# follows prolong2interfaces
function init_aux_surface_node_vars!(aux_vars, mesh::P4estMesh{2},
                                     equations, solver, cache)
    @unpack aux_node_vars, aux_surface_node_vars = aux_vars
    @unpack interfaces = cache
    index_range = eachnode(solver)

    @threaded for interface in eachinterface(solver, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        for i in index_range
            for v in axes(aux_surface_node_vars, 2)
                aux_surface_node_vars[1, v, i, interface] = aux_node_vars[v,
                                                                                      i_primary,
                                                                                      j_primary,
                                                                                      primary_element]
            end
            i_primary += i_primary_step
            j_primary += j_primary_step
        end

        # Copy solution data from the secondary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        secondary_element = interfaces.neighbor_ids[2, interface]
        secondary_indices = interfaces.node_indices[2, interface]

        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        for i in index_range
            for v in axes(aux_surface_node_vars, 2)
                aux_surface_node_vars[2, v, i, interface] = aux_node_vars[v,
                                                                                      i_secondary,
                                                                                      j_secondary,
                                                                                      secondary_element]
            end
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end
    return nothing
end

# Initialize auxiliary boundary node variables
# 2D P4est implementation, similar to prolong2boundaries
function init_aux_boundary_node_vars!(aux_vars, mesh::P4estMesh{2},
                                     equations, solver, cache)
    @unpack aux_node_vars, aux_boundary_node_vars = aux_vars
    @unpack neighbor_ids, node_indices = cache.boundaries
    index_range = eachnode(solver)

    @threaded for boundary in eachboundary(solver, cache)
        # Copy solution data from the element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        element = neighbor_ids[boundary]
        node_index = node_indices[boundary]

        i_node_start, i_node_step = index_to_start_step_2d(node_index[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_index[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for i in eachnode(solver)
            for v in axes(aux_boundary_node_vars, 2)
                aux_boundary_node_vars[1, v, i, boundary] = aux_node_vars[v, i_node, j_node, element]
            end
            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return nothing
end

# Initialize auxiliary mortar node variables
# 2D P4est implementation, similar to prolong2mortars
# Each mortar has two sides (indentified by first variable of u_upper / u_lower)
# On the side with two small elements, values can be copied from the aux vars field
# On the side with one large element, values are usually interpolated to small elements
# We do this differently here and use the same small element values on both side. This
# assumes that the aux_field computes a smooth variable field with no jumps
function init_aux_mortar_node_vars!(aux_vars, mesh::P4estMesh{2}, equations, solver,
                                    cache)
    @unpack aux_node_vars, aux_mortar_node_vars = aux_vars
    @unpack fstar_tmp_threaded = cache
    @unpack neighbor_ids, node_indices = cache.mortars
    index_range = eachnode(solver)

    @threaded for mortar in eachmortar(solver, cache)
        # Copy solution data from the small elements using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)
        
        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]
            for i in eachnode(solver)
                for v in axes(aux_mortar_node_vars, 2)
                    aux_mortar_node_vars[:, v, position, i, mortar] .=
                            aux_node_vars[v, i_small, j_small, element]
                end
                for v in eachvariable(equations)
                    cache.mortars.u[1, v, position, i, mortar] = u[v, i_small, j_small,
                                                                   element]
                end
                i_small += i_small_step
                j_small += j_small_step
            end
        end
    end
    return nothing
end

# Initialize auxiliary MPI interface node variables
# 2D TreeMesh implementation, similar to prolong2mpiinterfaces
# However we directly assign to both sides, assuming the aux field had no jumps. Therefore
# we do not need any exchange.
function init_aux_mpiinterface_node_vars!(aux_vars, mesh::ParallelP4estMesh{2},
                                          equations,
                                          solver, cache)
    @unpack aux_node_vars, aux_mpiinterface_node_vars = aux_vars
    @unpack mpi_interfaces = cache
    index_range = eachnode(solver)

    @threaded for interface in eachmpiinterface(dsolverg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_element = mpi_interfaces.local_neighbor_ids[interface]
        local_indices = mpi_interfaces.node_indices[interface]

        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start
        for i in eachnode(solver)
            for v in axes(aux_mpiinterface_node_vars, 2)
                aux_mpiinterface_node_vars[:, v, i, interface] .= aux_node_vars[v, i_element,
                                                                  j_element,
                                                                  local_element]
            end
            i_element += i_element_step
            j_element += j_element_step
        end
    end
    return nothing
end

# Initialize auxiliary MPI mortar node variables
# 2D P4est implementation, similar to prolong2mpimortars
# However: - We only assign the small element values (only leftright = 1 is used)
#          - These have to be communicated
function init_aux_mpimortar_node_vars!(aux_vars, mesh::ParallelP4estMesh{2}, equations,
                                       solver, cache)
    @unpack aux_node_vars, aux_mpimortar_node_vars = aux_vars
    @unpack node_indices = cache.mpi_mortars
    index_range = eachnode(solver)

    @threaded for mortar in eachmpimortar(solver, cache)
        local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

        # Get start value and step size for indices on both sides to get the correct face
        # and orientation
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position in (1, 2) # small element
                # Copy solution data from the small elements
                i_small = i_small_start
                j_small = j_small_start
                for i in eachnode(solver)
                    for v in axes(aux_mpimortar_node_vars, 2)
                        aux_mpimortar_node_vars[1, v, position, i, mortar] = aux_node_vars[v, i_small, j_small, element]
                    end
                    i_small += i_small_step
                    j_small += j_small_step
                end
            end
        end
    end

    data_size = nnodes(solver) * n_aux_node_vars(equations)
    exchange_aux_mpimortars!(aux_mpimortar_node_vars, cache, data_size)
    return nothing
end
end # @muladd
