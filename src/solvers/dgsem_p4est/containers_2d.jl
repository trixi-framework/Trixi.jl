# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize data structures in element container
function init_elements!(elements,
                        mesh::Union{P4estMesh{2}, P4estMeshView{2}, T8codeMesh{2}},
                        basis::AbstractBasisSBP)
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
                                basis::AbstractBasisSBP)
    # Hanging nodes will cause holes in the mesh if its polydeg is higher
    # than the polydeg of the solver.
    @assert length(basis.nodes)>=length(mesh.nodes) "The solver can't have a lower polydeg than the mesh"

    return calc_node_coordinates!(node_coordinates, mesh, basis.nodes)
end

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{2, NDIMS_AMBIENT},
                                nodes::AbstractVector) where {NDIMS_AMBIENT}
    # We use `StrideArray`s for `isbits` element types here since these buffers are used
    # in performance-critical places and the additional information passed to the
    # compiler makes them faster than native `Array`s (see `stride_array_or_array`).
    tmp1 = stride_array_or_array(real(mesh),
                                 StaticInt(NDIMS_AMBIENT), static_length(nodes),
                                 static_length(mesh.nodes))
    matrix1 = stride_array_or_array(real(mesh),
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

# For Gauss-Lobatto-Legendre (GLL) nodes, interface normals are computed on-the-fly
# during flux evaluation and do not need dedicated storage in the interface container.
function init_normal_directions!(interfaces::P4estInterfaceContainer{2},
                                 basis::LobattoLegendreBasis, elements)
    return nothing
end

# For Gauss-Legendre (GL) nodes, the interface normals are
# computed from interpolation of the volume node normals to the surface nodes.
function init_normal_directions!(interfaces::P4estInterfaceContainer{2},
                                 basis::GaussLegendreBasis, elements)
    @unpack neighbor_ids, node_indices, normal_directions = interfaces
    @unpack contravariant_vectors = elements
    @unpack boundary_interpolation = basis
    index_range = eachnode(basis)

    for interface in axes(neighbor_ids, 2)
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        # The index direction is identified based on `{i,j}_{primary, secondary}_step`.
        # For step = 0, the direction identified by this index is normal to the face.
        # For step != 0 (1 or -1), the direction identified by this index is tangential to the face.

        i_primary = i_primary_start
        j_primary = j_primary_start

        if i_primary_step == 0
            # i is the normal direction (constant), j varies along the surface
            # => Interpolate in first/normal direction
            # Interpolation side is governed by element orientation
            side = interpolation_side(primary_indices[1])
            for i in index_range
                normal_1 = zero(eltype(normal_directions))
                normal_2 = zero(eltype(normal_directions))
                for ii in index_range
                    factor = boundary_interpolation[ii, side]
                    # Retrieve normal directions at element/volume nodes
                    normal_direction = get_normal_direction(primary_direction,
                                                            contravariant_vectors,
                                                            ii, j_primary,
                                                            primary_element)
                    normal_1 = normal_1 + normal_direction[1] * factor
                    normal_2 = normal_2 + normal_direction[2] * factor
                end
                normal_directions[1, i, interface] = normal_1
                normal_directions[2, i, interface] = normal_2
                j_primary += j_primary_step # incrementing j_primary suffices (i_primary_step = 0)
            end
        else # j_primary_step == 0
            # j is the normal direction (constant), i varies along the surface
            # => Interpolate in second/normal direction
            # Interpolation side is governed by element orientation
            side = interpolation_side(primary_indices[2])
            for i in index_range
                normal_1 = zero(eltype(normal_directions))
                normal_2 = zero(eltype(normal_directions))
                for jj in index_range
                    factor = boundary_interpolation[jj, side]
                    # Retrieve normal directions at element/volume nodes
                    normal_direction = get_normal_direction(primary_direction,
                                                            contravariant_vectors,
                                                            i_primary, jj,
                                                            primary_element)
                    normal_1 = normal_1 + normal_direction[1] * factor
                    normal_2 = normal_2 + normal_direction[2] * factor
                end
                normal_directions[1, i, interface] = normal_1
                normal_directions[2, i, interface] = normal_2
                i_primary += i_primary_step # incrementing i_primary suffices (j_primary_step = 0)
            end
        end
    end

    return nothing
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

# Check whether the arrays in `mortars` and the thread-local storage used on CPUs
# have the axes we assume they must have in the inner loops of Trixi.jl.
function check_axes(mortars::P4estMortarContainer{2}, equations, solver::DG, cache)
    check_axes_mortar_container(mortars, equations, solver, cache)

    # Thread-local storage used for the mortar fluxes and projections
    buffer_axes = (eachvariable(equations), eachnode(solver))
    check_axes_threaded(cache.fstar_primary_upper_threaded, buffer_axes)
    check_axes_threaded(cache.fstar_primary_lower_threaded, buffer_axes)
    check_axes_threaded(cache.fstar_secondary_upper_threaded, buffer_axes)
    check_axes_threaded(cache.fstar_secondary_lower_threaded, buffer_axes)
    check_axes_threaded(cache.u_threaded, buffer_axes)
    return nothing
end

# Check only the arrays stored in `mortars`. This is used by the GPU kernels, which
# do not use the thread-local storage. Its size depends on `Threads.maxthreadid()`,
# which can grow at runtime, e.g., when CUDA.jl adopts foreign threads.
function check_axes_mortar_container(mortars::P4estMortarContainer{2}, equations,
                                     solver::DG, cache)
    check_axes(mortars.u,
               (Base.OneTo(2), eachvariable(equations),
                Base.OneTo(2),
                eachnode(solver),
                eachmortar(solver, cache)))
    check_axes(mortars.neighbor_ids, (Base.OneTo(3), eachmortar(solver, cache)))
    check_axes(mortars.node_indices, (Base.OneTo(2), eachmortar(solver, cache)))
    return nothing
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

# Auxiliary variables at the interface and mortar nodes, cf. `prolong2interfaces!` and
# `prolong2mortars!`.
# MPI is not supported yet since the auxiliary variables are not stored at MPI interfaces
# and MPI mortars, so parallel meshes use the fallback throwing an error. The values at
# the interfaces are copied from the volume nodes, which requires Gauss-Lobatto-Legendre
# nodes. Thus, other bases use the fallback throwing an error as well.
function init_aux_surface_node_vars!(aux_vars, mesh::P4estMeshSerial{2}, equations,
                                     solver::DGSEM{<:LobattoLegendreBasis}, cache)
    @unpack aux, aux_interface = aux_vars
    @unpack neighbor_ids, node_indices = cache.interfaces
    index_range = eachnode(solver)

    @threaded for interface in eachinterface(solver, cache)
        # Copy the auxiliary variables from the primary element using "delayed indexing"
        # with a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the indices of the primary side
        # will always run forwards.
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        for i in eachnode(solver)
            for v in eachauxvariable(equations)
                aux_interface[1, v, i, interface] = aux[v, i_primary, j_primary,
                                                        primary_element]
            end
            i_primary += i_primary_step
            j_primary += j_primary_step
        end

        # Copy the auxiliary variables from the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]

        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        for i in eachnode(solver)
            for v in eachauxvariable(equations)
                aux_interface[2, v, i, interface] = aux[v, i_secondary, j_secondary,
                                                        secondary_element]
            end
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    init_aux_mortar_node_vars!(aux_vars, mesh, equations, solver.mortar, solver, cache)

    return nothing
end

# The auxiliary variables are treated as the solution in `prolong2mortars!`: the values
# of the small elements are copied and the values of the large element are interpolated
# to the surface nodes of the small elements. Thus, the auxiliary variables do not need to
# be continuous across mortars.
function init_aux_mortar_node_vars!(aux_vars, mesh::P4estMeshSerial{2}, equations,
                                    mortar_l2::LobattoLegendreMortarL2, solver, cache)
    @unpack aux, aux_mortar = aux_vars
    @unpack neighbor_ids, node_indices = cache.mortars
    index_range = eachnode(solver)

    @threaded for mortar in eachmortar(solver, cache)
        # Copy the auxiliary variables from the small elements using "delayed indexing"
        # with a start value and a step size to get the correct face and orientation.
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
                for v in eachauxvariable(equations)
                    aux_mortar[1, v, position, i, mortar] = aux[v, i_small, j_small,
                                                                element]
                end
                i_small += i_small_step
                j_small += j_small_step
            end
        end

        # Buffer to copy the auxiliary variables of the large element face in the
        # correct orientation before interpolating. This is only done during the
        # initialization, so we do not need a pre-allocated buffer.
        aux_buffer = similar(aux, nauxvariables(equations), nnodes(solver))

        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_large = i_large_start
        j_large = j_large_start
        element = neighbor_ids[3, mortar]
        for i in eachnode(solver)
            for v in eachauxvariable(equations)
                aux_buffer[v, i] = aux[v, i_large, j_large, element]
            end
            i_large += i_large_step
            j_large += j_large_step
        end

        # Interpolate the auxiliary variables of the large element face to the surface
        # nodes of the small elements
        multiply_dimensionwise!(view(aux_mortar, 2, :, 1, :, mortar),
                                mortar_l2.forward_lower, aux_buffer)
        multiply_dimensionwise!(view(aux_mortar, 2, :, 2, :, mortar),
                                mortar_l2.forward_upper, aux_buffer)
    end

    return nothing
end
end # @muladd
