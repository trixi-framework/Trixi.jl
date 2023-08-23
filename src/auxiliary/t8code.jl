"""
    init_t8code()

Initialize `t8code` by calling `sc_init`, `p4est_init`, and `t8_init` while
setting the log level to `SC_LP_ERROR`. This function will check if `t8code`
is already initialized and if yes, do nothing, thus it is safe to call it
multiple times.
"""
function init_t8code()
    t8code_package_id = t8_get_package_id()
    if t8code_package_id >= 0
        return nothing
    end

    # Initialize the sc library, has to happen before we initialize t8code.
    let catch_signals = 0, print_backtrace = 0, log_handler = C_NULL
        T8code.Libt8.sc_init(mpi_comm(), catch_signals, print_backtrace, log_handler,
                             T8code.Libt8.SC_LP_ERROR)
    end

    if T8code.Libt8.p4est_is_initialized() == 0
        # Initialize `p4est` with log level ERROR to prevent a lot of output in AMR simulations
        T8code.Libt8.p4est_init(C_NULL, T8code.Libt8.SC_LP_ERROR)
    end

    # Initialize t8code with log level ERROR to prevent a lot of output in AMR simulations.
    t8_init(T8code.Libt8.SC_LP_ERROR)

    if haskey(ENV, "TRIXI_T8CODE_SC_FINALIZE")
        # Normally, `sc_finalize` should always be called during shutdown of an
        # application. It checks whether there is still un-freed memory by t8code
        # and/or T8code.jl and throws an exception if this is the case. For
        # production runs this is not mandatory, but is helpful during
        # development. Hence, this option is only activated when environment
        # variable TRIXI_T8CODE_SC_FINALIZE exists.
        @warn "T8code.jl: sc_finalize will be called during shutdown of Trixi.jl."
        MPI.add_finalize_hook!(T8code.Libt8.sc_finalize)
    end

    return nothing
end

function trixi_t8_unref_forest(forest)
    t8_forest_unref(Ref(forest))
end

function t8_free(ptr)
    T8code.Libt8.sc_free(t8_get_package_id(), ptr)
end

function trixi_t8_count_interfaces(forest)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(forest) != 0

    # Get the number of local elements of forest.
    num_local_elements = t8_forest_get_local_num_elements(forest)
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)
    # Get the number of trees that have elements of this process.
    num_local_trees = t8_forest_get_num_local_trees(forest)

    current_index = t8_locidx_t(0)

    local_num_conform = 0
    local_num_mortars = 0
    local_num_boundary = 0

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        # Get the number of elements of this tree.
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(forest, itree, ielement)

            level = t8_element_level(eclass_scheme, element)

            num_faces = t8_element_num_faces(eclass_scheme, element)

            for iface in 0:(num_faces - 1)
                pelement_indices_ref = Ref{Ptr{t8_locidx_t}}()
                pneighbor_leafs_ref = Ref{Ptr{Ptr{t8_element}}}()
                pneigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

                dual_faces_ref = Ref{Ptr{Cint}}()
                num_neighbors_ref = Ref{Cint}()

                forest_is_balanced = Cint(1)

                t8_forest_leaf_face_neighbors(forest, itree, element,
                                              pneighbor_leafs_ref, iface, dual_faces_ref,
                                              num_neighbors_ref,
                                              pelement_indices_ref, pneigh_scheme_ref,
                                              forest_is_balanced)

                num_neighbors = num_neighbors_ref[]
                neighbor_ielements = unsafe_wrap(Array, pelement_indices_ref[],
                                                 num_neighbors)
                neighbor_leafs = unsafe_wrap(Array, pneighbor_leafs_ref[], num_neighbors)
                neighbor_scheme = pneigh_scheme_ref[]

                if num_neighbors > 0
                    neighbor_level = t8_element_level(neighbor_scheme, neighbor_leafs[1])

                    # Conforming interface: The second condition ensures we only visit the interface once.
                    if level == neighbor_level && current_index <= neighbor_ielements[1]
                        local_num_conform += 1
                    elseif level < neighbor_level
                        local_num_mortars += 1
                    end

                else
                    local_num_boundary += 1
                end

                t8_free(dual_faces_ref[])
                t8_free(pneighbor_leafs_ref[])
                t8_free(pelement_indices_ref[])
            end # for

            current_index += 1
        end # for
    end # for

    return (interfaces = local_num_conform,
            mortars = local_num_mortars,
            boundaries = local_num_boundary)
end

function trixi_t8_fill_mesh_info(forest, elements, interfaces, mortars, boundaries,
                                 boundary_names)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(forest) != 0

    # Get the number of local elements of forest.
    num_local_elements = t8_forest_get_local_num_elements(forest)
    # Get the number of ghost elements of forest.
    num_ghost_elements = t8_forest_get_num_ghosts(forest)
    # Get the number of trees that have elements of this process.
    num_local_trees = t8_forest_get_num_local_trees(forest)

    current_index = t8_locidx_t(0)

    local_num_conform = 0
    local_num_mortars = 0
    local_num_boundary = 0

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        # Get the number of elements of this tree.
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(forest, itree, ielement)

            level = t8_element_level(eclass_scheme, element)

            num_faces = t8_element_num_faces(eclass_scheme, element)

            for iface in 0:(num_faces - 1)

                # Compute the `orientation` of the touching faces.
                if t8_element_is_root_boundary(eclass_scheme, element, iface) == 1
                    cmesh = t8_forest_get_cmesh(forest)
                    itree_in_cmesh = t8_forest_ltreeid_to_cmesh_ltreeid(forest, itree)
                    iface_in_tree = t8_element_tree_face(eclass_scheme, element, iface)
                    orientation_ref = Ref{Cint}()

                    t8_cmesh_get_face_neighbor(cmesh, itree_in_cmesh, iface_in_tree, C_NULL,
                                               orientation_ref)
                    orientation = orientation_ref[]
                else
                    orientation = zero(Cint)
                end

                pelement_indices_ref = Ref{Ptr{t8_locidx_t}}()
                pneighbor_leafs_ref = Ref{Ptr{Ptr{t8_element}}}()
                pneigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

                dual_faces_ref = Ref{Ptr{Cint}}()
                num_neighbors_ref = Ref{Cint}()

                forest_is_balanced = Cint(1)

                t8_forest_leaf_face_neighbors(forest, itree, element,
                                              pneighbor_leafs_ref, iface, dual_faces_ref,
                                              num_neighbors_ref,
                                              pelement_indices_ref, pneigh_scheme_ref,
                                              forest_is_balanced)

                num_neighbors = num_neighbors_ref[]
                dual_faces = unsafe_wrap(Array, dual_faces_ref[], num_neighbors)
                neighbor_ielements = unsafe_wrap(Array, pelement_indices_ref[],
                                                 num_neighbors)
                neighbor_leafs = unsafe_wrap(Array, pneighbor_leafs_ref[], num_neighbors)
                neighbor_scheme = pneigh_scheme_ref[]

                if num_neighbors > 0
                    neighbor_level = t8_element_level(neighbor_scheme, neighbor_leafs[1])

                    # Conforming interface: The second condition ensures we only visit the interface once.
                    if level == neighbor_level && current_index <= neighbor_ielements[1]
                        local_num_conform += 1

                        faces = (iface, dual_faces[1])
                        interface_id = local_num_conform

                        # Write data to interfaces container.
                        interfaces.neighbor_ids[1, interface_id] = current_index + 1
                        interfaces.neighbor_ids[2, interface_id] = neighbor_ielements[1] + 1

                        # Iterate over primary and secondary element.
                        for side in 1:2
                            # Align interface in positive coordinate direction of primary element.
                            # For orientation == 1, the secondary element needs to be indexed backwards
                            # relative to the interface.
                            if side == 1 || orientation == 0
                                # Forward indexing
                                indexing = :i_forward
                            else
                                # Backward indexing
                                indexing = :i_backward
                            end

                            if faces[side] == 0
                                # Index face in negative x-direction
                                interfaces.node_indices[side, interface_id] = (:begin,
                                                                               indexing)
                            elseif faces[side] == 1
                                # Index face in positive x-direction
                                interfaces.node_indices[side, interface_id] = (:end,
                                                                               indexing)
                            elseif faces[side] == 2
                                # Index face in negative y-direction
                                interfaces.node_indices[side, interface_id] = (indexing,
                                                                               :begin)
                            else # faces[side] == 3
                                # Index face in positive y-direction
                                interfaces.node_indices[side, interface_id] = (indexing,
                                                                               :end)
                            end
                        end

                        # Non-conforming interface.
                    elseif level < neighbor_level
                        local_num_mortars += 1

                        faces = (dual_faces[1], iface)

                        mortar_id = local_num_mortars

                        # Last entry is the large element.
                        mortars.neighbor_ids[end, mortar_id] = current_index + 1

                        # First `1:end-1` entries are the smaller elements.
                        mortars.neighbor_ids[1:(end - 1), mortar_id] .= neighbor_ielements .+
                                                                        1

                        for side in 1:2
                            # Align mortar in positive coordinate direction of small side.
                            # For orientation == 1, the large side needs to be indexed backwards
                            # relative to the mortar.
                            if side == 1 || orientation == 0
                                # Forward indexing for small side or orientation == 0.
                                indexing = :i_forward
                            else
                                # Backward indexing for large side with reversed orientation.
                                indexing = :i_backward
                                # Since the orientation is reversed we have to account for this
                                # when filling the `neighbor_ids` array.
                                mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[2] +
                                                                     1
                                mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[1] +
                                                                     1
                            end

                            if faces[side] == 0
                                # Index face in negative x-direction
                                mortars.node_indices[side, mortar_id] = (:begin, indexing)
                            elseif faces[side] == 1
                                # Index face in positive x-direction
                                mortars.node_indices[side, mortar_id] = (:end, indexing)
                            elseif faces[side] == 2
                                # Index face in negative y-direction
                                mortars.node_indices[side, mortar_id] = (indexing, :begin)
                            else # faces[side] == 3
                                # Index face in positive y-direction
                                mortars.node_indices[side, mortar_id] = (indexing, :end)
                            end
                        end

                        # else: "level > neighbor_level" is skipped since we visit the mortar interface only once.
                    end

                    # Domain boundary.
                else
                    local_num_boundary += 1
                    boundary_id = local_num_boundary

                    boundaries.neighbor_ids[boundary_id] = current_index + 1

                    if iface == 0
                        # Index face in negative x-direction.
                        boundaries.node_indices[boundary_id] = (:begin, :i_forward)
                    elseif iface == 1
                        # Index face in positive x-direction.
                        boundaries.node_indices[boundary_id] = (:end, :i_forward)
                    elseif iface == 2
                        # Index face in negative y-direction.
                        boundaries.node_indices[boundary_id] = (:i_forward, :begin)
                    else # iface == 3
                        # Index face in positive y-direction.
                        boundaries.node_indices[boundary_id] = (:i_forward, :end)
                    end

                    # One-based indexing.
                    boundaries.name[boundary_id] = boundary_names[iface + 1, itree + 1]
                end

                t8_free(dual_faces_ref[])
                t8_free(pneighbor_leafs_ref[])
                t8_free(pelement_indices_ref[])
            end # for iface = ...

            current_index += 1
        end # for
    end # for

    return (interfaces = local_num_conform,
            mortars = local_num_mortars,
            boundaries = local_num_boundary)
end

function trixi_t8_get_local_element_levels(forest)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(forest) != 0

    levels = Vector{Int}(undef, t8_forest_get_local_num_elements(forest))

    # Get the number of trees that have elements of this process.
    num_local_trees = t8_forest_get_num_local_trees(forest)

    current_index = 0

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

        # Get the number of elements of this tree.
        num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(forest, itree, ielement)
            current_index += 1
            levels[current_index] = t8_element_level(eclass_scheme, element)
        end # for
    end # for

    return levels
end

# Callback function prototype to decide for refining and coarsening.
# If `is_family` equals 1, the first `num_elements` in elements
# form a family and we decide whether this family should be coarsened
# or only the first element should be refined.
# Otherwise `is_family` must equal zero and we consider the first entry
# of the element array for refinement. 
# Entries of the element array beyond the first `num_elements` are undefined.
# \param [in] forest       the forest to which the new elements belong
# \param [in] forest_from  the forest that is adapted.
# \param [in] which_tree   the local tree containing `elements`
# \param [in] lelement_id  the local element id in `forest_old` in the tree of the current element
# \param [in] ts           the eclass scheme of the tree
# \param [in] is_family    if 1, the first `num_elements` entries in `elements` form a family. If 0, they do not.
# \param [in] num_elements the number of entries in `elements` that are defined
# \param [in] elements     Pointers to a family or, if `is_family` is zero,
#                          pointer to one element.
# \return greater zero if the first entry in `elements` should be refined,
#         smaller zero if the family `elements` shall be coarsened,
#         zero else.
function adapt_callback(forest,
                        forest_from,
                        which_tree,
                        lelement_id,
                        ts,
                        is_family,
                        num_elements,
                        elements)::Cint
    num_levels = t8_forest_get_local_num_elements(forest_from)

    indicator_ptr = Ptr{Int}(t8_forest_get_user_data(forest))
    indicators = unsafe_wrap(Array, indicator_ptr, num_levels)

    offset = t8_forest_get_tree_element_offset(forest_from, which_tree)

    # Only allow coarsening for complete families.
    if indicators[offset + lelement_id + 1] < 0 && is_family == 0
        return Cint(0)
    end

    return Cint(indicators[offset + lelement_id + 1])
end

function trixi_t8_adapt_new(old_forest, indicators)
    # Check that forest is a committed, that is valid and usable, forest.
    @assert t8_forest_is_committed(old_forest) != 0

    # Init new forest.
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    let set_from = C_NULL, recursive = 0, set_for_coarsening = 0, no_repartition = 0
        t8_forest_set_user_data(new_forest, pointer(indicators))
        t8_forest_set_adapt(new_forest, old_forest, @t8_adapt_callback(adapt_callback),
                            recursive)
        t8_forest_set_balance(new_forest, set_from, no_repartition)
        t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
        t8_forest_set_ghost(new_forest, 1, T8_GHOST_FACES) # Note: MPI support not available yet so it is a dummy call.
        t8_forest_commit(new_forest)
    end

    return new_forest
end

function trixi_t8_get_difference(old_levels, new_levels, num_children)
    old_nelems = length(old_levels)
    new_nelems = length(new_levels)

    changes = Vector{Int}(undef, old_nelems)

    # Local element indices.
    old_index = 1
    new_index = 1

    while old_index <= old_nelems && new_index <= new_nelems
        if old_levels[old_index] < new_levels[new_index]
            # Refined.

            changes[old_index] = 1

            old_index += 1
            new_index += num_children

        elseif old_levels[old_index] > new_levels[new_index]
            # Coarsend.

            for child_index in old_index:(old_index + num_children - 1)
                changes[child_index] = -1
            end

            old_index += num_children
            new_index += 1

        else
            # No changes.

            changes[old_index] = 0

            old_index += 1
            new_index += 1
        end
    end

    return changes
end

# Coarsen or refine marked cells and rebalance forest. Return a difference between
# old and new mesh.
function trixi_t8_adapt!(mesh, indicators)
    old_levels = trixi_t8_get_local_element_levels(mesh.forest)

    forest_cached = trixi_t8_adapt_new(mesh.forest, indicators)

    new_levels = trixi_t8_get_local_element_levels(forest_cached)

    differences = trixi_t8_get_difference(old_levels, new_levels, 2^ndims(mesh))

    mesh.forest = forest_cached

    return differences
end
