"""
    init_t8code()

Initialize `t8code` by calling `sc_init`, `p4est_init`, and `t8_init` while
setting the log level to `SC_LP_ERROR`. This function will check if `t8code`
is already initialized and if yes, do nothing, thus it is safe to call it
multiple times.
"""
function init_t8code()
    # Only initialize t8code if T8code.jl can be used
    if T8code.preferences_set_correctly()
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
            @info "T8code.jl: `sc_finalize` will be called during shutdown of Trixi.jl."
            MPI.add_finalize_hook!(T8code.Libt8.sc_finalize)
        end
    else
        @warn "Preferences for T8code.jl are not set correctly. Until fixed, using `T8codeMesh` will result in a crash. " *
              "See also https://trixi-framework.github.io/Trixi.jl/stable/parallelization/#parallel_system_MPI"
    end

    return nothing
end

function trixi_t8_unref_forest(forest)
    t8_forest_unref(Ref(forest))
end

function t8_free(ptr)
    T8code.Libt8.sc_free(t8_get_package_id(), ptr)
end

function t8_forest_ghost_get_remotes(forest)
    num_remotes_ref = Ref{Cint}()
    remotes_ptr = @ccall T8code.Libt8.libt8.t8_forest_ghost_get_remotes(forest::t8_forest_t,
                                                                        num_remotes_ref::Ptr{Cint})::Ptr{Cint}
    remotes = unsafe_wrap(Array, remotes_ptr, num_remotes_ref[])
end

function t8_forest_ghost_remote_first_elem(forest, remote)
    @ccall T8code.Libt8.libt8.t8_forest_ghost_remote_first_elem(forest::t8_forest_t,
                                                                remote::Cint)::t8_locidx_t
end

function t8_forest_ghost_num_trees(forest)
    @ccall T8code.Libt8.libt8.t8_forest_ghost_num_trees(forest::t8_forest_t)::t8_locidx_t
end

function t8_forest_ghost_get_tree_element_offset(forest, lghost_tree)
    @ccall T8code.Libt8.libt8.t8_forest_ghost_get_tree_element_offset(forest::t8_forest_t,
                                                                      lghost_tree::t8_locidx_t)::t8_locidx_t
end

function t8_forest_ghost_get_global_treeid(forest, lghost_tree)
    @ccall T8code.Libt8.libt8.t8_forest_ghost_get_global_treeid(forest::t8_forest_t,
                                                                lghost_tree::t8_locidx_t)::t8_gloidx_t
end

function trixi_t8_count_interfaces(mesh)
    @assert t8_forest_is_committed(mesh.forest) != 0

    num_local_elements = t8_forest_get_local_num_elements(mesh.forest)
    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)

    current_index = t8_locidx_t(0)

    local_num_conform = 0
    local_num_mortars = 0
    local_num_boundary = 0

    local_num_mpi_conform = 0
    local_num_mpi_mortars = 0

    visited_global_mortar_ids = Set{UInt64}([])

    max_level = t8_forest_get_maxlevel(mesh.forest) #UInt64
    max_tree_num_elements = UInt64(2^ndims(mesh))^max_level

    if mpi_isparallel()
        remotes = t8_forest_ghost_get_remotes(mesh.forest)
        ghost_remote_first_elem = [num_local_elements +
                                   t8_forest_ghost_remote_first_elem(mesh.forest, remote)
                                   for remote in remotes]

        ghost_num_trees = t8_forest_ghost_num_trees(mesh.forest)

        ghost_tree_element_offsets = [num_local_elements +
                                      t8_forest_ghost_get_tree_element_offset(mesh.forest,
                                                                              itree)
                                      for itree in 0:(ghost_num_trees - 1)]
        ghost_global_treeids = [t8_forest_ghost_get_global_treeid(mesh.forest, itree)
                                for itree in 0:(ghost_num_trees - 1)]
    end

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(mesh.forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(mesh.forest, tree_class)

        num_elements_in_tree = t8_forest_get_tree_num_elements(mesh.forest, itree)

        global_itree = t8_forest_global_tree_id(mesh.forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(mesh.forest, itree, ielement)

            level = t8_element_level(eclass_scheme, element)

            num_faces = t8_element_num_faces(eclass_scheme, element)

            # Note: This works only for forests of one element class.
            current_linear_id = global_itree * max_tree_num_elements +
                                t8_element_get_linear_id(eclass_scheme, element, max_level)

            for iface in 0:(num_faces - 1)
                pelement_indices_ref = Ref{Ptr{t8_locidx_t}}()
                pneighbor_leafs_ref = Ref{Ptr{Ptr{t8_element}}}()
                pneigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

                dual_faces_ref = Ref{Ptr{Cint}}()
                num_neighbors_ref = Ref{Cint}()

                forest_is_balanced = Cint(1)

                t8_forest_leaf_face_neighbors(mesh.forest, itree, element,
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

                if num_neighbors == 0
                    local_num_boundary += 1
                else
                    neighbor_level = t8_element_level(neighbor_scheme, neighbor_leafs[1])

                    if all(neighbor_ielements .< num_local_elements)
                        # Conforming interface: The second condition ensures we
                        # only visit the interface once.
                        if level == neighbor_level && current_index <= neighbor_ielements[1]
                            local_num_conform += 1
                        elseif level < neighbor_level
                            local_num_mortars += 1
                            # else `level > neighbor_level` is ignored.
                        end
                    else
                        if level == neighbor_level
                            local_num_mpi_conform += 1
                        elseif level < neighbor_level
                            local_num_mpi_mortars += 1

                            global_mortar_id = 2 * ndims(mesh) * current_linear_id + iface

                        else # level > neighbor_level
                            neighbor_global_ghost_itree = ghost_global_treeids[findlast(ghost_tree_element_offsets .<=
                                                                                        neighbor_ielements[1])]
                            neighbor_linear_id = neighbor_global_ghost_itree *
                                                 max_tree_num_elements +
                                                 t8_element_get_linear_id(neighbor_scheme,
                                                                          neighbor_leafs[1],
                                                                          max_level)
                            global_mortar_id = 2 * ndims(mesh) * neighbor_linear_id +
                                               dual_faces[1]

                            if !(global_mortar_id in visited_global_mortar_ids)
                                push!(visited_global_mortar_ids, global_mortar_id)
                                local_num_mpi_mortars += 1
                            end
                        end
                    end
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
            boundaries = local_num_boundary,
            mpi_interfaces = local_num_mpi_conform,
            mpi_mortars = local_num_mpi_mortars)
end

function trixi_t8_fill_mesh_info(mesh, elements, interfaces, mortars, boundaries,
                                 boundary_names; mpi_mesh_info = undef)
    @assert t8_forest_is_committed(mesh.forest) != 0

    num_local_elements = t8_forest_get_local_num_elements(mesh.forest)
    num_local_trees = t8_forest_get_num_local_trees(mesh.forest)

    current_index = t8_locidx_t(0)

    max_level = t8_forest_get_maxlevel(mesh.forest) #UInt64
    max_tree_num_elements = UInt64(2^ndims(mesh))^max_level

    if mpi_isparallel()
        #! format: off
        remotes = t8_forest_ghost_get_remotes(mesh.forest)
        ghost_num_trees = t8_forest_ghost_num_trees(mesh.forest)

        ghost_remote_first_elem = [num_local_elements +
                                   t8_forest_ghost_remote_first_elem(mesh.forest, remote)
                                   for remote in remotes]

        ghost_tree_element_offsets = [num_local_elements +
                                      t8_forest_ghost_get_tree_element_offset(mesh.forest, itree)
                                      for itree in 0:(ghost_num_trees - 1)]

        ghost_global_treeids = [t8_forest_ghost_get_global_treeid(mesh.forest, itree)
                                for itree in 0:(ghost_num_trees - 1)]
        #! format: on
    end

    local_num_conform = 0
    local_num_mortars = 0
    local_num_boundary = 0

    local_num_mpi_conform = 0
    local_num_mpi_mortars = 0

    # Works for quads and hexs.
    map_iface_to_ichild_to_position = [
        # 0  1  2  3  4  5  6  7 ichild/iface
        [1, 0, 2, 0, 3, 0, 4, 0], # 0
        [0, 1, 0, 2, 0, 3, 0, 4], # 1
        [1, 2, 0, 0, 3, 4, 0, 0], # 2
        [0, 0, 1, 2, 0, 0, 3, 4], # 3
        [1, 2, 3, 4, 0, 0, 0, 0], # 4
        [0, 0, 0, 0, 1, 2, 3, 4], # 5
    ]

    visited_global_mortar_ids = Set{UInt64}([])
    global_mortar_id_to_local = Dict{UInt64, Int}([])

    for itree in 0:(num_local_trees - 1)
        tree_class = t8_forest_get_tree_class(mesh.forest, itree)
        eclass_scheme = t8_forest_get_eclass_scheme(mesh.forest, tree_class)

        num_elements_in_tree = t8_forest_get_tree_num_elements(mesh.forest, itree)

        global_itree = t8_forest_global_tree_id(mesh.forest, itree)

        for ielement in 0:(num_elements_in_tree - 1)
            element = t8_forest_get_element_in_tree(mesh.forest, itree, ielement)

            level = t8_element_level(eclass_scheme, element)

            num_faces = t8_element_num_faces(eclass_scheme, element)

            # Note: This works only for forests of one element class.
            current_linear_id = global_itree * max_tree_num_elements +
                                t8_element_get_linear_id(eclass_scheme, element, max_level)

            for iface in 0:(num_faces - 1)
                # Compute the `orientation` of the touching faces.
                if t8_element_is_root_boundary(eclass_scheme, element, iface) == 1
                    cmesh = t8_forest_get_cmesh(mesh.forest)
                    itree_in_cmesh = t8_forest_ltreeid_to_cmesh_ltreeid(mesh.forest, itree)
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

                t8_forest_leaf_face_neighbors(mesh.forest, itree, element,
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

                if num_neighbors == 0 # Domain boundary.
                    local_num_boundary += 1
                    boundary_id = local_num_boundary

                    boundaries.neighbor_ids[boundary_id] = current_index + 1

                    init_boundary_node_indices!(boundaries, iface, boundary_id)

                    # One-based indexing.
                    boundaries.name[boundary_id] = boundary_names[iface + 1, itree + 1]

                else # Interfaces/mortars.
                    neighbor_level = t8_element_level(neighbor_scheme, neighbor_leafs[1])

                    # Local interfaces/mortars.
                    if all(neighbor_ielements .< num_local_elements)
                        # Conforming interface: The second condition ensures we only visit the interface once.
                        if level == neighbor_level && current_index <= neighbor_ielements[1]
                            local_num_conform += 1

                            interfaces.neighbor_ids[1, local_num_conform] = current_index +
                                                                            1
                            interfaces.neighbor_ids[2, local_num_conform] = neighbor_ielements[1] +
                                                                            1

                            init_interface_node_indices!(interfaces, (iface, dual_faces[1]),
                                                         orientation,
                                                         local_num_conform)
                            # Non-conforming interface.
                        elseif level < neighbor_level
                            local_num_mortars += 1

                            # Last entry is the large element.
                            mortars.neighbor_ids[end, local_num_mortars] = current_index + 1

                            init_mortar_neighbor_ids!(mortars, iface, dual_faces[1],
                                                      orientation, neighbor_ielements,
                                                      local_num_mortars)

                            init_mortar_node_indices!(mortars, (dual_faces[1], iface),
                                                      orientation, local_num_mortars)

                            # else: `level > neighbor_level` is skipped since we visit the mortar interface only once.
                        end
                    else # MPI interfaces/mortars.
                        # Conforming MPI interface.
                        if level == neighbor_level
                            local_num_mpi_conform += 1

                            neighbor_global_ghost_itree = ghost_global_treeids[findlast(ghost_tree_element_offsets .<=
                                                                                        neighbor_ielements[1])]

                            neighbor_linear_id = neighbor_global_ghost_itree *
                                                 max_tree_num_elements +
                                                 t8_element_get_linear_id(neighbor_scheme,
                                                                          neighbor_leafs[1],
                                                                          max_level)

                            if current_linear_id < neighbor_linear_id
                                local_side = 1
                                smaller_iface = iface
                                smaller_linear_id = current_linear_id
                                faces = (iface, dual_faces[1])
                            else
                                local_side = 2
                                smaller_iface = dual_faces[1]
                                smaller_linear_id = neighbor_linear_id
                                faces = (dual_faces[1], iface)
                            end

                            global_interface_id = 2 * ndims(mesh) * smaller_linear_id +
                                                  smaller_iface

                            mpi_mesh_info.mpi_interfaces.local_neighbor_ids[local_num_mpi_conform] = current_index +
                                                                                                     1
                            mpi_mesh_info.mpi_interfaces.local_sides[local_num_mpi_conform] = local_side

                            init_mpi_interface_node_indices!(mpi_mesh_info.mpi_interfaces,
                                                             faces, local_side, orientation,
                                                             local_num_mpi_conform)

                            neighbor_rank = remotes[findlast(ghost_remote_first_elem .<=
                                                             neighbor_ielements[1])]
                            mpi_mesh_info.neighbor_ranks_interface[local_num_mpi_conform] = neighbor_rank

                            mpi_mesh_info.global_interface_ids[local_num_mpi_conform] = global_interface_id

                            # MPI Mortar.
                        elseif level < neighbor_level
                            local_num_mpi_mortars += 1

                            global_mortar_id = 2 * ndims(mesh) * current_linear_id + iface

                            neighbor_ids = neighbor_ielements .+ 1

                            local_neighbor_positions = findall(neighbor_ids .<=
                                                               num_local_elements)
                            local_neighbor_ids = [neighbor_ids[i]
                                                  for i in local_neighbor_positions]
                            local_neighbor_positions = [map_iface_to_ichild_to_position[dual_faces[1] + 1][t8_element_child_id(neighbor_scheme, neighbor_leafs[i]) + 1]
                                                        for i in local_neighbor_positions]

                            # Last entry is the large element.
                            push!(local_neighbor_ids, current_index + 1)
                            push!(local_neighbor_positions, 2^(ndims(mesh) - 1) + 1)

                            mpi_mesh_info.mpi_mortars.local_neighbor_ids[local_num_mpi_mortars] = local_neighbor_ids
                            mpi_mesh_info.mpi_mortars.local_neighbor_positions[local_num_mpi_mortars] = local_neighbor_positions

                            init_mortar_node_indices!(mpi_mesh_info.mpi_mortars,
                                                      (dual_faces[1], iface), orientation,
                                                      local_num_mpi_mortars)

                            neighbor_ranks = [remotes[findlast(ghost_remote_first_elem .<=
                                                               ineighbor_ghost)]
                                              for ineighbor_ghost in filter(x -> x >=
                                                                                 num_local_elements,
                                                                            neighbor_ielements)]
                            mpi_mesh_info.neighbor_ranks_mortar[local_num_mpi_mortars] = neighbor_ranks

                            mpi_mesh_info.global_mortar_ids[local_num_mpi_mortars] = global_mortar_id

                            # MPI Mortar: larger element is ghost
                        else
                            neighbor_global_ghost_itree = ghost_global_treeids[findlast(ghost_tree_element_offsets .<=
                                                                                        neighbor_ielements[1])]
                            neighbor_linear_id = neighbor_global_ghost_itree *
                                                 max_tree_num_elements +
                                                 t8_element_get_linear_id(neighbor_scheme,
                                                                          neighbor_leafs[1],
                                                                          max_level)
                            global_mortar_id = 2 * ndims(mesh) * neighbor_linear_id +
                                               dual_faces[1]

                            if global_mortar_id in visited_global_mortar_ids
                                local_mpi_mortar_id = global_mortar_id_to_local[global_mortar_id]

                                push!(mpi_mesh_info.mpi_mortars.local_neighbor_ids[local_mpi_mortar_id],
                                      current_index + 1)
                                push!(mpi_mesh_info.mpi_mortars.local_neighbor_positions[local_mpi_mortar_id],
                                      map_iface_to_ichild_to_position[iface + 1][t8_element_child_id(eclass_scheme, element) + 1])
                            else
                                local_num_mpi_mortars += 1
                                local_mpi_mortar_id = local_num_mpi_mortars
                                push!(visited_global_mortar_ids, global_mortar_id)
                                global_mortar_id_to_local[global_mortar_id] = local_mpi_mortar_id

                                mpi_mesh_info.mpi_mortars.local_neighbor_ids[local_mpi_mortar_id] = [
                                    current_index + 1,
                                ]
                                mpi_mesh_info.mpi_mortars.local_neighbor_positions[local_mpi_mortar_id] = [
                                    map_iface_to_ichild_to_position[iface + 1][t8_element_child_id(eclass_scheme, element) + 1],
                                ]
                                init_mortar_node_indices!(mpi_mesh_info.mpi_mortars,
                                                          (iface, dual_faces[1]),
                                                          orientation, local_mpi_mortar_id)

                                neighbor_ranks = [
                                    remotes[findlast(ghost_remote_first_elem .<=
                                                     neighbor_ielements[1])],
                                ]
                                mpi_mesh_info.neighbor_ranks_mortar[local_mpi_mortar_id] = neighbor_ranks

                                mpi_mesh_info.global_mortar_ids[local_mpi_mortar_id] = global_mortar_id
                            end
                        end
                    end
                end

                t8_free(dual_faces_ref[])
                t8_free(pneighbor_leafs_ref[])
                t8_free(pelement_indices_ref[])
            end # for iface

            current_index += 1
        end # for ielement
    end # for itree

    return nothing
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
    new_forest_ref = Ref{t8_forest_t}()
    t8_forest_init(new_forest_ref)
    new_forest = new_forest_ref[]

    let set_from = C_NULL, recursive = 0, set_for_coarsening = 1, no_repartition = 1,
        do_ghost = 1

        t8_forest_set_user_data(new_forest, pointer(indicators))
        t8_forest_set_adapt(new_forest, old_forest, @t8_adapt_callback(adapt_callback),
                            recursive)
        t8_forest_set_balance(new_forest, set_from, no_repartition)
        t8_forest_set_ghost(new_forest, do_ghost, T8_GHOST_FACES)
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
