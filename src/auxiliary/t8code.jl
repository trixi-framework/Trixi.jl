"""
    init_t8code()

Initialize `t8code` by calling `t8_init` and setting the log level to `SC_LP_ERROR`.
This function will check if `t8code` is already initialized
and if yes, do nothing, thus it is safe to call it multiple times.
"""
function init_t8code()
  t8code_package_id = t8_get_package_id()
  if t8code_package_id >= 0
    return nothing
  end

  # Initialize `t8code` with log level ERROR to prevent a lot of output in AMR simulations
  t8_init(SC_LP_ERROR)

  return nothing
end

function trixi_t8_unref_forest(forest)
  t8_forest_unref(Ref(forest))
end

function t8_free(ptr)
  sc_free(t8_get_package_id(), ptr)
end

function trixi_t8_count_interfaces(forest)
  # Check that forest is a committed, that is valid and usable, forest.
  @T8_ASSERT (t8_forest_is_committed(forest) != 0)

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

  for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

    # Get the number of elements of this tree.
    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

    for ielement = 0:num_elements_in_tree-1
      element = t8_forest_get_element_in_tree(forest, itree, ielement)

      level = t8_element_level(eclass_scheme, element)

      num_faces = t8_element_num_faces(eclass_scheme,element)

      for iface = 0:num_faces-1

        pelement_indices_ref = Ref{Ptr{t8_locidx_t}}()
        pneighbor_leafs_ref = Ref{Ptr{Ptr{t8_element}}}()
        pneigh_scheme_ref = Ref{Ptr{t8_eclass_scheme}}()

        dual_faces_ref = Ref{Ptr{Cint}}()
        num_neighbors_ref = Ref{Cint}()

        forest_is_balanced = Cint(1)

        t8_forest_leaf_face_neighbors(forest, itree, element,
          pneighbor_leafs_ref, iface, dual_faces_ref, num_neighbors_ref,
          pelement_indices_ref, pneigh_scheme_ref, forest_is_balanced)

        num_neighbors      = num_neighbors_ref[]
        neighbor_ielements = unsafe_wrap(Array,pelement_indices_ref[],num_neighbors)
        neighbor_leafs     = unsafe_wrap(Array,pneighbor_leafs_ref[],num_neighbors)
        neighbor_scheme    = pneigh_scheme_ref[]

        if num_neighbors > 0
          neighbor_level = t8_element_level(neighbor_scheme, neighbor_leafs[1])

          # Conforming interface: The second condition ensures we only visit the interface once.
          if level == neighbor_level && current_index <= neighbor_ielements[1]
          # TODO: Find a fix for the case: Single element on root level with periodic boundaries.
          # elseif level == neighbor_level && 
          #   (all(Int32(current_index) .< neighbor_ielements) || 
          #   level == 0 && (iface == 0 || iface == 2 || iface == 4))
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
          mortars    = local_num_mortars,
          boundaries = local_num_boundary)
end

function trixi_t8_fill_mesh_info(forest, elements, interfaces, mortars, boundaries, boundary_names)
  # Check that forest is a committed, that is valid and usable, forest.
  @T8_ASSERT (t8_forest_is_committed(forest) != 0)

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

  for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

    # Get the number of elements of this tree.
    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

    for ielement = 0:num_elements_in_tree-1
      element = t8_forest_get_element_in_tree(forest, itree, ielement)

      level = t8_element_level(eclass_scheme, element)

      num_faces = t8_element_num_faces(eclass_scheme,element)

      for iface = 0:num_faces-1

        # Compute the `orientation` of the touching faces.
        if t8_element_is_root_boundary(eclass_scheme, element, iface) == 1
          cmesh = t8_forest_get_cmesh(forest)
          itree_in_cmesh = t8_forest_ltreeid_to_cmesh_ltreeid(forest, itree)
          iface_in_tree = t8_element_tree_face(eclass_scheme, element, iface)
          orientation_ref = Ref{Cint}()

          t8_cmesh_get_face_neighbor(cmesh, itree_in_cmesh, iface_in_tree, C_NULL, orientation_ref)
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

        t8_forest_leaf_face_neighbors(forest,itree,element,
          pneighbor_leafs_ref, iface, dual_faces_ref, num_neighbors_ref,
          pelement_indices_ref, pneigh_scheme_ref, forest_is_balanced)

        num_neighbors      = num_neighbors_ref[]
        dual_faces         = unsafe_wrap(Array,dual_faces_ref[],num_neighbors)
        neighbor_ielements = unsafe_wrap(Array,pelement_indices_ref[],num_neighbors)
        neighbor_leafs     = unsafe_wrap(Array,pneighbor_leafs_ref[],num_neighbors)
        neighbor_scheme    = pneigh_scheme_ref[]

        if num_neighbors > 0
          neighbor_level = t8_element_level(neighbor_scheme, neighbor_leafs[1])

          # Conforming interface: The second condition ensures we only visit the interface once.
          if level == neighbor_level && current_index <= neighbor_ielements[1]
          # TODO: Find a fix for the case: Single element on root level with periodic boundaries.
          # elseif level == neighbor_level &&
          #   (all(Int32(current_index) .< neighbor_ielements) ||
          #   level == 0 && (iface == 0 || iface == 2 || iface == 4))
              local_num_conform += 1

              faces = (iface, dual_faces[1])
              interface_id = local_num_conform

              # Write data to interfaces container.
              interfaces.neighbor_ids[1, interface_id] = current_index + 1
              interfaces.neighbor_ids[2, interface_id] = neighbor_ielements[1] + 1

              # Iterate over primary and secondary element.
              for side = 1:2
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
                  interfaces.node_indices[side, interface_id] = (:begin, indexing)
                elseif faces[side] == 1
                  # Index face in positive x-direction
                  interfaces.node_indices[side, interface_id] = (:end, indexing)
                elseif faces[side] == 2
                  # Index face in negative y-direction
                  interfaces.node_indices[side, interface_id] = (indexing, :begin)
                else # faces[side] == 3
                  # Index face in positive y-direction
                  interfaces.node_indices[side, interface_id] = (indexing, :end)
                end
              end

          # Non-conforming interface.
          elseif level < neighbor_level 
              local_num_mortars += 1

              faces = (dual_faces[1],iface)

              mortar_id = local_num_mortars

              # Last entry is the large element.
              mortars.neighbor_ids[end, mortar_id] = current_index + 1

              # First `1:end-1` entries are the smaller elements.
              mortars.neighbor_ids[1:end-1, mortar_id] .= neighbor_ielements[:] .+ 1

              for side = 1:2
                # Align mortar in positive coordinate direction of small side.
                # For orientation == 1, the large side needs to be indexed backwards
                # relative to the mortar.
                if side == 1 || orientation == 0
                  # Forward indexing for small side or orientation == 0.
                  indexing = :i_forward
                else
                  # Backward indexing for large side with reversed orientation.
                  indexing = :i_backward
                  # TODO: Fully understand what is going on here. Generalize this for 3D.
                  # Has something to do with Morton ordering.
                  mortars.neighbor_ids[1, mortar_id] = neighbor_ielements[2] + 1
                  mortars.neighbor_ids[2, mortar_id] = neighbor_ielements[1] + 1
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
          mortars    = local_num_mortars,
          boundaries = local_num_boundary)
end

function trixi_t8_get_local_element_levels(forest)
  # Check that forest is a committed, that is valid and usable, forest.
  @T8_ASSERT (t8_forest_is_committed(forest) != 0)

  levels = Vector{Int}(undef, t8_forest_get_local_num_elements(forest))

  # Get the number of trees that have elements of this process.
  num_local_trees = t8_forest_get_num_local_trees(forest)

  current_index = 0

  for itree = 0:num_local_trees-1
    tree_class = t8_forest_get_tree_class(forest, itree)
    eclass_scheme = t8_forest_get_eclass_scheme(forest, tree_class)

    # Get the number of elements of this tree.
    num_elements_in_tree = t8_forest_get_tree_num_elements(forest, itree)

    for ielement = 0:num_elements_in_tree-1
      element = t8_forest_get_element_in_tree(forest, itree, ielement)
      current_index += 1
      levels[current_index] = t8_element_level(eclass_scheme, element)
    end # for
  end # for

  return levels
end

function adapt_callback(forest,
                         forest_from,
                         which_tree,
                         lelement_id,
                         ts,
                         is_family, 
                         num_elements,
                         elements) :: Cint

  num_levels = t8_forest_get_local_num_elements(forest_from)

  indicator_ptr = Ptr{Int}(t8_forest_get_user_data(forest))
  indicators = unsafe_wrap(Array,indicator_ptr,num_levels)

  offset = t8_forest_get_tree_element_offset(forest_from, which_tree)

  # Only allow coarsening for complete families.
  if indicators[offset + lelement_id + 1] < 0 && is_family == 0
    return Cint(0)
  end

  return Cint(indicators[offset + lelement_id + 1])

end

function trixi_t8_adapt_new(old_forest, indicators)
  # Check that forest is a committed, that is valid and usable, forest.
  @T8_ASSERT (t8_forest_is_committed(old_forest) != 0)

  # Init new forest.
  new_forest_ref = Ref{t8_forest_t}()
  t8_forest_init(new_forest_ref)
  new_forest = new_forest_ref[]

  let set_from = C_NULL, recursive = 0, set_for_coarsening = 0, no_repartition = 0
    t8_forest_set_user_data(new_forest, pointer(indicators))
    t8_forest_set_adapt(new_forest, old_forest, @t8_adapt_callback(adapt_callback), recursive)
    t8_forest_set_balance(new_forest, set_from, no_repartition)
    t8_forest_set_partition(new_forest, set_from, set_for_coarsening)
    # t8_forest_set_ghost(new_forest, 1, T8_GHOST_FACES)
    t8_forest_commit(new_forest)
  end

  return new_forest
end

function trixi_t8_get_difference(old_levels, new_levels)

  old_nelems = length(old_levels)
  new_nelems = length(new_levels)

  changes = Vector{Int}(undef, old_nelems)

  # Local element indices.
  old_index = 1
  new_index = 1

  # TODO: Make general for 2D/3D and hybrid grids.
  T8_CHILDREN = 4

  while old_index <= old_nelems && new_index <= new_nelems

    if old_levels[old_index] < new_levels[new_index] 
      # Refined.
    
      changes[old_index] = 1

      old_index += 1
      new_index += T8_CHILDREN

    elseif old_levels[old_index] > new_levels[new_index] 
      # Coarsend.

      for child_index = old_index:old_index+T8_CHILDREN-1
        changes[child_index] = -1
      end

      old_index += T8_CHILDREN
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

  differences = trixi_t8_get_difference(old_levels, new_levels)

  mesh.forest = forest_cached

  return differences
end
