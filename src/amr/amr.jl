
# Obtain AMR indicators from solver, then adapt mesh, and finally adapt solver
#
# If `only_refine` is true, no coarsening will be performed, independent of the indicator values.
# If `only_coarsen` is true, no refinement will be performed, independent of the indicator values.
# Passively adapted solvers in `passive_solvers` will be adapated according to the primary solver.
#
# Return true if anything was changed, false if no cells where coarsened/refined
function adapt!(mesh::TreeMesh, solver::AbstractSolver, time;
                only_refine=false, only_coarsen=false, passive_solvers=[])
  # Debug output
  globals[:verbose] && print("Begin adaptation...")

  # Alias for convenience
  tree = mesh.tree
  println("Refine 2")
  # Determine indicator value
  lambda = @timeit timer() "indicator" calc_amr_indicator(solver, mesh, time)

  # Get list of current leaf cells
  leaf_cell_ids = leaf_cells(tree)
  @assert length(lambda) == length(leaf_cell_ids) ("Indicator and leaf cell arrays have " *
                                                   "different length")

  # Set thresholds for refinement and coarsening
  indicator_threshold_refinement = parameter("indicator_threshold_refinement",  0.5)
  indicator_threshold_coarsening = parameter("indicator_threshold_coarsening", -0.5)
## Add for p4est
  local_num_quadrants = mesh.tree.forest.local_num_quadrants
  # @assert local_num_quadrants == length(leaf_cell_ids)
  to_refine_to_coarse = zeros(Int32, 1,local_num_quadrants)
  for i = 1:local_num_quadrants
    if lambda[i] > indicator_threshold_refinement && !only_coarsen
      to_refine_to_coarse[i] = 1
    end
    if lambda[i] < indicator_threshold_coarsening && !only_refine
      to_refine_to_coarse[i] = -1
    end
  end
  to_refine_to_coarse_ptr = pointer(to_refine_to_coarse)

  # *  # 0. Set inner old quad data to Zero
  p4est = mesh.tree.forest
  CvolumeIterate = @cfunction(setOldIdtoZero, Cvoid, (Ptr{P4est.p4est_iter_volume_info_t}, Ptr{Cvoid}))
  P4est.p4est_iterate( p4est,  C_NULL, C_NULL, CvolumeIterate, C_NULL, C_NULL)

  recursive = 0
  allowed_level = 8 #P4est.P4EST_QMAXLEVEL
  #  * # 1. Call p4est with coarse and refine 
  p4est.user_pointer = pointer(to_refine_to_coarse)
  refine_fn = @cfunction(refine_function, Cint, (Ptr{P4est.p4est_t}, P4est.p4est_topidx_t,  
                        Ptr{P4est.p4est_quadrant_t}))
  replace_quads_fn = @cfunction(replace_quads, Cvoid, 
                                (Ptr{P4est.p4est_t}, P4est.p4est_topidx_t,  
                                Cint, 
                                Ptr{Ptr{P4est.p4est_quadrant_t}} ,
                                Cint, 
                                Ptr{Ptr{P4est.p4est_quadrant_t}}))

  # @show replace_quads_fn
  P4est.p4est_refine_ext(p4est, recursive, allowed_level,
                        refine_fn, C_NULL,
                        replace_quads_fn)

  coarse_fn = @cfunction(coarse_function, Int32, (Ptr{P4est.p4est_t}, P4est.p4est_topidx_t, Ptr{Ptr{P4est.p4est_quadrant_t}}))
  Callbackorphans = 0

  P4est.p4est_coarsen_ext(p4est, recursive, Callbackorphans,
                          coarse_fn, C_NULL, replace_quads_fn);
  P4est.p4est_balance_ext(p4est, P4est.P4EST_CONNECT_FACE, C_NULL, replace_quads_fn)
  
  p4est.user_pointer = C_NULL
  # @show coarse_fn

# 


  # TODO 2. Need array Changes
  # if ChangesInfo [1, iElem] < 0 - The element was refined. 
  #     Then the ChangesInfo [2:4, iElem] := 0
  #     ChangesInfo [1, iElem] - is the number of the old Element 
  # if ChangesInfo [1, iElem] > 0:
  #   if ChangesInfo [2:4, iElem] > 0 
  #       The element was coarsened from Elements 
  #       ChangesInfo [1:4, iElem] - 4 coarsened Elements id 
  #   if ChangesInfo [2:4, iElem] = 0 
  #     The Element just changes his number => copy data to the new Element
  getchanges_fn = @cfunction(GetChanges, Cvoid, (Ptr{P4est.p4est_iter_volume_info_t}, Ptr{Cvoid}))
  local_num_quads = Int64(p4est.local_num_quadrants)
  ChangesInfo = zeros(Int64, 4,local_num_quads)
  ChangesInfo_ptr = pointer(ChangesInfo)
  P4est.p4est_iterate(p4est,  C_NULL, ChangesInfo_ptr, getchanges_fn, C_NULL, C_NULL)
  # @show ChangesInfo[:,:]
  # @assert 3 == 5
  # TODO  3. Rebuld mesh structure
  Connection = zeros(Int32, 11,local_num_quads)
  conn_ptr = pointer(Connection)
  CfaceIterate = @cfunction(faceIterate, Cvoid, (Ptr{P4est.p4est_iter_face_info_t}, Ptr{Cvoid}))
  P4est.p4est_iterate(p4est,  C_NULL, conn_ptr, C_NULL, CfaceIterate, C_NULL)
  t = mesh.tree
  QuadInfo = zeros(Int32, 4,local_num_quads)
  quadinfo_ptr = pointer(QuadInfo)
  CvolumeIterate = @cfunction(volumeIterate, Cvoid, (Ptr{P4est.p4est_iter_volume_info_t}, Ptr{Cvoid}))

  t.length = local_num_quads
  t.parent_ids[1:local_num_quads] .= 0
  t.child_ids[:, 1:local_num_quads] .= 0
  t.levels[1:local_num_quads] .= QuadInfo[2,1:local_num_quads]
  
 
  P4est.p4est_iterate(t.forest,  C_NULL, quadinfo_ptr, CvolumeIterate, C_NULL, C_NULL)
   # Get domain boundaries
  coordinates_min = parameter("coordinates_min")
  coordinates_max = parameter("coordinates_max")
   # t.coordinates[:, 1] .= t.center_level_0
  for id = 1:local_num_quads
    t.coordinates[1, id] = coordinates_min[1] + (coordinates_max[1] - coordinates_min[1]) / 512 *
        (QuadInfo[3, id] + 512/(2^(QuadInfo[2,id] + 1)))
    t.coordinates[2, id] = coordinates_min[2] +  (coordinates_max[2] - coordinates_min[2]) / 512 *
        (QuadInfo[4, id] + 512/(2^(QuadInfo[2,id] + 1)))
  end
  # dg = solver
  t.original_cell_ids[1:local_num_quads] .= QuadInfo[1,1:local_num_quads]
  for id = 1:local_num_quads
      t.neighbor_ids[1, id] = Connection[4,id]
      t.neighbor_ids[2, id] = Connection[6,id]
      t.neighbor_ids[3, id] = Connection[8,id]
      t.neighbor_ids[4, id] = Connection[10,id]
  end

  # # Get new list of leaf cells
  # leaf_cell_ids = leaf_cells(tree)


  # Initialize new elements container
  # elements = init_elements(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG))
  # n_elements = nelements(elements)
  
  # TODO: Create function to refine Coarse  
  @timeit timer() "solver" p4_adapt!(solver, mesh, ChangesInfo)
    
  # @show Connection
  # @assert 1 == 0
  # 4. 
# #### 
#   # Determine list of cells to refine or coarsen
#   to_refine = leaf_cell_ids[lambda .> indicator_threshold_refinement]
#   to_coarsen = leaf_cell_ids[lambda .< indicator_threshold_coarsening]

#   # Start by refining cells
#   @timeit timer() "refine" if !only_coarsen && !isempty(to_refine)
#     # Refine mesh
#     refined_original_cells = @timeit timer() "mesh" refine!(tree, to_refine)

#     # Refine solver
#     @timeit timer() "solver" refine!(solver, mesh, refined_original_cells)
#     if !isempty(passive_solvers)
#       @timeit timer() "passive solvers" for ps in passive_solvers
#         refine!(ps, mesh, refined_original_cells)
#       end
#     end
#   else
#     # If there is nothing to refine, create empty array for later use
#     refined_original_cells = Int[]
#   end

#   # Then, coarsen cells
#   @timeit timer() "coarsen" if !only_refine && !isempty(to_coarsen)
#     # Since the cells may have been shifted due to refinement, first we need to
#     # translate the old cell ids to the new cell ids
#     if !isempty(to_coarsen)
#       to_coarsen = original2refined(to_coarsen, refined_original_cells, mesh)
#     end

#     # Next, determine the parent cells from which the fine cells are to be
#     # removed, since these are needed for the coarsen! function. However, since
#     # we only want to coarsen if *all* child cells are marked for coarsening,
#     # we count the coarsening indicators for each parent cell and only coarsen
#     # if all children are marked as such (i.e., where the count is 2^ndims). At
#     # the same time, check if a cell is marked for coarsening even though it is
#     # *not* a leaf cell -> this can only happen if it was refined due to 2:1
#     # smoothing during the preceding refinement operation.
#     parents_to_coarsen = zeros(Int, length(tree))
#     for cell_id in to_coarsen
#       # If cell has no parent, it cannot be coarsened
#       if !has_parent(tree, cell_id)
#         continue
#       end

#       # If cell is not leaf (anymore), it cannot be coarsened
#       if !is_leaf(tree, cell_id)
#         continue
#       end

#       # Increase count for parent cell
#       parent_id = tree.parent_ids[cell_id]
#       parents_to_coarsen[parent_id] += 1
#     end

#     # Extract only those parent cells for which all children should be coarsened
#     to_coarsen = collect(1:length(parents_to_coarsen))[parents_to_coarsen .== 2^ndims(mesh)]

#     # Finally, coarsen mesh
#     coarsened_original_cells = @timeit timer() "mesh" coarsen!(tree, to_coarsen)

#     # Convert coarsened parent cell ids to the list of child cell ids that have
#     # been removed, since this is the information that is expected by the solver
#     removed_child_cells = zeros(Int, n_children_per_cell(tree) * length(coarsened_original_cells))
#     for (index, coarse_cell_id) in enumerate(coarsened_original_cells)
#       for child in 1:n_children_per_cell(tree)
#         removed_child_cells[n_children_per_cell(tree) * (index-1) + child] = coarse_cell_id + child
#       end
#     end

#     # Coarsen solver
#     @timeit timer() "solver" coarsen!(solver, mesh, removed_child_cells)
#     if !isempty(passive_solvers)
#       @timeit timer() "passive solvers" for ps in passive_solvers
#         coarsen!(ps, mesh, removed_child_cells)
#       end
#     end
#   else
#     # If there is nothing to coarsen, create empty array for later use
#     coarsened_original_cells = Int[]
#   end

  # Debug output
  globals[:verbose] && println("done (refined: $(length(refined_original_cells)), " *
                               "coarsened: $(length(coarsened_original_cells)), " *
                               "new number of cells/elements: " *
                               "$(length(tree))/$(solver.n_elements))")

  return false
  # Return true if there were any cells coarsened or refined, otherwise false
  return !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
end


# After refining cells, shift original cell ids to match new locations
# Note: Assumes sorted lists of original and refined cell ids!
# Note: `mesh` is only required to extract ndims
function original2refined(original_cell_ids, refined_original_cells, mesh)
  # Sanity check
  @assert issorted(original_cell_ids) "`original_cell_ids` not sorted"
  @assert issorted(refined_original_cells) "`refined_cell_ids` not sorted"

  # Create array with original cell ids (not yet shifted)
  shifted_cell_ids = collect(1:original_cell_ids[end])

  # Loop over refined original cells and apply shift for all following cells
  for cell_id in refined_original_cells
    # Only calculate shifts for cell ids that are relevant
    if cell_id > length(shifted_cell_ids)
      break
    end

    # Shift all subsequent cells by 2^ndims ids
    shifted_cell_ids[(cell_id + 1):end] .+= 2^ndims(mesh)
  end

  # Convert original cell ids to their shifted values
  return shifted_cell_ids[original_cell_ids]
end
