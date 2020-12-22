
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
  # println("Refine 2")
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
  #   # Determine list of cells to refine or coarsen
  #   to_refine = leaf_cell_ids[lambda .> indicator_threshold_refinement]
  #   to_coarsen = leaf_cell_ids[lambda .< indicator_threshold_coarsening]
  to_refine_to_coarse = zeros(Int32, 1,local_num_quadrants)
  for i = 1:local_num_quadrants
    if lambda[i] > indicator_threshold_refinement && !only_coarsen
      to_refine_to_coarse[i] = 1 # refine
    end
    if lambda[i] < indicator_threshold_coarsening && !only_refine
      to_refine_to_coarse[i] = -1 # coarse
    end
  end
  to_refine_to_coarse_ptr = pointer(to_refine_to_coarse)

  # *  # 0. Set inner old quad data to Zero
  p4est = mesh.tree.forest
  CvolumeIterate = @cfunction(setOldIdtoZero, Cvoid, (Ptr{P4est.p4est_iter_volume_info_t}, Ptr{Cvoid}))
  P4est.p4est_iterate( p4est,  C_NULL, C_NULL, CvolumeIterate, C_NULL, C_NULL)

  recursive = 0
  # Set maximal Level from the toml file.
  allowed_level = parameter("max_refinement_level") #P4est.P4EST_QMAXLEVEL
  # * use innere p4est pointer p4est->user_pointer to pass array to the 
  # * refine, coarse and balance functions.
  p4est.user_pointer = pointer(to_refine_to_coarse)
  # * Define the functions for refine and replace
  refine_fn = @cfunction(refine_function, Cint, (Ptr{P4est.p4est_t}, P4est.p4est_topidx_t,  
                        Ptr{P4est.p4est_quadrant_t}))
  replace_quads_fn = @cfunction(replace_quads, Cvoid, 
                                (Ptr{P4est.p4est_t}, P4est.p4est_topidx_t,  
                                Cint, 
                                Ptr{Ptr{P4est.p4est_quadrant_t}} ,
                                Cint, 
                                Ptr{Ptr{P4est.p4est_quadrant_t}}))

  #  * Start by refining cells
  P4est.p4est_refine_ext(p4est, recursive, allowed_level,
                        refine_fn, C_NULL,
                        replace_quads_fn)

  coarse_fn = @cfunction(coarse_function, Int32, (Ptr{P4est.p4est_t}, P4est.p4est_topidx_t, Ptr{Ptr{P4est.p4est_quadrant_t}}))
  Callbackorphans = 0
  #  * Then, coarsen cells
  P4est.p4est_coarsen_ext(p4est, recursive, Callbackorphans,
                          coarse_fn, C_NULL, replace_quads_fn);
  #  * The last operation - 2-to-1 balance
  P4est.p4est_balance_ext(p4est, P4est.P4EST_CONNECT_FACE, C_NULL, replace_quads_fn)
  p4est.user_pointer = C_NULL



  # * 2. Get information about elements' changes 
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
  
  # * 3. Rebuld mesh structure
  Connection = p4_get_connections(p4est)

  t = mesh.tree
  # * Get information about Quad info (level, coordinates ...)
  QuadInfo = p4_get_quadinfo(p4est)
  
  t.length = local_num_quads
  t.parent_ids[1:local_num_quads] .= 0
  t.child_ids[:, 1:local_num_quads] .= 0
  t.levels[1:local_num_quads] .= QuadInfo[2,1:local_num_quads]
  
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

  t.original_cell_ids[1:local_num_quads] .= QuadInfo[1,1:local_num_quads]

  for id = 1:local_num_quads
      t.neighbor_ids[1, id] = Connection[4,id]
      t.neighbor_ids[2, id] = Connection[6,id]
      t.neighbor_ids[3, id] = Connection[8,id]
      t.neighbor_ids[4, id] = Connection[10,id]
  end

  # Initialize new elements container

  
  # TODO: Create function to refine Coarse  
  @timeit timer() "solver" p4_adapt!(solver, mesh, ChangesInfo)

  # Debug output
  # globals[:verbose] && println("done (refined: $(length(refined_original_cells)), " *
  #                              "coarsened: $(length(coarsened_original_cells)), " *
  #                              "new number of cells/elements: " *
  #                              "$(length(tree))/$(solver.n_elements))")
  # * Additional check if the mesh was changed. See desrciotion if p4est_checksum function
  check = P4est.p4est_checksum(p4est)
  @show check
  @show mesh.tree.forestchecksum
  mesh_changed = check != mesh.tree.forestchecksum
  mesh.tree.forestchecksum = check
  @show mesh_changed
  return mesh_changed
  
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
