
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

  # Determine indicator value
  lambda = @timeit timer() "indicator" calc_amr_indicator(solver, mesh, time)

  # Get list of current leaf cells
  leaf_cell_ids = leaf_cells(tree)
  @assert length(lambda) == length(leaf_cell_ids) ("Indicator and leaf cell arrays have " *
                                                   "different length")

  # Set thresholds for refinement and coarsening
  indicator_threshold_refinement = parameter("indicator_threshold_refinement",  0.5)
  indicator_threshold_coarsening = parameter("indicator_threshold_coarsening", -0.5)

  # Determine list of cells to refine or coarsen
  to_refine = leaf_cell_ids[lambda .> indicator_threshold_refinement]
  to_coarsen = leaf_cell_ids[lambda .< indicator_threshold_coarsening]

  # Start by refining cells
  @timeit timer() "refine" if !only_coarsen && !isempty(to_refine)
    # Refine mesh
    refined_original_cells = @timeit timer() "mesh" refine!(tree, to_refine)

    # Refine solver
    @timeit timer() "solver" refine!(solver, mesh, refined_original_cells)
    if !isempty(passive_solvers)
      @timeit timer() "passive solvers" for ps in passive_solvers
        refine!(ps, mesh, refined_original_cells)
      end
    end
  else
    # If there is nothing to refine, create empty array for later use
    refined_original_cells = Int[]
  end

  # Then, coarsen cells
  @timeit timer() "coarsen" if !only_refine && !isempty(to_coarsen)
    # Since the cells may have been shifted due to refinement, first we need to
    # translate the old cell ids to the new cell ids
    if !isempty(to_coarsen)
      to_coarsen = original2refined(to_coarsen, refined_original_cells)
    end

    # Next, determine the parent cells from which the fine cells are to be
    # removed, since these are needed for the coarsen! function. However, since
    # we only want to coarsen if *all* child cells are marked for coarsening,
    # we count the coarsening indicators for each parent cell and only coarsen
    # if all children are marked as such (i.e., where the count is 2^ndim). At
    # the same time, check if a cell is marked for coarsening even though it is
    # *not* a leaf cell -> this can only happen if it was refined due to 2:1
    # smoothing during the preceding refinement operation.
    parents_to_coarsen = zeros(Int, length(tree))
    for cell_id in to_coarsen
      # If cell has no parent, it cannot be coarsened
      if !has_parent(tree, cell_id)
        continue
      end

      # If cell is not leaf (anymore), it cannot be coarsened
      if !is_leaf(tree, cell_id)
        continue
      end

      # Increase count for parent cell
      parent_id = tree.parent_ids[cell_id]
      parents_to_coarsen[parent_id] += 1
    end

    # Extract only those parent cells for which all children should be coarsened
    to_coarsen = collect(1:length(parents_to_coarsen))[parents_to_coarsen .== 2^ndim]

    # Finally, coarsen mesh
    coarsened_original_cells = @timeit timer() "mesh" coarsen!(tree, to_coarsen)

    # Convert coarsened parent cell ids to the list of child cell ids that have
    # been removed, since this is the information that is expected by the solver
    removed_child_cells = zeros(Int, n_children_per_cell(tree) * length(coarsened_original_cells))
    for (index, coarse_cell_id) in enumerate(coarsened_original_cells)
      for child in 1:n_children_per_cell(tree)
        removed_child_cells[n_children_per_cell(tree) * (index-1) + child] = coarse_cell_id + child
      end
    end

    # Coarsen solver
    @timeit timer() "solver" coarsen!(solver, mesh, removed_child_cells)
    if !isempty(passive_solvers)
      @timeit timer() "passive solvers" for ps in passive_solvers
        coarsen!(ps, mesh, removed_child_cells)
      end
    end
  else
    # If there is nothing to coarsen, create empty array for later use
    coarsened_original_cells = Int[]
  end

  # Debug output
  globals[:verbose] && println("done (refined: $(length(refined_original_cells)), " *
                               "coarsened: $(length(coarsened_original_cells)), " *
                               "new number of cells/elements: " *
                               "$(length(tree))/$(solver.n_elements))")

  # Return true if there were any cells coarsened or refined, otherwise false
  return !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
end


# After refining cells, shift original cell ids to match new locations
# Note: Assumes sorted lists of original and refined cell ids!
function original2refined(original_cell_ids::AbstractVector{Int},
                          refined_original_cells::AbstractVector{Int})
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

    # Shift all subsequent cells by 2^ndim ids
    shifted_cell_ids[(cell_id + 1):end] .+= 2^ndim
  end

  # Convert original cell ids to their shifted values
  return shifted_cell_ids[original_cell_ids]
end
