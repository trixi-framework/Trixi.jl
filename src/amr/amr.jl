module AMR

using ..Trixi
using ..Auxiliary: parameter, timer
using ..Auxiliary.Containers: append!
using ..Mesh: TreeMesh
using ..Mesh.Trees: leaf_cells
using ..Solvers: AbstractSolver, calc_amr_indicator
import ..Mesh # to use refine!
import ..Mesh.Trees # to use refine!
import ..Solvers # to use refine!

using TimerOutputs: @timeit, print_timer
using HDF5: h5open, attrs


export adapt!


function adapt!(mesh::TreeMesh, solver::AbstractSolver, only_refine=false)
  print("Begin adaptation...")
  # Alias for convenience
  tree = mesh.tree

  # Determine indicator value
  lambda = calc_amr_indicator(solver, mesh)

  # Get list of current leaf cells
  leaf_cell_ids = leaf_cells(tree)
  @assert length(lambda) == length(leaf_cell_ids) ("Indicator and leaf cell arrays have " *
                                                   "different length")

  # Set thresholds for refinement and coarsening
  refinement_threshold = parameter("refinement_threshold",  0.5)
  coarsening_threshold = parameter("coarsening_threshold", -0.5)

  # Determine cells that should be refined or coarsened
  to_refine = leaf_cell_ids[lambda .> refinement_threshold]
  to_coarsen = leaf_cell_ids[lambda .< coarsening_threshold]

  if !isempty(to_refine)
    # Refine cells
    refined_original_cells = Mesh.Trees.refine!(tree, to_refine)

    # Refine elements
    Solvers.refine!(solver, mesh, refined_original_cells)
  else
    refined_original_cells = Int[]
  end

  if !only_refine && !isempty(to_coarsen)
    # ...
  else
    coarsened_original_cells = Int[]
  end

  println("done (refined: $(length(refined_original_cells)), " *
                 "coarsened: $(length(coarsened_original_cells)), " *
                 "new number of cells/elements: $(length(tree))/$(solver.n_elements))")

  return !isempty(refined_original_cells) || !isempty(coarsened_original_cells)
end


end # module AMR
