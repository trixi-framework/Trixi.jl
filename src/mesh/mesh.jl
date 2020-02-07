module Mesh

include("trees.jl")

using ..Jul1dge
using ..Auxiliary: parameter, timer
using .Trees: Tree, refine!, refine_box!

using TimerOutputs: @timeit, print_timer

export generate_mesh


function generate_mesh()
  # Get domain boundaries
  coordinates_min = parameter("coordinates_min")
  coordinates_max = parameter("coordinates_max")

  # Domain length is calculated as the maximum length in any axis direction
  domain_center = @. (coordinates_min + coordinates_max) / 2
  domain_length = max(coordinates_max .- coordinates_min)

  # Create mesh
  @timeit timer() "creation" mesh = Tree(Val{ndim}(), parameter("nnodesmax"),
                                         domain_center, domain_length)

  # Create initial refinement
  initial_refinement_level = parameter("initial_refinement_level")
  @timeit timer() "initial refinement" for l = 1:initial_refinement_level
    refine!(mesh)
  end

  # Apply refinement patches
  @timeit timer() "refinement patches" for patch in parameter("refinement_patches", [])
    if patch["type"] == "box"
      refine_box!(mesh, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown refinement patch type '$type_'")
    end
  end

  return mesh
end


end
