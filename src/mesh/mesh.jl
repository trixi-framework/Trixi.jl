module Mesh

include("trees.jl")

using ..Jul1dge
using ..Auxiliary: parameter, timer
using .Trees: Tree, refine!, refine_box!

using TimerOutputs: @timeit, print_timer

export generate_mesh


# Composite type to hold the actual tree in addition to other mesh-related data
# that is not strictly part of the tree.
mutable struct TreeMesh{D}
  tree::Tree{D}
  current_filename::String

  function TreeMesh{D}() where D
    # Verify that D is an integer
    @assert D isa Integer

    # Get domain boundaries
    coordinates_min = parameter("coordinates_min")
    coordinates_max = parameter("coordinates_max")

    # Domain length is calculated as the maximum length in any axis direction
    domain_center = @. (coordinates_min + coordinates_max) / 2
    domain_length = max(coordinates_max .- coordinates_min)

    # Create mesh
    m = new()
    m.tree =  Tree{D}(parameter("n_cells_max"), domain_center, domain_length)
    m.current_filename = ""

    return m
  end
end

# Constructor for passing the dimension as an argument
TreeMesh(::Val{D}) where D = TreeMesh{D}()


function generate_mesh()
  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(Val{ndim}())

  # Create initial refinement
  initial_refinement_level = parameter("initial_refinement_level")
  @timeit timer() "initial refinement" for l = 1:initial_refinement_level
    refine!(mesh.tree)
  end

  # Apply refinement patches
  @timeit timer() "refinement patches" for patch in parameter("refinement_patches", [])
    if patch["type"] == "box"
      refine_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown refinement patch type '$type_'")
    end
  end

  return mesh
end


end
