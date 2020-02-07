module Mesh

include("trees.jl")

using ..Jul1dge
using ..Auxiliary: parameter
using .Trees: Tree, refine!

export generate_mesh


function generate_mesh()
  # Get domain boundaries
  coordinates_min = parameter("coordinates_min")
  coordinates_max = parameter("coordinates_max")

  # Domain length is calculated as the maximum length in any axis direction
  domain_center = @. (coordinates_min + coordinates_max) / 2
  domain_length = max(coordinates_max .- coordinates_min)

  # Create mesh
  mesh = Tree(Val{ndim}(), parameter("nnodesmax"), domain_center, domain_length)

  # Create initial refinement
  initial_refinement_level = parameter("initial_refinement_level")
  for l = 1:initial_refinement_level
    refine!(mesh)
  end

  return mesh
end


end
