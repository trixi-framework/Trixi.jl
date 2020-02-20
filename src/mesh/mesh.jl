module Mesh

include("trees.jl")

using ..Trixi
using ..Auxiliary: parameter, timer
using .Trees: Tree, refine!, refine_box!, append!, coarsen_box!

using TimerOutputs: @timeit, print_timer
using HDF5: h5open, attrs

export generate_mesh


# Composite type to hold the actual tree in addition to other mesh-related data
# that is not strictly part of the tree.
mutable struct TreeMesh{D}
  tree::Tree{D}
  current_filename::String

  function TreeMesh{D}(n_cells_max::Integer) where D
    # Verify that D is an integer
    @assert D isa Integer

    # Create mesh
    m = new()
    m.tree = Tree{D}(n_cells_max)
    m.current_filename = ""

    return m
  end

  function TreeMesh{D}(n_cells_max::Integer,
                       domain_center::AbstractArray{Float64}, domain_length) where D
    # Verify that D is an integer
    @assert D isa Integer

    # Create mesh
    m = new()
    m.tree = Tree{D}(n_cells_max, domain_center, domain_length)
    m.current_filename = ""

    return m
  end
end

# Constructor for passing the dimension as an argument
TreeMesh(::Val{D}, args...) where D = TreeMesh{D}(args...)

# Constructor accepting a single number as center (as opposed to an array) for 1D
TreeMesh{1}(n::Int, center::Real, len::Real) = TreeMesh{1}(n, [convert(Float64, center)], len)


# Generate initial mesh
function generate_mesh()
  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Get domain boundaries
  coordinates_min = parameter("coordinates_min")
  coordinates_max = parameter("coordinates_max")

  # Domain length is calculated as the maximum length in any axis direction
  domain_center = @. (coordinates_min + coordinates_max) / 2
  domain_length = max(coordinates_max .- coordinates_min)

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(Val{ndim}(), n_cells_max, domain_center, domain_length)

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

  # Apply coarsening patches
  @timeit timer() "coarsening patches" for patch in parameter("coarsening_patches", [])
    if patch["type"] == "box"
      coarsen_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown coarsening patch type '$type_'")
    end
  end

  return mesh
end


# Load existing mesh from file
function load_mesh(restart_filename::String)
  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(Val{ndim}(), n_cells_max)

  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_filename)
  mesh.current_filename = filename

  # Open mesh file
  h5open(filename, "r") do file
    # Set domain information
    mesh.tree.center_level_0 = read(attrs(file)["center_level_0"])
    mesh.tree.length_level_0 = read(attrs(file)["length_level_0"])

    # Set length
    n_cells = read(attrs(file)["n_cells"])
    append!(mesh.tree, n_cells)

    # Read in data
    mesh.tree.parent_ids[1:n_cells] = read(file["parent_ids"])
    mesh.tree.child_ids[:, 1:n_cells] = read(file["child_ids"])
    mesh.tree.neighbor_ids[:, 1:n_cells] = read(file["neighbor_ids"])
    mesh.tree.levels[1:n_cells] = read(file["levels"])
    mesh.tree.coordinates[:, 1:n_cells] = read(file["coordinates"])
  end

  return mesh
end


# Obtain the mesh filename from a restart file
function get_restart_mesh_filename(restart_filename::String)
  # Get directory name
  dirname, _ = splitdir(restart_filename)

  # Read mesh filename from restart file
  mesh_file = ""
  h5open(restart_filename, "r") do file
    mesh_file = read(attrs(file)["mesh_file"])
  end

  # Construct and return filename
  return joinpath(dirname, mesh_file)
end


end
