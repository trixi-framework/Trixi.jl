
include("tree.jl")

# Composite type to hold the actual tree in addition to other mesh-related data
# that is not strictly part of the tree.
mutable struct TreeMesh{D}
  tree::Tree{D}
  current_filename::String
  unsaved_changes::Bool

  function TreeMesh{D}(n_cells_max::Integer) where D
    # Verify that D is an integer
    @assert D isa Integer

    # Create mesh
    m = new()
    m.tree = Tree{D}(n_cells_max)
    m.current_filename = ""
    m.unsaved_changes = false

    return m
  end

  # TODO: Taal refactor, order of important arguments, use of n_cells_max?
  # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
  # TODO: Taal refactor, use NTuple instead of domain_center::AbstractArray{Float64}
  function TreeMesh{D}(n_cells_max::Integer, domain_center::AbstractArray{Float64},
                       domain_length, periodicity=true) where D
    # Verify that D is an integer
    @assert D isa Integer

    # Create mesh
    m = new()
    m.tree = Tree{D}(n_cells_max, domain_center, domain_length, periodicity)
    m.current_filename = ""
    m.unsaved_changes = false

    return m
  end
end

# Constructor for passing the dimension as an argument
TreeMesh(::Val{D}, args...) where D = TreeMesh{D}(args...)

# Constructor accepting a single number as center (as opposed to an array) for 1D
TreeMesh{1}(n::Int, center::Real, len::Real, periodicity=true) = TreeMesh{1}(n, [convert(Float64, center)], len, periodicity)

function TreeMesh(n_cells_max::Integer, domain_center::NTuple{NDIMS,Real}, domain_length::Real, periodicity=true) where {NDIMS}
  # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
  TreeMesh{NDIMS}(n_cells_max, [convert.(Float64, domain_center)...], convert(Float64, domain_length), periodicity)
end

function TreeMesh(coordinates_min::NTuple{NDIMS,Real}, coordinates_max::NTuple{NDIMS,Real};
                  n_cells_max=10^6,
                  periodicity=true,
                  initial_refinement_level=1,
                  refinement_patches=(),
                  coarsening_patches=(),
                  ) where {NDIMS}
  # Domain length is calculated as the maximum length in any axis direction
  domain_center = @. (coordinates_min + coordinates_max) / 2
  domain_length = maximum(coordinates_max .- coordinates_min)

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(n_cells_max, domain_center, domain_length, periodicity)

  # Create initial refinement
  @timeit timer() "initial refinement" for _ in 1:initial_refinement_level
    refine!(mesh.tree)
  end

  # Apply refinement patches
  @timeit timer() "refinement patches" for patch in refinement_patches
    # TODO: Taal refactor, use multiple dispatch
    if patch.type == "box"
      refine_box!(mesh.tree, patch.coordinates_min, patch.coordinates_max)
    else
      error("unknown refinement patch type '$(patch.type)'")
    end
  end

  # Apply coarsening patches
  @timeit timer() "coarsening patches" for patch in coarsening_patches
    # TODO: Taal refactor, use multiple dispatch
    if patch.type == "box"
      coarsen_box!(mesh.tree, patch.coordinates_min, patch.coordinates_max)
    else
      error("unknown coarsening patch type '$(patch.type)'")
    end
  end

  return mesh
end


function Base.show(io::IO, mesh::TreeMesh{NDIMS}) where {NDIMS}
  print(io, "TreeMesh{", NDIMS, "} with length ", mesh.tree.length)
end

function Base.show(io::IO, ::MIME"text/plain", mesh::TreeMesh{NDIMS}) where {NDIMS}
  println(io, "TreeMesh{", NDIMS, "} with")
  println(io, "- center_level_0: ", mesh.tree.center_level_0)
  println(io, "- length_level_0: ", mesh.tree.length_level_0)
  println(io, "- periodicity   : ", mesh.tree.periodicity)
  println(io, "- capacity      : ", mesh.tree.capacity)
  println(io, "- length        : ", mesh.tree.length)
end


@inline Base.ndims(mesh::TreeMesh) = ndims(mesh.tree)


# Generate initial mesh
function generate_mesh()
  # Get number of spatial dimensions
  ndims_ = parameter("ndims")

  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Get domain boundaries
  coordinates_min = parameter("coordinates_min")
  coordinates_max = parameter("coordinates_max")

  # Domain length is calculated as the maximum length in any axis direction
  domain_center = @. (coordinates_min + coordinates_max) / 2
  domain_length = maximum(coordinates_max .- coordinates_min)

  # By default, mesh is periodic in all dimensions
  periodicity = parameter("periodicity", true)

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(Val{ndims_}(), n_cells_max, domain_center,
                                             domain_length, periodicity)

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
      error("unknown refinement patch type '$(patch["type"])'")
    end
  end

  # Apply coarsening patches
  @timeit timer() "coarsening patches" for patch in parameter("coarsening_patches", [])
    if patch["type"] == "box"
      coarsen_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown coarsening patch type '$(patch["type"])'")
    end
  end

  return mesh
end

# TODO: Taal implement, loading meshes etc.

# Load existing mesh from file
function load_mesh(restart_filename)
  # Get number of spatial dimensions
  ndims_ = parameter("ndims")

  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(Val{ndims_}(), n_cells_max)

  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_filename)
  mesh.current_filename = filename
  mesh.unsaved_changes = false

  # Open mesh file
  h5open(filename, "r") do file
    # Set domain information
    mesh.tree.center_level_0 = read(attrs(file)["center_level_0"])
    mesh.tree.length_level_0 = read(attrs(file)["length_level_0"])
    mesh.tree.periodicity    = Tuple(read(attrs(file)["periodicity"]))

    # Set length
    n_cells = read(attrs(file)["n_cells"])
    resize!(mesh.tree, n_cells)

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
function get_restart_mesh_filename(restart_filename)
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
