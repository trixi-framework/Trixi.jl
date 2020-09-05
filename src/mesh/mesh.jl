
include("tree.jl")
include("parallel_tree.jl")

# Composite type to hold the actual tree in addition to other mesh-related data
# that is not strictly part of the tree.
mutable struct TreeMesh{NDIMS, TreeType}
  tree::TreeType
  current_filename::String
  unsaved_changes::Bool
  first_cell_by_domain::Vector{Int}
  n_cells_by_domain::Vector{Int}

  function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer) where {NDIMS, TreeType}
    # Verify that NDIMS is an integer
    @assert NDIMS == ndims(TreeType)

    # Create mesh
    m = new()
    m.tree = TreeType{NDIMS}(n_cells_max)
    m.current_filename = ""
    m.unsaved_changes = false
    m.first_cell_by_domain = Int[]
    m.n_cells_by_domain = Int[]

    return m
  end

  function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer, domain_center::AbstractArray{Float64},
                                     domain_length, periodicity=true) where{NDIMS, TreeType} 
    # Verify that NDIMS matches the tree
    @assert NDIMS == ndims(TreeType)

    # Create mesh
    m = new()
    m.tree = TreeType(n_cells_max, domain_center, domain_length, periodicity)
    m.current_filename = ""
    m.unsaved_changes = false
    m.first_cell_by_domain = Int[]
    m.n_cells_by_domain = Int[]

    return m
  end
end

# Constructor for passing the dimension and mesh type as an argument
function TreeMesh(::Val{NDIMS}, ::Val{TreeType}, args...) where {NDIMS, TreeType}
  return TreeMesh{NDIMS, TreeType}(args...)
end

# Constructor accepting a single number as center (as opposed to an array) for 1D
function TreeMesh{1, TreeType}(n::Int, center::Real, len::Real, periodicity=true) where TreeType
  return TreeMesh{1, TreeType}(n, [convert(Float64, center)], len, periodicity)
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
  if is_parallel()
    @timeit timer() "creation" mesh = TreeMesh(Val{ndims_}(), Val{ParallelTree{ndims_}}(),
                                               n_cells_max,
                                               domain_center, domain_length, periodicity)
  else
    @timeit timer() "creation" mesh = TreeMesh(Val{ndims_}(), Val{Tree{ndims_}}(), n_cells_max,
                                               domain_center, domain_length, periodicity)
  end

  # Create initial refinement
  initial_refinement_level = parameter("initial_refinement_level")
  @timeit timer() "initial refinement" for l = 1:initial_refinement_level
    refine!(mesh.tree)
  end

  # Partition mesh
  if is_parallel()
    partition(mesh)
  end

  # Apply refinement patches
  @timeit timer() "refinement patches" for patch in parameter("refinement_patches", [])
    is_parallel() && error("non-uniform meshes not supported in parallel")
    if patch["type"] == "box"
      refine_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown refinement patch type '$(patch["type"])'")
    end
  end

  # Apply coarsening patches
  @timeit timer() "coarsening patches" for patch in parameter("coarsening_patches", [])
    is_parallel() && error("non-uniform meshes not supported in parallel")
    if patch["type"] == "box"
      coarsen_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown coarsening patch type '$(patch["type"])'")
    end
  end

  return mesh
end


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


# Partition mesh using a static domain decomposition algorithm based on leaf cell count alone
# Return first cell id for each domain
function partition(mesh)
  # Determine number of leaf cells per domain
  leaves = leaf_cells(mesh.tree)
  n_leaves_per_domain = fill(div(length(leaves), n_domains()), n_domains())
  for d in 1:rem(length(leaves), n_domains())
    n_leaves_per_domain[d] += 1
  end
  @assert sum(n_leaves_per_domain) == length(leaves)

  # Assign domain ids to all cells such that all ancestors of each cell - if not yet assigned to a
  # domain - belong to the same domain
  mesh.first_cell_by_domain = similar(n_leaves_per_domain)
  mesh.n_cells_by_domain = similar(n_leaves_per_domain)

  leaf_count = 0
  last_id = leaves[n_leaves_per_domain[1]]
  mesh.first_cell_by_domain[1] = 1
  mesh.n_cells_by_domain[1] = last_id
  mesh.tree.domain_ids[1:last_id] .= 0
  for d in 2:length(n_leaves_per_domain)
    leaf_count += n_leaves_per_domain[d-1]
    last_id = leaves[leaf_count + n_leaves_per_domain[d]]
    mesh.first_cell_by_domain[d] = mesh.first_cell_by_domain[d-1] + mesh.n_cells_by_domain[d-1]
    mesh.n_cells_by_domain[d] = last_id - mesh.first_cell_by_domain[d] + 1
    mesh.tree.domain_ids[mesh.first_cell_by_domain[d]:last_id] .= d-1
  end

  @assert all(x->x >= 0, mesh.tree.domain_ids[1:length(mesh.tree)])
  @assert sum(mesh.n_cells_by_domain) == length(mesh.tree)

  return nothing
end
