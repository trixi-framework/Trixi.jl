include("abstract_tree.jl")
include("serial_tree.jl")
include("parallel_tree.jl")

# Composite type to hold the actual tree in addition to other mesh-related data
# that is not strictly part of the tree.
# The mesh is really just about the connectivity, size, and location of the individual
# tree nodes. Neighbor information between interfaces or the large sides for mortars is
# something that is solver-specific and that might not be needed by all solvers (or in a
# different form). Also, these data values can be performance critical, so a mesh would
# have to store them for all solvers in an efficient way - OTOH, different solvers might
# use different cells of a shared mesh, so "efficient" is again solver dependent.
mutable struct TreeMesh{NDIMS, TreeType<:AbstractTree{NDIMS}}
  tree::TreeType
  current_filename::String
  unsaved_changes::Bool
  first_cell_by_rank::OffsetVector{Int, Vector{Int}}
  n_cells_by_rank::OffsetVector{Int, Vector{Int}}

  function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer) where {NDIMS, TreeType<:AbstractTree{NDIMS}}
    # Create mesh
    m = new()
    m.tree = TreeType(n_cells_max)
    m.current_filename = ""
    m.unsaved_changes = true
    m.first_cell_by_rank = OffsetVector(Int[], 0)
    m.n_cells_by_rank = OffsetVector(Int[], 0)

    return m
  end

  # TODO: Taal refactor, order of important arguments, use of n_cells_max?
  # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
  # TODO: Taal refactor, use NTuple instead of domain_center::AbstractArray{Float64}
  function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer, domain_center::AbstractArray{Float64},
                                     domain_length, periodicity=true) where {NDIMS, TreeType<:AbstractTree{NDIMS}}
    @assert NDIMS isa Integer && NDIMS > 0

    # Create mesh
    m = new()
    m.tree = TreeType(n_cells_max, domain_center, domain_length, periodicity)
    m.current_filename = ""
    m.unsaved_changes = true
    m.first_cell_by_rank = OffsetVector(Int[], 0)
    m.n_cells_by_rank = OffsetVector(Int[], 0)

    return m
  end
end

const TreeMesh1D = TreeMesh{1, TreeType} where {TreeType <: AbstractTree{1}}
const TreeMesh2D = TreeMesh{2, TreeType} where {TreeType <: AbstractTree{2}}
const TreeMesh3D = TreeMesh{3, TreeType} where {TreeType <: AbstractTree{3}}

const SerialTreeMesh{NDIMS}   = TreeMesh{NDIMS, <:SerialTree{NDIMS}}
const ParallelTreeMesh{NDIMS} = TreeMesh{NDIMS, <:ParallelTree{NDIMS}}

@inline mpi_parallel(mesh::SerialTreeMesh) = Val(false)
@inline mpi_parallel(mesh::ParallelTreeMesh) = Val(true)

partition!(mesh::SerialTreeMesh) = nothing


# Constructor for passing the dimension and mesh type as an argument
TreeMesh(::Type{TreeType}, args...) where {NDIMS, TreeType<:AbstractTree{NDIMS}} = TreeMesh{NDIMS, TreeType}(args...)

# Constructor accepting a single number as center (as opposed to an array) for 1D
function TreeMesh{1, TreeType}(n::Int, center::Real, len::Real, periodicity=true) where {TreeType<:AbstractTree{1}}
  return TreeMesh{1, TreeType}(n, [convert(Float64, center)], len, periodicity)
end

function TreeMesh{NDIMS, TreeType}(n_cells_max::Integer, domain_center::NTuple{NDIMS,Real}, domain_length::Real, periodicity=true) where {NDIMS, TreeType<:AbstractTree{NDIMS}}
  # TODO: Taal refactor, allow other RealT for the mesh, not just Float64
  TreeMesh{NDIMS, TreeType}(n_cells_max, [convert.(Float64, domain_center)...], convert(Float64, domain_length), periodicity)
end

function TreeMesh(coordinates_min::NTuple{NDIMS,Real}, coordinates_max::NTuple{NDIMS,Real};
                  n_cells_max,
                  periodicity=true,
                  initial_refinement_level,
                  refinement_patches=(),
                  coarsening_patches=(),
                  ) where {NDIMS}
  # check arguments
  if !(n_cells_max isa Integer && n_cells_max > 0)
    throw(ArgumentError("`n_cells_max` must be a positive integer (provided `n_cells_max = $n_cells_max`)"))
  end
  if !(initial_refinement_level isa Integer && initial_refinement_level >= 0)
    throw(ArgumentError("`initial_refinement_level` must be a non-negative integer (provided `initial_refinement_level = $initial_refinement_level`)"))
  end

  # Domain length is calculated as the maximum length in any axis direction
  domain_center = @. (coordinates_min + coordinates_max) / 2
  domain_length = maximum(coordinates_max .- coordinates_min)

  # TODO: MPI, create nice interface for a parallel tree/mesh
  if mpi_parallel() === Val(true)
    TreeType = ParallelTree{NDIMS}
  else
    TreeType = SerialTree{NDIMS}
  end

  # Create mesh
  mesh = @timeit timer() "creation" TreeMesh{NDIMS, TreeType}(n_cells_max, domain_center, domain_length, periodicity)

  # Create initial refinement
  @timeit timer() "initial refinement" for _ in 1:initial_refinement_level
    refine!(mesh.tree)
  end

  # Apply refinement patches
  @timeit timer() "refinement patches" for patch in refinement_patches
    mpi_isparallel() && error("non-uniform meshes not supported in parallel")
    # TODO: Taal refactor, use multiple dispatch?
    if patch.type == "box"
      refine_box!(mesh.tree, patch.coordinates_min, patch.coordinates_max)
    else
      error("unknown refinement patch type '$(patch.type)'")
    end
  end

  # Apply coarsening patches
  @timeit timer() "coarsening patches" for patch in coarsening_patches
    mpi_isparallel() && error("non-uniform meshes not supported in parallel")
    # TODO: Taal refactor, use multiple dispatch
    if patch.type == "box"
      coarsen_box!(mesh.tree, patch.coordinates_min, patch.coordinates_max)
    else
      error("unknown coarsening patch type '$(patch.type)'")
    end
  end

  # Partition the mesh among multiple MPI ranks (does nothing if run in serial)
  partition!(mesh)

  return mesh
end

function TreeMesh(coordinates_min::Real, coordinates_max::Real; kwargs...)
  TreeMesh((coordinates_min,), (coordinates_max,); kwargs...)
end


function Base.show(io::IO, mesh::TreeMesh{NDIMS, TreeType}) where {NDIMS, TreeType}
  print(io, "TreeMesh{", NDIMS, ", ", TreeType, "} with length ", mesh.tree.length)
end

function Base.show(io::IO, ::MIME"text/plain", mesh::TreeMesh{NDIMS, TreeType}) where {NDIMS, TreeType}
  println(io, "TreeMesh{", NDIMS, ", ", TreeType, "} with")
  println(io, "- center_level_0: ", mesh.tree.center_level_0)
  println(io, "- length_level_0: ", mesh.tree.length_level_0)
  println(io, "- periodicity   : ", mesh.tree.periodicity)
  println(io, "- capacity      : ", mesh.tree.capacity)
  print(io,   "- length        : ", mesh.tree.length)
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
  if mpi_isparallel()
    tree_type = ParallelTree{ndims_}
  else
    tree_type = SerialTree{ndims_}
  end
  mesh = @timeit timer() "creation" TreeMesh(tree_type, n_cells_max, domain_center,
                                             domain_length, periodicity)

  # Create initial refinement
  initial_refinement_level = parameter("initial_refinement_level")
  @timeit timer() "initial refinement" for l = 1:initial_refinement_level
    refine!(mesh.tree)
  end

  # Apply refinement patches
  @timeit timer() "refinement patches" for patch in parameter("refinement_patches", [])
    mpi_isparallel() && error("non-uniform meshes not supported in parallel")
    if patch["type"] == "box"
      refine_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown refinement patch type '$(patch["type"])'")
    end
  end

  # Apply coarsening patches
  @timeit timer() "coarsening patches" for patch in parameter("coarsening_patches", [])
    mpi_isparallel() && error("non-uniform meshes not supported in parallel")
    if patch["type"] == "box"
      coarsen_box!(mesh.tree, patch["coordinates_min"], patch["coordinates_max"])
    else
      error("unknown coarsening patch type '$(patch["type"])'")
    end
  end

  # Partition the mesh among multiple MPI ranks (does nothing if run in serial)
  partition!(mesh)

  return mesh
end

# TODO: Taal remove the function below
# Load existing mesh from file
load_mesh_old(restart_filename) = load_mesh_old(restart_filename, mpi_parallel())

function load_mesh_old(restart_filename, mpi_parallel::Val{false})
  # Get number of spatial dimensions
  ndims_ = parameter("ndims")

  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Create mesh
  mesh = @timeit timer() "creation" TreeMesh(SerialTree{ndims_}, n_cells_max)

  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_filename, Val(false))
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
function get_restart_mesh_filename(restart_filename, mpi_parallel::Val{false})
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


include("parallel.jl")
include("mesh_io.jl")
