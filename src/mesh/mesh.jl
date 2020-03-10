module Mesh

include("trees.jl")

using ..Trixi
using ..Auxiliary: parameter, timer
using ..Auxiliary.Containers: append!
using .Trees: Tree, refine!, refine_box!, coarsen_box!, leaf_cells, minimum_level, maximum_level
using ..Parallel: n_domains, @mpi_root, is_parallel, mpi_root, Bcast!, comm, is_mpi_root, domain_id

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
  domain_length = maximum(coordinates_max .- coordinates_min)

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

  # Initialize mesh parallelization
  init_parallel(mesh)

  return mesh
end


# Load existing mesh from file
function load_mesh(restart_filename::String)
  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(Val{ndim}(), n_cells_max)

  # Read mesh only on root domain and distribute data to all domains
  if is_mpi_root()
    # Determine mesh filename
    filename = get_restart_mesh_filename(restart_filename)
    mesh.current_filename = filename

    # Open & read mesh file
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

    if is_parallel()
      # Exchange filename as byte array
      buffer = Vector{UInt8}(filename)
      count = Int[length(buffer)]
      Bcast!(count, mpi_root(), comm())
      Bcast!(buffer, mpi_root(), comm())

      # Exchange domain information
      info = Float64[mesh.tree.center_level_0..., mesh.tree.length_level_0]
      Bcast!(info, mpi_root(), comm())

      # Exchange number of cells
      count[1] = n_cells
      Bcast!(count, mpi_root(), comm())

      # Exchange data
      Bcast!(mesh.tree.parent_ids[1:n_cells], mpi_root(), comm())
      Bcast!(mesh.tree.child_ids[:, 1:n_cells], mpi_root(), comm())
      Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), comm())
      Bcast!(mesh.tree.levels[1:n_cells], mpi_root(), comm())
      Bcast!(mesh.tree.coordinates[:, 1:n_cells], mpi_root(), comm())
    end
  else
    # Exchange filename as byte array
    count = Int[0]
    Bcast!(count, mpi_root(), comm())
    buffer = Vector{UInt8}(undef, count[1])
    Bcast!(buffer, mpi_root(), comm())
    mesh.current_filename = String(buffer)

    # Exchange domain information
    info = Vector{Float64}(undef, ndim + 1)
    Bcast!(info, mpi_root(), comm())
    mesh.tree.center_level_0 = info[1:ndim]
    mesh.tree.length_level_0 = info[ndim + 1]

    # Exchange number of cells
    count[1] = 0
    Bcast!(count, mpi_root(), comm())
    n_cells = count[1]
    append!(mesh.tree, n_cells)

    # Exchange data
    @views Bcast!(mesh.tree.parent_ids[1:n_cells], mpi_root(), comm())
    @views Bcast!(mesh.tree.child_ids[:, 1:n_cells], mpi_root(), comm())
    @views Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), comm())
    @views Bcast!(mesh.tree.levels[1:n_cells], mpi_root(), comm())
    @views Bcast!(mesh.tree.coordinates[:, 1:n_cells], mpi_root(), comm())
  end

  # Initialize mesh parallelization
  init_parallel(mesh)

  return mesh
end


function init_parallel(mesh::TreeMesh)
  if is_parallel()
    # Perform mesh sanity checks for MPI
    @assert minimum_level(mesh.tree) == maximum_level(mesh.tree) "MPI + non-unform mesh not yet supported"
    @assert n_domains() & (n_domains()-1) == 0 "Number of MPI ranks must be a power of two"
    @assert 4^minimum_level(mesh.tree) >= n_domains() "Not enough domains for simple partitioning"

    # Set domain_id for each cell (simple partitioning with equally sized domains at the leaf level)
    # FIXME: Implement proper partitioning

    # Reset all cells to belong to root domain
    mesh.tree.domain_ids .= 0

    # Divide up leaf_cell_ids
    leaf_cell_ids = leaf_cells(mesh.tree)
    cells_per_domain = div(length(leaf_cell_ids), n_domains())
    for (idx, cell_id) in enumerate(leaf_cell_ids)
      mesh.tree.domain_ids[cell_id] = div(idx - 1, cells_per_domain)
    end

    # Another sanity check: count cells
    leaf_cells_per_domain = zeros(Int, n_domains())
    for cell_id in leaf_cell_ids
      leaf_cells_per_domain[mesh.tree.domain_ids[cell_id] + 1] += 1
    end
    @assert all(leaf_cells_per_domain .== cells_per_domain) "Leaf cells are not equally distributed"
  else
    # If non-parallel run (no MPI or just one rank), assign all cells to rank 0
    mesh.tree.domain_ids .= 0
  end
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
