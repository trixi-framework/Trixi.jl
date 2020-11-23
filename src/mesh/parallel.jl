# Partition mesh using a static domain decomposition algorithm based on leaf cell count alone
function partition!(mesh::ParallelTreeMesh)
  # Determine number of leaf cells per rank
  leaves = leaf_cells(mesh.tree)
  @assert length(leaves) > mpi_nranks()
  n_leaves_per_rank = OffsetArray(fill(div(length(leaves), mpi_nranks()), mpi_nranks()),
                                  0:(mpi_nranks() - 1))
  for d in 0:(rem(length(leaves), mpi_nranks()) - 1)
    n_leaves_per_rank[d] += 1
  end
  @assert sum(n_leaves_per_rank) == length(leaves)

  # Assign MPI ranks to all cells such that all ancestors of each cell - if not yet assigned to a
  # rank - belong to the same rank
  mesh.first_cell_by_rank = similar(n_leaves_per_rank)
  mesh.n_cells_by_rank = similar(n_leaves_per_rank)

  leaf_count = 0
  last_id = leaves[n_leaves_per_rank[0]]
  mesh.first_cell_by_rank[0] = 1
  mesh.n_cells_by_rank[0] = last_id
  mesh.tree.mpi_ranks[1:last_id] .= 0
  for d in 1:(length(n_leaves_per_rank)-1)
    leaf_count += n_leaves_per_rank[d-1]
    last_id = leaves[leaf_count + n_leaves_per_rank[d]]
    mesh.first_cell_by_rank[d] = mesh.first_cell_by_rank[d-1] + mesh.n_cells_by_rank[d-1]
    mesh.n_cells_by_rank[d] = last_id - mesh.first_cell_by_rank[d] + 1
    mesh.tree.mpi_ranks[mesh.first_cell_by_rank[d]:last_id] .= d
  end

  @assert all(x->x >= 0, mesh.tree.mpi_ranks[1:length(mesh.tree)])
  @assert sum(mesh.n_cells_by_rank) == length(mesh.tree)

  return nothing
end


function get_restart_mesh_filename(restart_filename, mpi_parallel::Val{true})
  # Get directory name
  dirname, _ = splitdir(restart_filename)

  if mpi_isroot()
    # Read mesh filename from restart file
    mesh_file = ""
    h5open(restart_filename, "r") do file
      mesh_file = read(attributes(file)["mesh_file"])
    end

    buffer = Vector{UInt8}(mesh_file)
    MPI.Bcast!(Ref(length(buffer)), mpi_root(), mpi_comm())
    MPI.Bcast!(buffer, mpi_root(), mpi_comm())
  else # non-root ranks
    count = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())
    buffer = Vector{UInt8}(undef, count[])
    MPI.Bcast!(buffer, mpi_root(), mpi_comm())
    mesh_file = String(buffer)
  end

  # Construct and return filename
  return joinpath(dirname, mesh_file)
end
