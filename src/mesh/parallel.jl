function load_mesh(restart_filename, mpi_parallel::Val{true})
  # Get number of spatial dimensions
  ndims_ = parameter("ndims")

  # Get maximum number of cells that should be supported
  n_cells_max = parameter("n_cells_max")

  # Create mesh
  @timeit timer() "creation" mesh = TreeMesh(ParallelTree{ndims_}, n_cells_max)

  # Determine mesh filename
  if is_mpi_root()
    filename = get_restart_mesh_filename(restart_filename)
    buffer = Vector{UInt8}(filename)
    MPI.Bcast!(Ref(length(buffer)), mpi_root(), mpi_comm())
    MPI.Bcast!(buffer, mpi_root(), mpi_comm())
  else # non-root ranks
    count = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())
    buffer = Vector{UInt8}(undef, count[])
    MPI.Bcast!(buffer, mpi_root(), mpi_comm())
    filename = String(buffer)
  end
  mesh.current_filename = filename
  mesh.unsaved_changes = false

  # Read mesh file
  if is_mpi_root()
    h5open(filename, "r") do file
      # Set domain information
      mesh.tree.center_level_0 = read(attrs(file)["center_level_0"])
      mesh.tree.length_level_0 = read(attrs(file)["length_level_0"])
      mesh.tree.periodicity    = Tuple(read(attrs(file)["periodicity"]))
      MPI.Bcast!(collect(mesh.tree.center_level_0), mpi_root(), mpi_comm())
      MPI.Bcast!(collect(mesh.tree.length_level_0), mpi_root(), mpi_comm())
      MPI.Bcast!(collect(mesh.tree.periodicity),    mpi_root(), mpi_comm())

      # Set length
      n_cells = read(attrs(file)["n_cells"])
      MPI.Bcast!(Ref(n_cells), mpi_root(), mpi_comm())
      resize!(mesh.tree, n_cells)

      # Read in data
      mesh.tree.parent_ids[1:n_cells] = read(file["parent_ids"])
      mesh.tree.child_ids[:, 1:n_cells] = read(file["child_ids"])
      mesh.tree.neighbor_ids[:, 1:n_cells] = read(file["neighbor_ids"])
      mesh.tree.levels[1:n_cells] = read(file["levels"])
      mesh.tree.coordinates[:, 1:n_cells] = read(file["coordinates"])
      @views MPI.Bcast!(mesh.tree.parent_ids[1:n_cells],      mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.child_ids[:, 1:n_cells],    mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.levels[1:n_cells],          mpi_root(), mpi_comm())
      @views MPI.Bcast!(mesh.tree.coordinates[:, 1:n_cells],  mpi_root(), mpi_comm())
    end
  else # non-root ranks
    # Set domain information
    mesh.tree.center_level_0 = MPI.Bcast!(collect(mesh.tree.center_level_0), mpi_root(), mpi_comm())
    mesh.tree.length_level_0 = MPI.Bcast!(collect(mesh.tree.length_level_0), mpi_root(), mpi_comm())[1]
    mesh.tree.periodicity    = Tuple(MPI.Bcast!(collect(mesh.tree.periodicity),    mpi_root(), mpi_comm()))

    # Set length
    n_cells = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
    resize!(mesh.tree, n_cells)

    # Read in data
    @views MPI.Bcast!(mesh.tree.parent_ids[1:n_cells],      mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.child_ids[:, 1:n_cells],    mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.neighbor_ids[:, 1:n_cells], mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.levels[1:n_cells],          mpi_root(), mpi_comm())
    @views MPI.Bcast!(mesh.tree.coordinates[:, 1:n_cells],  mpi_root(), mpi_comm())
  end

  # Partition mesh
  partition!(mesh)

  return mesh
end

