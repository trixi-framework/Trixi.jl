
# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::TreeMesh, output_directory, timestep=0)
  save_mesh_file(mesh, output_directory, timestep, mpi_parallel())
end

function save_mesh_file(mesh::TreeMesh, output_directory, timestep,
                        mpi_parallel::Val{false})
  # Create output directory (if it does not exist)
  mkpath(output_directory)

  # Determine file name based on existence of meaningful time step
  if timestep > 0
    filename = joinpath(output_directory, @sprintf("mesh_%06d.h5", timestep))
  else
    filename = joinpath(output_directory, "mesh.h5")
  end

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    n_cells = length(mesh.tree)
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["n_cells"] = n_cells
    attributes(file)["n_leaf_cells"] = count_leaf_cells(mesh.tree)
    attributes(file)["minimum_level"] = minimum_level(mesh.tree)
    attributes(file)["maximum_level"] = maximum_level(mesh.tree)
    attributes(file)["center_level_0"] = mesh.tree.center_level_0
    attributes(file)["length_level_0"] = mesh.tree.length_level_0
    attributes(file)["periodicity"] = collect(mesh.tree.periodicity)

    # Add tree data
    file["parent_ids"] = @view mesh.tree.parent_ids[1:n_cells]
    file["child_ids"] = @view mesh.tree.child_ids[:, 1:n_cells]
    file["neighbor_ids"] = @view mesh.tree.neighbor_ids[:, 1:n_cells]
    file["levels"] = @view mesh.tree.levels[1:n_cells]
    file["coordinates"] = @view mesh.tree.coordinates[:, 1:n_cells]
  end

  return filename
end

# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::TreeMesh, output_directory, timestep,
                        mpi_parallel::Val{true})
  # Create output directory (if it does not exist)
  mpi_isroot() && mkpath(output_directory)

  # Determine file name based on existence of meaningful time step
  if timestep >= 0
    filename = joinpath(output_directory, @sprintf("mesh_%06d.h5", timestep))
  else
    filename = joinpath(output_directory, "mesh.h5")
  end

  # Since the mesh is replicated on all ranks, only save from MPI root
  if !mpi_isroot()
    return filename
  end

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    n_cells = length(mesh.tree)
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["n_cells"] = n_cells
    attributes(file)["n_leaf_cells"] = count_leaf_cells(mesh.tree)
    attributes(file)["minimum_level"] = minimum_level(mesh.tree)
    attributes(file)["maximum_level"] = maximum_level(mesh.tree)
    attributes(file)["center_level_0"] = mesh.tree.center_level_0
    attributes(file)["length_level_0"] = mesh.tree.length_level_0
    attributes(file)["periodicity"] = collect(mesh.tree.periodicity)

    # Add tree data
    file["parent_ids"] = @view mesh.tree.parent_ids[1:n_cells]
    file["child_ids"] = @view mesh.tree.child_ids[:, 1:n_cells]
    file["neighbor_ids"] = @view mesh.tree.neighbor_ids[:, 1:n_cells]
    file["levels"] = @view mesh.tree.levels[1:n_cells]
    file["coordinates"] = @view mesh.tree.coordinates[:, 1:n_cells]
  end

  return filename
end


"""
    load_mesh(restart_file::AbstractString; n_cells_max)

Load the mesh from the `restart_file`.
"""
function load_mesh(restart_file::AbstractString; n_cells_max)
  load_mesh(restart_file, mpi_parallel(); n_cells_max=n_cells_max)
end

function load_mesh(restart_file::AbstractString, mpi_parallel::Val{false};
                   n_cells_max)
  ndims_ = h5open(restart_file, "r") do file
    read(attributes(file)["ndims"])
  end

  mesh = TreeMesh(SerialTree{ndims_}, n_cells_max)
  load_mesh!(mesh, restart_file)
end

function load_mesh!(mesh::SerialTreeMesh, restart_file::AbstractString)
  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_file, mpi_parallel(mesh))
  mesh.current_filename = filename
  mesh.unsaved_changes = false

  # Read mesh file
  h5open(filename, "r") do file
    # Set domain information
    mesh.tree.center_level_0 = read(attributes(file)["center_level_0"])
    mesh.tree.length_level_0 = read(attributes(file)["length_level_0"])
    mesh.tree.periodicity    = Tuple(read(attributes(file)["periodicity"]))

    # Set length
    n_cells = read(attributes(file)["n_cells"])
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


function load_mesh(restart_file::AbstractString, mpi_parallel::Val{true};
                   n_cells_max)
  if mpi_isroot()
    ndims_ = h5open(restart_file, "r") do file
      read(attributes(file)["ndims"])
    end
    MPI.Bcast!(Ref(ndims_), mpi_root(), mpi_comm())
  else
    ndims_ = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
  end

  mesh = TreeMesh(ParallelTree{ndims_}, n_cells_max)
  load_mesh!(mesh, restart_file)
end

function load_mesh!(mesh::ParallelTreeMesh, restart_file::AbstractString)
  # Determine mesh filename
  filename = get_restart_mesh_filename(restart_file, mpi_parallel(mesh))
  mesh.current_filename = filename
  mesh.unsaved_changes = false

  if mpi_isroot()
    h5open(filename, "r") do file
      # Set domain information
      mesh.tree.center_level_0 = read(attributes(file)["center_level_0"])
      mesh.tree.length_level_0 = read(attributes(file)["length_level_0"])
      mesh.tree.periodicity    = Tuple(read(attributes(file)["periodicity"]))
      MPI.Bcast!(collect(mesh.tree.center_level_0), mpi_root(), mpi_comm())
      MPI.Bcast!(collect(mesh.tree.length_level_0), mpi_root(), mpi_comm())
      MPI.Bcast!(collect(mesh.tree.periodicity),    mpi_root(), mpi_comm())

      # Set length
      n_cells = read(attributes(file)["n_cells"])
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
