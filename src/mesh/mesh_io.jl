
# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::TreeMesh, output_directory, timestep=0)
  save_mesh_file(mesh, output_directory, timestep, mpi_parallel(mesh))
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
    attributes(file)["mesh_type"] = get_name(mesh)
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
    attributes(file)["mesh_type"] = get_name(mesh)
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


# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the CurvedMesh and its node coordinates are reconstructured from
# these attributes for plotting purposes
function save_mesh_file(mesh::CurvedMesh, output_directory)
  # Create output directory (if it does not exist)
  mkpath(output_directory)

  filename = joinpath(output_directory, "mesh.h5")

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["size"] = collect(size(mesh))
    attributes(file)["mapping"] = mesh.mapping_as_string
  end

  return filename
end


# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the corresponding `.mesh` file used to construct the mesh.
# Then, within Trixi2Vtk, the UnstructuredQuadMesh and its node coordinates are reconstructured
# from these attributes for plotting purposes
function save_mesh_file(mesh::UnstructuredQuadMesh, output_directory)
  # Create output directory (if it does not exist)
  mkpath(output_directory)

  filename = joinpath(output_directory, "mesh.h5")

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["size"] = length(mesh)
    attributes(file)["mesh_filename"] = mesh.filename
    attributes(file)["periodicity"] = collect(mesh.periodicity)
  end

  return filename
end


# Does not save the mesh itself to an HDF5 file. Instead saves important attributes
# of the mesh, like its size and the type of boundary mapping function.
# Then, within Trixi2Vtk, the P4estMesh and its node coordinates are reconstructured from
# these attributes for plotting purposes
function save_mesh_file(mesh::P4estMesh, output_directory, timestep=0)
  # Create output directory (if it does not exist)
  mkpath(output_directory)

  # Determine file name based on existence of meaningful time step
  if timestep > 0
    filename = joinpath(output_directory, @sprintf("mesh_%06d.h5", timestep))
    p4est_filename = @sprintf("p4est_data_%06d", timestep)
  else
    filename = joinpath(output_directory, "mesh.h5")
    p4est_filename = "p4est_data"
  end

  p4est_file = joinpath(output_directory, p4est_filename)

  # Save the complete connectivity/p4est data to disk.
  p4est_save(p4est_file, mesh.p4est, false)

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["p4est_file"] = p4est_filename

    file["tree_node_coordinates"] = mesh.tree_node_coordinates
    file["nodes"] = Vector(mesh.nodes) # the mesh uses `SVector`s for the nodes
                                       # to increase the runtime performance
                                       # but HDF5 can only handle plain arrays
    file["boundary_names"] = mesh.boundary_names .|> String
  end

  return filename
end


"""
    load_mesh(restart_file::AbstractString; n_cells_max)

Load the mesh from the `restart_file`.
"""
function load_mesh(restart_file::AbstractString; n_cells_max=0, RealT=Float64)
  # Determine mesh filename
  mesh_file = get_restart_mesh_filename(restart_file, Val(mpi_isparallel()))

  if mpi_isparallel()
    return load_mesh_parallel(mesh_file; n_cells_max=n_cells_max, RealT=RealT)
  end

  load_mesh_serial(mesh_file; n_cells_max=n_cells_max, RealT=RealT)
end

function load_mesh_serial(mesh_file::AbstractString; n_cells_max, RealT)
  ndims, mesh_type = h5open(mesh_file, "r") do file
    return read(attributes(file)["ndims"]),
           read(attributes(file)["mesh_type"])
  end

  if mesh_type == "TreeMesh"
    n_cells = h5open(mesh_file, "r") do file
      return read(attributes(file)["n_cells"])
    end
    mesh = TreeMesh(SerialTree{ndims}, max(n_cells, n_cells_max))
    load_mesh!(mesh, mesh_file)
  elseif mesh_type == "CurvedMesh"
    size_, mapping_as_string = h5open(mesh_file, "r") do file
      return read(attributes(file)["size"]),
             read(attributes(file)["mapping"])
    end

    size = Tuple(size_)

    # TODO: `@eval` is evil
    # A temporary workaround to evaluate the code that defines the domain mapping in a local scope.
    # This prevents errors when multiple restart elixirs are executed in one session, where one
    # defines `mapping` as a variable, while the other defines it as a function.
    #
    # This should be replaced with something more robust and secure,
    # see https://github.com/trixi-framework/Trixi.jl/issues/541).
    expr = Meta.parse(mapping_as_string)
    if expr.head == :toplevel
      expr.head = :block
    end

    if ndims == 1
      mapping = @eval function(xi)
        $expr
        mapping(xi)
      end
    elseif ndims == 2
      mapping = @eval function(xi, eta)
        $expr
        mapping(xi, eta)
      end
    else # ndims == 3
      mapping = @eval function(xi, eta, zeta)
        $expr
        mapping(xi, eta, zeta)
      end
    end

    mesh = CurvedMesh(size, mapping; RealT=RealT, unsaved_changes=false, mapping_as_string=mapping_as_string)
  elseif mesh_type == "UnstructuredQuadMesh"
    mesh_filename, periodicity_ = h5open(mesh_file, "r") do file
      return read(attributes(file)["mesh_filename"]),
             read(attributes(file)["periodicity"])
    end
    mesh = UnstructuredQuadMesh(mesh_filename; RealT=RealT, periodicity=periodicity_, unsaved_changes=false)
  elseif mesh_type == "P4estMesh"
    p4est_filename, tree_node_coordinates,
        nodes, boundary_names_ = h5open(mesh_file, "r") do file
      return read(attributes(file)["p4est_file"]),
             read(file["tree_node_coordinates"]),
             read(file["nodes"]),
             read(file["boundary_names"])
    end

    boundary_names = boundary_names_ .|> Symbol

    p4est_file = joinpath(dirname(mesh_file), p4est_filename)
    # Prevent Julia crashes when p4est can't find the file
    @assert isfile(p4est_file)

    conn_vec = Vector{Ptr{p4est_connectivity_t}}(undef, 1)
    p4est = p4est_load(p4est_file, C_NULL, 0, false, C_NULL, pointer(conn_vec))

    mesh = P4estMesh{ndims}(p4est, tree_node_coordinates,
                            nodes, boundary_names, "", false)
  else
    error("Unknown mesh type!")
  end

  return mesh
end

function load_mesh!(mesh::SerialTreeMesh, mesh_file::AbstractString)
  mesh.current_filename = mesh_file
  mesh.unsaved_changes = false

  # Read mesh file
  h5open(mesh_file, "r") do file
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


function load_mesh_parallel(mesh_file::AbstractString; n_cells_max, RealT)
  if mpi_isroot()
    ndims_, n_cells = h5open(mesh_file, "r") do file
      read(attributes(file)["ndims"]),
      read(attributes(file)["n_cells"])
    end
    MPI.Bcast!(Ref(ndims_), mpi_root(), mpi_comm())
    MPI.Bcast!(Ref(n_cells), mpi_root(), mpi_comm())
  else
    ndims_ = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
    n_cells = MPI.Bcast!(Ref(0), mpi_root(), mpi_comm())[]
  end

  mesh = TreeMesh(ParallelTree{ndims_}, max(n_cells, n_cells_max))
  load_mesh!(mesh, mesh_file)

  return mesh
end

function load_mesh!(mesh::ParallelTreeMesh, mesh_file::AbstractString)
  mesh.current_filename = mesh_file
  mesh.unsaved_changes = false

  if mpi_isroot()
    h5open(mesh_file, "r") do file
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
