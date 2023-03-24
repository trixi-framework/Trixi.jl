# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function save_restart_file(u, time, dt, timestep,
                           mesh::Union{SerialTreeMesh, StructuredMesh, UnstructuredMesh2D, SerialP4estMesh},
                           equations, dg::DG, cache,
                           restart_callback)
  @unpack output_directory = restart_callback

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("restart_%06d.h5", timestep))

  # Restart files always store conservative variables
  data = u

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["equations"] = get_name(equations)
    attributes(file)["polydeg"] = polydeg(dg)
    attributes(file)["n_vars"] = nvariables(equations)
    attributes(file)["n_elements"] = nelements(dg, cache)
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
    attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
    attributes(file)["timestep"] = timestep

    # Store each variable of the solution
    for v in eachvariable(equations)
      # Convert to 1D array
      file["variables_$v"] = vec(data[v, .., :])

      # Add variable name as attribute
      var = file["variables_$v"]
      attributes(var)["name"] = varnames(cons2cons, equations)[v]
    end
  end

  return filename
end


function load_restart_file(mesh::Union{SerialTreeMesh, StructuredMesh, UnstructuredMesh2D, SerialP4estMesh},
                           equations, dg::DG, cache, restart_file)

  # allocate memory
  u_ode = allocate_coefficients(mesh, equations, dg, cache)
  u = wrap_array_native(u_ode, mesh, equations, dg, cache)

  h5open(restart_file, "r") do file
    # Read attributes to perform some sanity checks
    if read(attributes(file)["ndims"]) != ndims(mesh)
      error("restart mismatch: ndims differs from value in restart file")
    end
    if read(attributes(file)["equations"]) != get_name(equations)
      error("restart mismatch: equations differ from value in restart file")
    end
    if read(attributes(file)["polydeg"]) != polydeg(dg)
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end
    if read(attributes(file)["n_elements"]) != nelements(dg, cache)
      error("restart mismatch: number of elements in solver differs from value in restart file")
    end

    # Read data
    for v in eachvariable(equations)
      # Check if variable name matches
      var = file["variables_$v"]
      if (name = read(attributes(var)["name"])) != varnames(cons2cons, equations)[v]
        error("mismatch: variables_$v should be '$(varnames(cons2cons, equations)[v])', but found '$name'")
      end

      # Read variable
      u[v, .., :] = read(file["variables_$v"])
    end
  end

  return u_ode
end


function save_restart_file(u, time, dt, timestep,
                           mesh::Union{ParallelTreeMesh, ParallelP4estMesh}, equations, dg::DG, cache,
                           restart_callback)
  @unpack output_directory = restart_callback

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("restart_%06d.h5", timestep))

  # Restart files always store conservative variables
  data = u

  # Calculate element and node counts by MPI rank
  element_size = nnodes(dg)^ndims(mesh)
  element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
  node_counts = element_counts * Cint(element_size)

  # non-root ranks only send data
  if !mpi_isroot()
    # Send nodal data to root
    for v in eachvariable(equations)
      MPI.Gatherv!(vec(data[v, .., :]), nothing, mpi_root(), mpi_comm())
    end

    return filename
  end

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["equations"] = get_name(equations)
    attributes(file)["polydeg"] = polydeg(dg)
    attributes(file)["n_vars"] = nvariables(equations)
    attributes(file)["n_elements"] = nelements(dg, cache)
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
    attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
    attributes(file)["timestep"] = timestep

    # Store each variable of the solution
    for v in eachvariable(equations)
      # Convert to 1D array
      recv = Vector{eltype(data)}(undef, sum(node_counts))
      MPI.Gatherv!(vec(data[v, .., :]), MPI.VBuffer(recv, node_counts), mpi_root(), mpi_comm())
      file["variables_$v"] = recv

      # Add variable name as attribute
      var = file["variables_$v"]
      attributes(var)["name"] = varnames(cons2cons, equations)[v]
    end
  end

  return filename
end


function load_restart_file(mesh::Union{ParallelTreeMesh, ParallelP4estMesh}, equations, dg::DG, cache, restart_file)

  # Calculate element and node counts by MPI rank
  element_size = nnodes(dg)^ndims(mesh)
  element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
  node_counts = element_counts * Cint(element_size)

  # allocate memory
  u_ode = allocate_coefficients(mesh, equations, dg, cache)
  u = wrap_array_native(u_ode, mesh, equations, dg, cache)

  # non-root ranks only receive data
  if !mpi_isroot()
    # Receive nodal data from root
    for v in eachvariable(equations)
      # put Scatterv in both blocks of the if condition to avoid type instability
      if isempty(u)
        data = eltype(u)[]
        MPI.Scatterv!(nothing, data, mpi_root(), mpi_comm())
      else
        data = @view u[v, .., :]
        MPI.Scatterv!(nothing, data, mpi_root(), mpi_comm())
      end
    end

    return u_ode
  end

  # read only on MPI root
  h5open(restart_file, "r") do file
    # Read attributes to perform some sanity checks
    if read(attributes(file)["ndims"]) != ndims(mesh)
      error("restart mismatch: ndims differs from value in restart file")
    end
    if read(attributes(file)["equations"]) != get_name(equations)
      error("restart mismatch: equations differ from value in restart file")
    end
    if read(attributes(file)["polydeg"]) != polydeg(dg)
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end
    if read(attributes(file)["n_elements"]) != nelements(dg, cache)
      error("restart mismatch: number of elements in solver differs from value in restart file")
    end

    # Read data
    for v in eachvariable(equations)
      # Check if variable name matches
      var = file["variables_$v"]
      if (name = read(attributes(var)["name"])) != varnames(cons2cons, equations)[v]
        error("mismatch: variables_$v should be '$(varnames(cons2cons, equations)[v])', but found '$name'")
      end

      # Read variable
      println("Reading variables_$v ($name)...")
      sendbuf = MPI.VBuffer(read(file["variables_$v"]), node_counts)
      MPI.Scatterv!(sendbuf, @view(u[v, .., :]), mpi_root(), mpi_comm())
    end
  end

  return u_ode
end


end # @muladd
