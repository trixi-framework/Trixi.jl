
# Load restart file and store solution in solver
function load_restart_file!(dg::AbstractDg, restart_filename, mpi_parallel::Val{true})
  # Create variables to be returned later
  time = NaN
  step = -1

  # Calculate node counts by MPI rank
  element_size = nnodes(dg)^ndims(dg)
  node_counts = convert(Vector{Cint}, collect(dg.n_elements_by_rank)) * Cint(element_size)

  if mpi_isroot()
    # Open file
    h5open(restart_filename, "r") do file
      # Read attributes to perform some sanity checks
      if read(attrs(file)["ndims"]) != ndims(dg)
        error("restart mismatch: ndims in solver differs from value in restart file")
      end
      if read(attrs(file)["equations"]) != get_name(equations(dg))
        error("restart mismatch: equations in solver differs from value in restart file")
      end
      if read(attrs(file)["polydeg"]) != polydeg(dg)
        error("restart mismatch: polynomial degree in solver differs from value in restart file")
      end
      if read(attrs(file)["n_elements"]) != dg.n_elements_global
        error("restart mismatch: polynomial degree in solver differs from value in restart file")
      end

      # Read time and time step
      time = read(attrs(file)["time"])
      step = read(attrs(file)["timestep"])
      MPI.Bcast!(Ref(time), mpi_root(), mpi_comm())
      MPI.Bcast!(Ref(step), mpi_root(), mpi_comm())

      # Read data
      varnames = varnames_cons(equations(dg))
      for v in 1:nvariables(dg)
        # Check if variable name matches
        var = file["variables_$v"]
        if (name = read(attrs(var)["name"])) != varnames[v]
          error("mismatch: variables_$v should be '$(varnames[v])', but found '$name'")
        end

        # Read variable
        dg.elements.u[v, .., :] = MPI.Scatterv(read(file["variables_$v"]), node_counts, mpi_root(), mpi_comm())
      end
    end
  else # on non-root ranks, receive data from root
    time = MPI.Bcast!(Ref(time), mpi_root(), mpi_comm())[]
    step = MPI.Bcast!(Ref(step), mpi_root(), mpi_comm())[]
    for v in 1:nvariables(dg)
      # Read variable
      dg.elements.u[v, .., :] = MPI.Scatterv(eltype(dg.elements.u)[], node_counts, mpi_root(), mpi_comm())
    end
  end

  return time, step
end

function save_restart_file(dg::AbstractDg, mesh::TreeMesh, time, dt, timestep,
                           mpi_parallel::Val{true})
  # Calculate node counts by MPI rank
  element_size = nnodes(dg)^ndims(dg)
  node_counts = convert(Vector{Cint}, collect(dg.n_elements_by_rank)) * Cint(element_size)

  # Restart files always store conservative variables
  data = dg.elements.u
  varnames = varnames_cons(equations(dg))

  # Only write from MPI root (poor man's version of parallel I/O)
  if mpi_isroot()
    # Create output directory (if it does not exist)
    output_directory = parameter("output_directory", "out")
    if mpi_isroot()
      mkpath(output_directory)
    end

    # Filename without extension based on current time step
    filename = joinpath(output_directory, @sprintf("restart_%06d", timestep))

    # Convert time and time step size to floats
    time = convert(Float64, time)
    dt = convert(Float64, dt)

    # Open file (clobber existing content)
    h5open(filename * ".h5", "w") do file
      # Add context information as attributes
      attrs(file)["ndims"] = ndims(dg)
      attrs(file)["equations"] = get_name(equations(dg))
      attrs(file)["polydeg"] = polydeg(dg)
      attrs(file)["n_vars"] = nvariables(dg)
      attrs(file)["n_elements"] = dg.n_elements_global
      attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
      attrs(file)["time"] = time
      attrs(file)["dt"] = dt
      attrs(file)["timestep"] = timestep

      # Store each variable of the solution
      for v in 1:nvariables(dg)
        # Write to file
        file["variables_$v"] = MPI.Gatherv(vec(data[v, .., :]), node_counts, mpi_root(), mpi_comm())

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end
    end
  else # non-root ranks only send data
    # Send nodal data to root
    for v in 1:nvariables(dg)
      MPI.Gatherv(vec(data[v, .., :]), node_counts, mpi_root(), mpi_comm())
    end
  end
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(dg::AbstractDg, mesh::TreeMesh, time, dt, timestep, system,
                            mpi_parallel::Val{true})

  # Calculate element and node counts by MPI rank
  element_size = nnodes(dg)^ndims(dg)
  element_counts = convert(Vector{Cint}, collect(dg.n_elements_by_rank))
  node_counts = element_counts * Cint(element_size)

  # Convert to primitive variables if requested
  solution_variables = parameter("solution_variables", "primitive",
                                valid=["conservative", "primitive"])
  if solution_variables == "conservative"
    data = dg.elements.u
    varnames = varnames_cons(equations(dg))
  else
    # Reinterpret the solution array as an array of conservative variables,
    # compute the primitive variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    data = Array(reinterpret(eltype(dg.elements.u),
          cons2prim.(reinterpret(SVector{nvariables(dg),eltype(dg.elements.u)}, dg.elements.u),
                      Ref(equations(dg)))))
    varnames = varnames_prim(equations(dg))
  end

  # Only write from MPI root (poor man's version of parallel I/O)
  if mpi_isroot()
    # Create output directory (if it does not exist)
    output_directory = parameter("output_directory", "out")
    mkpath(output_directory)

    # Filename without extension based on current time step
    if isempty(system)
      filename = joinpath(output_directory, @sprintf("solution_%06d", timestep))
    else
      filename = joinpath(output_directory, @sprintf("solution_%s_%06d", system, timestep))
    end

    # Convert time and time step size to floats
    time = convert(Float64, time)
    dt = convert(Float64, dt)

    # Open file (clobber existing content)
    h5open(filename * ".h5", "w") do file
      # Add context information as attributes
      attrs(file)["ndims"] = ndims(dg)
      attrs(file)["equations"] = get_name(equations(dg))
      attrs(file)["polydeg"] = polydeg(dg)
      attrs(file)["n_vars"] = nvariables(dg)
      attrs(file)["n_elements"] = dg.n_elements_global
      attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
      attrs(file)["time"] = time
      attrs(file)["dt"] = dt
      attrs(file)["timestep"] = timestep

      # Store each variable of the solution
      for v in 1:nvariables(dg)
        # Write to file
        file["variables_$v"] = MPI.Gatherv(vec(data[v, .., :]), node_counts, mpi_root(), mpi_comm())

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end

      # Store element variables
      for (v, (key, element_variables)) in enumerate(dg.element_variables)
        # Add to file
        file["element_variables_$v"] = MPI.Gatherv(element_variables, element_counts, mpi_root(), mpi_comm())

        # Add variable name as attribute
        var = file["element_variables_$v"]
        attrs(var)["name"] = string(key)
      end
    end
  else # non-root ranks only send data
    # Send nodal data to root
    for v in 1:nvariables(dg)
      MPI.Gatherv(vec(data[v, .., :]), node_counts, mpi_root(), mpi_comm())
    end

    # Send element data to root
    for (v, (key, element_variables)) in enumerate(dg.element_variables)
      MPI.Gatherv(element_variables, element_counts, mpi_root(), mpi_comm())
    end
  end
end
