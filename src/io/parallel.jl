function save_restart_file(dg::AbstractDg, mesh::TreeMesh, time, dt, timestep,
                           mpi_parallel::Val{true})
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  if is_mpi_root()
    mkpath(output_directory)
  end

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("restart_%06d", timestep))

  # Convert time and time step size to floats
  time = convert(Float64, time)
  dt = convert(Float64, dt)

  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    equation = equations(dg)

    # Add context information as attributes
    attrs(file)["ndims"] = ndims(dg)
    attrs(file)["equations"] = get_name(equation)
    attrs(file)["polydeg"] = polydeg(dg)
    attrs(file)["n_vars"] = nvariables(dg)
    attrs(file)["n_elements"] = dg.n_elements_global
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = time
    attrs(file)["dt"] = dt
    attrs(file)["timestep"] = timestep

    # Restart files always store conservative variables
    data = dg.elements.u
    varnames = varnames_cons(equation)

    # Only write from MPI root (poor man's version of parallel I/O)
    element_size = nnodes(dg)^ndims(dg)
    counts = convert(Vector{Cint}, collect(dg.n_elements_by_domain)) * Cint(element_size)

    # Store each variable of the solution
    for v in 1:nvariables(dg)
      # Collect data on root domain
      buffer = MPI.Gatherv(vec(data[v, .., :]), counts, mpi_root(), mpi_comm())

      # Write only from root domain
      if is_mpi_root()
        # Write to file
        file["variables_$v"] = buffer

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end
    end
  end
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(dg::AbstractDg, mesh::TreeMesh, time, dt, timestep, system,
                            mpi_parallel::Val{true})
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  if is_mpi_root()
    mkpath(output_directory)
  end

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
    equation = equations(dg)

    # Add context information as attributes
    attrs(file)["ndims"] = ndims(dg)
    attrs(file)["equations"] = get_name(equation)
    attrs(file)["polydeg"] = polydeg(dg)
    attrs(file)["n_vars"] = nvariables(dg)
    attrs(file)["n_elements"] = dg.n_elements_global
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = time
    attrs(file)["dt"] = dt
    attrs(file)["timestep"] = timestep

    # Convert to primitive variables if requested
    solution_variables = parameter("solution_variables", "primitive",
                                  valid=["conservative", "primitive"])
    if solution_variables == "conservative"
      data = dg.elements.u
      varnames = varnames_cons(equation)
    else
      # Reinterpret the solution array as an array of conservative variables,
      # compute the primitive variables via broadcasting, and reinterpret the
      # result as a plain array of floating point numbers
      data = Array(reinterpret(eltype(dg.elements.u),
            cons2prim.(reinterpret(SVector{nvariables(dg),eltype(dg.elements.u)}, dg.elements.u),
                        Ref(equations(dg)))))
      varnames = varnames_prim(equation)
    end

    # Only write from MPI root (poor man's version of parallel I/O)
    element_size = nnodes(dg)^ndims(dg)
    counts_elements = convert(Vector{Cint}, collect(dg.n_elements_by_domain))
    counts_nodes = counts_elements * Cint(element_size)

    # Store each variable of the solution
    for v in 1:nvariables(dg)
      # Collect data on root domain
      buffer = MPI.Gatherv(vec(data[v, .., :]), counts_nodes, mpi_root(), mpi_comm())

      # Write only from root domain
      if is_mpi_root()
        # Convert to 1D array
        file["variables_$v"] = buffer

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end
    end

    # Store element variables
    for (v, (key, element_variables)) in enumerate(dg.element_variables)
      # Collect data on root domain
      buffer = MPI.Gatherv(element_variables, counts_elements, mpi_root(), mpi_comm())

      # Write only from root domain
      if is_mpi_root()
        # Add to file
        file["element_variables_$v"] = buffer

        # Add variable name as attribute
        var = file["element_variables_$v"]
        attrs(var)["name"] = string(key)
      end
    end
  end
end

