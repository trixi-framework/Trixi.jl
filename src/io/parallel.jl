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

    # Store data in buffer
    if is_mpi_root()
      first_buffer_index = (dg.first_element_global_id - 1) * element_size + 1
      local_data_size = element_size * dg.n_elements
      last_buffer_index = first_buffer_index + local_data_size - 1

      # Create buffer for global element data
      buffer = Vector{eltype(data)}(undef, element_size * dg.n_elements_global)

      # Store each variable of the solution
      for v in 1:nvariables(dg)
        # Convert to 1D array and store in global buffer
        if ndims(dg) == 2
          buffer[first_buffer_index:last_buffer_index] = vec(data[v, :, :, :])
        elseif ndims(dg) == 3
          buffer[first_buffer_index:last_buffer_index] = vec(data[v, :, :, :, :])
        else
          error("Unsupported number of spatial dimensions: ", ndims(dg))
        end

        # Collect data on root domain
        # Note: `collect(...)` is required since we store domain info in OffsetArrays
        MPI.Gatherv!(nothing, buffer, counts, mpi_root(), mpi_comm())

        # Write to file
        file["variables_$v"] = buffer

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end
    else # On non-root domains
      # Create buffer for local element data
      buffer = Vector{eltype(data)}(undef, element_size * dg.n_elements)

      # Store each variable of the solution
      for v in 1:nvariables(dg)
        # Convert to 1D array and store in global buffer
        if ndims(dg) == 2
          buffer[:] = vec(data[v, :, :, :])
        elseif ndims(dg) == 3
          buffer[:] = vec(data[v, :, :, :, :])
        else
          error("Unsupported number of spatial dimensions: ", ndims(dg))
        end

        # Collect data on root domain
        # Note: `collect(...)` is required since we store domain info in OffsetArrays
        MPI.Gatherv!(buffer, nothing, counts, mpi_root(), mpi_comm())
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
    attrs(file)["n_elements"] = dg.n_elements
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
    counts = convert(Vector{Cint}, collect(dg.n_elements_by_domain)) * Cint(element_size)

    # Store data in buffer
    if is_mpi_root()
      first_buffer_index = (dg.first_element_global_id - 1) * element_size + 1
      local_data_size = element_size * dg.n_elements
      last_buffer_index = first_buffer_index + local_data_size - 1

      # Create buffer for global element data
      buffer = Vector{eltype(data)}(undef, element_size * dg.n_elements_global)

      # Store each variable of the solution
      for v in 1:nvariables(dg)
        # Convert to 1D array
        if ndims(dg) == 2
          file["variables_$v"] = vec(data[v, :, :, :])
        elseif ndims(dg) == 3
          file["variables_$v"] = vec(data[v, :, :, :, :])
        else
          error("Unsupported number of spatial dimensions: ", ndims(dg))
        end

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end

      # Store element variables
      for (v, (key, element_variables)) in enumerate(dg.element_variables)
        # Add to file
        file["element_variables_$v"] = element_variables

        # Add variable name as attribute
        var = file["element_variables_$v"]
        attrs(var)["name"] = string(key)
      end
    else # On non-root domains
      # Add coordinates as 1D arrays
      if ndims(dg) == 2
        file["x"] = vec(dg.elements.node_coordinates[1, :, :, :])
        file["y"] = vec(dg.elements.node_coordinates[2, :, :, :])
      elseif ndims(dg) == 3
        file["x"] = vec(dg.elements.node_coordinates[1, :, :, :, :])
        file["y"] = vec(dg.elements.node_coordinates[2, :, :, :, :])
        file["z"] = vec(dg.elements.node_coordinates[3, :, :, :, :])
      else
        error("Unsupported number of spatial dimensions: ", ndims(dg))
      end

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

      # Store each variable of the solution
      for v in 1:nvariables(dg)
        # Convert to 1D array
        if ndims(dg) == 2
          file["variables_$v"] = vec(data[v, :, :, :])
        elseif ndims(dg) == 3
          file["variables_$v"] = vec(data[v, :, :, :, :])
        else
          error("Unsupported number of spatial dimensions: ", ndims(dg))
        end

        # Add variable name as attribute
        var = file["variables_$v"]
        attrs(var)["name"] = varnames[v]
      end

      # Store element variables
      for (v, (key, element_variables)) in enumerate(dg.element_variables)
        # Add to file
        file["element_variables_$v"] = element_variables

        # Add variable name as attribute
        var = file["element_variables_$v"]
        attrs(var)["name"] = string(key)
      end
    end
  end
end

