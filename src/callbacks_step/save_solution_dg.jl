
function save_solution_file(u, time, dt, timestep,
                            mesh::SerialTreeMesh, equations, dg::DG, cache,
                            solution_callback, element_variables=Dict{Symbol,Any}();
                            system="")
  @unpack output_directory, solution_variables = solution_callback

  # Filename without extension based on current time step
  if isempty(system)
    filename = joinpath(output_directory, @sprintf("solution_%06d.h5", timestep))
  else
    filename = joinpath(output_directory, @sprintf("solution_%s_%06d.h5", system, timestep))
  end

  # Convert to primitive variables if requested
  if solution_variables === :conservative
    data = u
    varnames = varnames_cons(equations)
  elseif solution_variables === :primitive
    # Reinterpret the solution array as an array of conservative variables,
    # compute the primitive variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    data = Array(reinterpret(eltype(u),
           cons2prim.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
                      Ref(equations))))
    varnames = varnames_prim(equations)
  else
    error("Unknown solution_variables $solution_variables")
  end

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["equations"] = get_name(equations)
    attributes(file)["polydeg"] = polydeg(dg)
    attributes(file)["n_vars"] = nvariables(equations)
    attributes(file)["n_elements"] = nelements(dg, cache)
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
      attributes(var)["name"] = varnames[v]
    end

    # Store element variables
    for (v, (key, element_variable)) in enumerate(element_variables)
      # Add to file
      file["element_variables_$v"] = element_variable

      # Add variable name as attribute
      var = file["element_variables_$v"]
      attributes(var)["name"] = string(key)
    end
  end

  return filename
end


function save_solution_file(u, time, dt, timestep,
                            mesh::ParallelTreeMesh, equations, dg::DG, cache,
                            solution_callback, element_variables=Dict{Symbol,Any}();
                            system="")
  @unpack output_directory, solution_variables = solution_callback

  # Filename without extension based on current time step
  if isempty(system)
    filename = joinpath(output_directory, @sprintf("solution_%06d.h5", timestep))
  else
    filename = joinpath(output_directory, @sprintf("solution_%s_%06d.h5", system, timestep))
  end

  # Convert to primitive variables if requested
  if solution_variables === :conservative
    data = u
    varnames = varnames_cons(equations)
  elseif solution_variables === :primitive
    # Reinterpret the solution array as an array of conservative variables,
    # compute the primitive variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    data = Array(reinterpret(eltype(u),
           cons2prim.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
                      Ref(equations))))
    varnames = varnames_prim(equations)
  else
    error("Unknown solution_variables $solution_variables")
  end

  # Calculate element and node counts by MPI rank
  element_size = nnodes(dg)^ndims(mesh)
  element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
  node_counts = element_counts * Cint(element_size)

  # non-root ranks only send data
  if !mpi_isroot()
    # Send nodal data to root
    for v in eachvariable(equations)
      MPI.Gatherv(vec(data[v, .., :]), node_counts, mpi_root(), mpi_comm())
    end

    # Send element data to root
    for (key, element_variable) in element_variables
      MPI.Gatherv(element_variable, element_counts, mpi_root(), mpi_comm())
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
    attributes(file)["n_elements"] = cache.mpi_cache.n_elements_global
    attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
    attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
    attributes(file)["timestep"] = timestep

    # Store each variable of the solution
    for v in eachvariable(equations)
      # Convert to 1D array
      file["variables_$v"] = MPI.Gatherv(vec(data[v, .., :]), node_counts, mpi_root(), mpi_comm())

      # Add variable name as attribute
      var = file["variables_$v"]
      attributes(var)["name"] = varnames[v]
    end

    # Store element variables
    for (v, (key, element_variable)) in enumerate(element_variables)
      # Add to file
      file["element_variables_$v"] = MPI.Gatherv(element_variable, element_counts, mpi_root(), mpi_comm())

      # Add variable name as attribute
      var = file["element_variables_$v"]
      attributes(var)["name"] = string(key)
    end
  end

  return filename
end
