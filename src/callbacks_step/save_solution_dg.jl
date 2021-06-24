# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function save_solution_file(u, time, dt, timestep,
                            mesh::Union{SerialTreeMesh, CurvedMesh, UnstructuredQuadMesh, P4estMesh},
                            equations, dg::DG, cache,
                            solution_callback, element_variables=Dict{Symbol,Any}();
                            system="")
  @unpack output_directory, solution_variables = solution_callback

  # Filename without extension based on current time step
  if isempty(system)
    filename = joinpath(output_directory, @sprintf("solution_%06d.h5", timestep))
  else
    filename = joinpath(output_directory, @sprintf("solution_%s_%06d.h5", system, timestep))
  end

  # Convert to different set of variables if requested
  if solution_variables === cons2cons
    data = u
    n_vars = nvariables(equations)
  else
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    data = Array(reinterpret(eltype(u),
           solution_variables.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
                      Ref(equations))))

    # Find out variable count by looking at output from `solution_variables` function
    n_vars = size(data, 1)
  end

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["equations"] = get_name(equations)
    attributes(file)["polydeg"] = polydeg(dg)
    attributes(file)["n_vars"] = n_vars
    attributes(file)["n_elements"] = nelements(dg, cache)
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
    attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
    attributes(file)["timestep"] = timestep

    # Store each variable of the solution data
    for v in 1:n_vars
      # Convert to 1D array
      file["variables_$v"] = vec(data[v, .., :])

      # Add variable name as attribute
      var = file["variables_$v"]
      attributes(var)["name"] = varnames(solution_variables, equations)[v]
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

  # Convert to different set of variables if requested
  if solution_variables === cons2cons
    data = u
    n_vars = nvariables(equations)
  else
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    data = Array(reinterpret(eltype(u),
           solution_variables.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
                      Ref(equations))))

    # Find out variable count by looking at output from `solution_variables` function
    n_vars = size(data, 1)
  end

  # Calculate element and node counts by MPI rank
  element_size = nnodes(dg)^ndims(mesh)
  element_counts = convert(Vector{Cint}, collect(cache.mpi_cache.n_elements_by_rank))
  node_counts = element_counts * Cint(element_size)

  # non-root ranks only send data
  if !mpi_isroot()
    # Send nodal data to root
    for v in 1:n_vars
      MPI.Gatherv!(vec(data[v, .., :]), nothing, mpi_root(), mpi_comm())
    end

    # Send element data to root
    for (key, element_variable) in element_variables
      MPI.Gatherv!(element_variable, nothing, mpi_root(), mpi_comm())
    end

    return filename
  end

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attributes(file)["ndims"] = ndims(mesh)
    attributes(file)["equations"] = get_name(equations)
    attributes(file)["polydeg"] = polydeg(dg)
    attributes(file)["n_vars"] = n_vars
    attributes(file)["n_elements"] = nelementsglobal(dg, cache)
    attributes(file)["mesh_type"] = get_name(mesh)
    attributes(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attributes(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
    attributes(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
    attributes(file)["timestep"] = timestep

    # Store each variable of the solution data
    for v in 1:n_vars
      # Convert to 1D array
      recv = Vector{eltype(data)}(undef, sum(node_counts))
      MPI.Gatherv!(vec(data[v, .., :]), MPI.VBuffer(recv, node_counts), mpi_root(), mpi_comm())
      file["variables_$v"] = recv

      # Add variable name as attribute
      var = file["variables_$v"]
      attributes(var)["name"] = varnames(solution_variables, equations)[v]
    end

    # Store element variables
    for (v, (key, element_variable)) in enumerate(element_variables)
      # Add to file
      recv = Vector{eltype(data)}(undef, sum(element_counts))
      MPI.Gatherv!(element_variable, MPI.VBuffer(recv, element_counts), mpi_root(), mpi_comm())
      file["element_variables_$v"] = recv

      # Add variable name as attribute
      var = file["element_variables_$v"]
      attributes(var)["name"] = string(key)
    end
  end

  return filename
end


end # @muladd
