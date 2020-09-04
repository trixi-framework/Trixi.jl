
# Load restart file and store solution in solver
function load_restart_file!(dg::AbstractDg, restart_filename)
  # Create variables to be returned later
  time = NaN
  step = -1

  # Open file
  h5open(restart_filename, "r") do file
    equation = equations(dg)

    # Read attributes to perform some sanity checks
    if read(attrs(file)["ndims"]) != ndims(dg)
      error("restart mismatch: ndims in solver differs from value in restart file")
    end
    if read(attrs(file)["equations"]) != get_name(equation)
      error("restart mismatch: equations in solver differs from value in restart file")
    end
    if read(attrs(file)["polydeg"]) != polydeg(dg)
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end
    if read(attrs(file)["n_elements"]) != dg.n_elements
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end

    # Read time and time step
    time = read(attrs(file)["time"])
    step = read(attrs(file)["timestep"])

    # Read data
    varnames = varnames_cons(equation)
    for v in 1:nvariables(dg)
      # Check if variable name matches
      var = file["variables_$v"]
      if (name = read(attrs(var)["name"])) != varnames[v]
        error("mismatch: variables_$v should be '$(varnames[v])', but found '$name'")
      end

      # Read variable
      println("Reading variables_$v ($name)...")
      if ndims(dg) == 2
        dg.elements.u[v, :, :, :] = read(file["variables_$v"])
      elseif ndims(dg) == 3
        dg.elements.u[v, :, :, :, :] = read(file["variables_$v"])
      else
        error("Unsupported number of spatial dimensions: ", ndims(dg))
      end
    end
  end

  return time, step
end


# Save current DG solution with some context information as a HDF5 file for
# restarting.
function save_restart_file(dg::AbstractDg, mesh::TreeMesh, time, dt, timestep)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

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
    attrs(file)["n_elements"] = dg.n_elements
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = time
    attrs(file)["dt"] = dt
    attrs(file)["timestep"] = timestep

    # Restart files always store conservative variables
    data = dg.elements.u
    varnames = varnames_cons(equation)

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
  end
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(dg::AbstractDg, mesh::TreeMesh, time, dt, timestep, system="")
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
      data = cons2prim(dg.elements.u, equation)
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


# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::TreeMesh, timestep=-1)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Determine file name based on existence of meaningful time step
  if timestep >= 0
    filename = joinpath(output_directory, @sprintf("mesh_%06d", timestep))
  else
    filename = joinpath(output_directory, "mesh")
  end

  # Create output directory (if it does not exist)
  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    # Add context information as attributes
    n_cells = length(mesh.tree)
    attrs(file)["ndims"] = ndims(mesh)
    attrs(file)["n_cells"] = n_cells
    attrs(file)["n_leaf_cells"] = count_leaf_cells(mesh.tree)
    attrs(file)["minimum_level"] = minimum_level(mesh.tree)
    attrs(file)["maximum_level"] = maximum_level(mesh.tree)
    attrs(file)["center_level_0"] = mesh.tree.center_level_0
    attrs(file)["length_level_0"] = mesh.tree.length_level_0
    attrs(file)["periodicity"] = collect(mesh.tree.periodicity)

    # Add tree data
    file["parent_ids"] = @view mesh.tree.parent_ids[1:n_cells]
    file["child_ids"] = @view mesh.tree.child_ids[:, 1:n_cells]
    file["neighbor_ids"] = @view mesh.tree.neighbor_ids[:, 1:n_cells]
    file["levels"] = @view mesh.tree.levels[1:n_cells]
    file["coordinates"] = @view mesh.tree.coordinates[:, 1:n_cells]
  end

  return filename * ".h5"
end
