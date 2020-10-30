
function save_restart_file(u, time, dt, timestep,
                           mesh, equations, dg::DG, cache,
                           restart_callback)
  @unpack output_directory = restart_callback

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("restart_%06d.h5", timestep))

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attrs(file)["ndims"] = ndims(mesh)
    attrs(file)["equations"] = get_name(equations)
    attrs(file)["polydeg"] = polydeg(dg)
    attrs(file)["n_vars"] = nvariables(equations)
    attrs(file)["n_elements"] = nelements(dg, cache)
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = convert(Float64, time) # Ensure that `time` is written as a double precision scalar
    attrs(file)["dt"] = convert(Float64, dt) # Ensure that `dt` is written as a double precision scalar
    attrs(file)["timestep"] = timestep

    # Restart files always store conservative variables
    data = u
    varnames = varnames_cons(equations)

    # Store each variable of the solution
    for v in eachvariable(equations)
      # Convert to 1D array
      file["variables_$v"] = vec(data[v, .., :])

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = varnames[v]
    end
  end

  return filename
end


function load_restart_file(mesh, equations, dg::DG, cache, restart_file)
  h5open(restart_file, "r") do file
    # Read attributes to perform some sanity checks
    if read(attrs(file)["ndims"]) != ndims(mesh)
      error("restart mismatch: ndims differs from value in restart file")
    end
    if read(attrs(file)["equations"]) != get_name(equations)
      error("restart mismatch: equations differ from value in restart file")
    end
    if read(attrs(file)["polydeg"]) != polydeg(dg)
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end
    if read(attrs(file)["n_elements"]) != nelements(dg, cache)
      error("restart mismatch: number of elements in solver differs from value in restart file")
    end

    # Read data
    u_ode = allocate_coefficients(mesh, equations, dg, cache)
    u = wrap_array(u_ode, mesh, equations, dg, cache)
    varnames = varnames_cons(equations)
    for v in eachvariable(equations)
      # Check if variable name matches
      var = file["variables_$v"]
      if (name = read(attrs(var)["name"])) != varnames[v]
        error("mismatch: variables_$v should be '$(varnames[v])', but found '$name'")
      end

      # Read variable
      println("Reading variables_$v ($name)...")
      u[v, .., :] = read(file["variables_$v"])
    end

    return u_ode
  end
end
