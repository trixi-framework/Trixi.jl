
function save_solution_file(u, time, dt, timestep,
                            mesh, equations, dg::DG, cache,
                            solution_callback;
                            system="")
  @unpack output_directory, solution_variables = solution_callback

  # Filename without extension based on current time step
  if isempty(system)
    filename = joinpath(output_directory, @sprintf("solution_%06d.h5", timestep))
  else
    filename = joinpath(output_directory, @sprintf("solution_%s_%06d.h5", system, timestep))
  end

  # Convert time and time step size to floats
  time = convert(Float64, time)
  dt   = convert(Float64, dt)

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    # Add context information as attributes
    attrs(file)["ndims"] = ndims(mesh)
    attrs(file)["equations"] = get_name(equations)
    attrs(file)["polydeg"] = polydeg(dg)
    attrs(file)["n_vars"] = nvariables(equations)
    attrs(file)["n_elements"] = nelements(dg, cache)
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = time
    attrs(file)["dt"] = dt
    attrs(file)["timestep"] = timestep

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

    # Store each variable of the solution
    for v in eachvariable(equations)
      # Convert to 1D array
      file["variables_$v"] = vec(data[v, .., :])

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = varnames[v]
    end

    # TODO: Taal implement, save element variables
    # Store element variables
    # for (v, (key, element_variables)) in enumerate(cache.element_variables)
    #   # Add to file
    #   file["element_variables_$v"] = element_variables

    #   # Add variable name as attribute
    #   var = file["element_variables_$v"]
    #   attrs(var)["name"] = string(key)
    # end
  end

  return filename
end
