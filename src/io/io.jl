module Io

using ..Trixi
using ..Solvers: AbstractSolver, polydeg, equations, Dg
using ..Solvers.DgSolver: polydeg
using ..Equations: nvariables, cons2prim
using ..Auxiliary: parameter
using ..Mesh: TreeMesh
using ..Mesh.Trees: Tree, count_leaf_cells, minimum_level, maximum_level,
                    n_children_per_cell, n_directions

using HDF5: h5open, attrs
using Printf: @sprintf

export load_restart_file!
export save_restart_file
export save_solution_file
export save_mesh_file


# Load restart file and store solution in solver
function load_restart_file!(dg::Dg, restart_filename::String)
  # Create variables to be returned later
  time = NaN
  step = -1

  # Open file
  h5open(restart_filename, "r") do file
    equation = equations(dg)
    N = polydeg(dg)

    # Read attributes to perform some sanity checks
    if read(attrs(file)["ndim"]) != ndim
      error("restart mismatch: ndim in solver differs from value in restart file")
    end
    if read(attrs(file)["equations"]) != equation.name
      error("restart mismatch: equations in solver differs from value in restart file")
    end
    if read(attrs(file)["N"]) != N
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end
    if read(attrs(file)["n_elements"]) != dg.n_elements
      error("restart mismatch: polynomial degree in solver differs from value in restart file")
    end

    # Read time and time step
    time = read(attrs(file)["time"])
    step = read(attrs(file)["timestep"])

    # Read data
    varnames = equation.varnames_cons
    for v = 1:nvariables(dg)
      # Check if variable name matches
      var = file["variables_$v"]
      if (name = read(attrs(var)["name"])) != varnames[v]
        error("mismatch: variables_$v should be '$(varnames[v])', but found '$name'")
      end

      # Read variable
      println("Reading variables_$v ($name)...")
      dg.u[v, :, :] = read(file["variables_$v"])
    end
  end

  return time, step
end


# Save file capable for restarting the solver.
function save_restart_file(solver::AbstractSolver, mesh::TreeMesh,
                           time::Real, dt::Real, timestep::Integer)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("restart_%06d", timestep))

  # Check output format - restart files are always written in HDF5, as loading
  # from a text file is not supported
  if parameter("output_format", "hdf5") != "hdf5"
    error("parameter 'output_format' must be set to 'hdf5' to write restart files")
  end

  save_restart_file(filename, solver, mesh, convert(Float64, time), convert(Float64, dt), timestep)
end


# Save current DG solution with some context information as a HDF5 file for
# restarting.
function save_restart_file(filename::String, dg::Dg, mesh::TreeMesh,
                           time::Float64, dt::Float64, timestep::Integer)
  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    equation = equations(dg)
    N = polydeg(dg)

    # Add context information as attributes
    attrs(file)["ndim"] = ndim
    attrs(file)["equations"] = equation.name
    attrs(file)["N"] = N
    attrs(file)["n_vars"] = nvariables(dg)
    attrs(file)["n_elements"] = dg.n_elements
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = time
    attrs(file)["dt"] = dt
    attrs(file)["timestep"] = timestep

    # Restart files always store conservative variables
    data = dg.u
    varnames = equation.varnames_cons

    # Store each variable of the solution
    for v = 1:nvariables(dg)
      # Convert to 1D array
      file["variables_$v"] = data[v, :, :][:]

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = varnames[v]
    end
  end
end


# Save current solution by forming a timestep-based filename and then
# dispatching on the 'output_format' parameter.
function save_solution_file(solver::AbstractSolver, mesh::TreeMesh,
                            time::Real, dt::Real, timestep::Integer)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("solution_%06d", timestep))

  # Dispatch on format property
  output_format = parameter("output_format", "hdf5", valid=["hdf5", "text"])
  save_solution_file(Val(Symbol(output_format)), filename, solver, mesh,
                     convert(Float64, time), convert(Float64, dt), timestep)
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(::Val{:hdf5}, filename::String, dg::Dg,
                            mesh::TreeMesh, time::Float64, dt::Float64,
                            timestep::Integer)
  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    equation = equations(dg)
    N = polydeg(dg)

    # Add context information as attributes
    attrs(file)["ndim"] = ndim
    attrs(file)["equations"] = equation.name
    attrs(file)["N"] = N
    attrs(file)["n_vars"] = nvariables(dg)
    attrs(file)["n_elements"] = dg.n_elements
    attrs(file)["mesh_file"] = splitdir(mesh.current_filename)[2]
    attrs(file)["time"] = time
    attrs(file)["dt"] = dt
    attrs(file)["timestep"] = timestep

    # Add coordinates as 1D arrays
    file["x"] = dg.node_coordinates[:]

    # Convert to primitive variables if requested
    solution_variables = parameter("solution_variables", "conservative",
                                   valid=["conservative", "primitive"])
    if solution_variables == "conservative"
      data = dg.u
      varnames = equation.varnames_cons
    else
      data = cons2prim(equation, dg.u)
      varnames = equation.varnames_prim
    end

    # Store each variable of the solution
    for v = 1:nvariables(dg)
      # Convert to 1D array
      file["variables_$v"] = data[v, :, :][:]

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = varnames[v]
    end
  end
end


# Save current DG solution as a plain text file with fixed-width space-separated
# values, with the first line containing the column names.
function save_solution_file(::Val{:text}, filename::String, dg::Dg, mesh::TreeMesh,
                            time::Float64, dt::Float64, timestep::Integer)
  # Open file (clobber existing content)
  open(filename * ".dat", "w") do file
    equation = equations(dg)
    N = polydeg(dg)
    n_nodes = N + 1

    # Convert to primitive variables if requested
    output_variables = parameter("output_variables",
                                "conservative",
                                valid=["conservative", "primitive"])
    if output_variables == "conservative"
      data = dg.u
      varnames = equation.varnames_cons
    else
      data = cons2prim(equation, dg.u)
      varnames = equation.varnames_prim
    end

    # Add context information as comments in the first lines of the file
    println(file, "# ndim = $ndim")
    println(file, "# equations = \"$(equation.name)\"")
    println(file, "# N = $N")
    println(file, "# n_vars = $(nvariables(dg))")
    println(file, "# n_elements = $(dg.n_elements)")
    println(file, "# mesh_file = \"$(mesh.current_filename)\"")
    println(file, "# time = $(time)")
    println(file, "# dt = $(dt)")
    println(file, "# timestep = $(timestep)")

    # Write column names, put in quotation marks to account for whitespace in names
    columns = Vector{String}(undef, ndim + nvariables(dg))
    columns[1] = @sprintf("%-15s", "\"x\"")
    for v = 1:nvariables(dg)
      columns[v+1] = @sprintf("%-15s", "\"$(varnames[v])\"")
    end
    println(file, strip(join(columns, " ")))

    # Write data
    for element_id = 1:dg.n_elements, i = 1:n_nodes
      data_out = Vector{String}(undef, ndim + nvariables(dg))
      data_out[1] = @sprintf("%+10.8e", dg.node_coordinates[i, element_id])
      for v = 1:nvariables(dg)
        data_out[v+1] = @sprintf("%+10.8e", data[v, i, element_id])
      end
      println(file, join(data_out, " "))
    end
  end
end


# Save mesh file
function save_mesh_file(mesh::TreeMesh, timestep::Integer=-1)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Determine file name based on existence of meaningful time step
  if timestep >= 0
    filename = joinpath(output_directory, @sprintf("mesh_%06d", timestep))
  else
    filename = joinpath(output_directory, "mesh")
  end

  # Dispatch on format property
  output_format = parameter("output_format", "hdf5", valid=["hdf5", "text"])
  filename = save_mesh_file(Val(Symbol(output_format)), filename, mesh)

  return filename
end


# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(::Val{:hdf5}, filename::String, mesh::TreeMesh)
  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    # Add context information as attributes
    n_cells = length(mesh.tree)
    attrs(file)["ndim"] = ndim
    attrs(file)["n_cells"] = n_cells
    attrs(file)["n_leaf_cells"] = count_leaf_cells(mesh.tree)
    attrs(file)["minimum_level"] = minimum_level(mesh.tree)
    attrs(file)["maximum_level"] = maximum_level(mesh.tree)
    attrs(file)["center_level_0"] = mesh.tree.center_level_0
    attrs(file)["length_level_0"] = mesh.tree.length_level_0

    # Add tree data
    file["parent_ids"] = @view mesh.tree.parent_ids[1:n_cells]
    file["child_ids"] = @view mesh.tree.child_ids[:, 1:n_cells]
    file["neighbor_ids"] = @view mesh.tree.neighbor_ids[:, 1:n_cells]
    file["levels"] = @view mesh.tree.levels[1:n_cells]
    file["coordinates"] = @view mesh.tree.coordinates[:, 1:n_cells]
  end

  return filename * ".h5"
end


# Save current mesh with some context information as a text file.
function save_mesh_file(::Val{:text}, filename::String, mesh::TreeMesh)
  # Open file (clobber existing content)
  open(filename * ".dat", "w") do file
    # Add context information as comments in the first lines of the file
    n_cells = length(mesh.tree)
    println(file, "# ndim = $(ndim)")
    println(file, "# n_cells = $(n_cells)")
    println(file, "# n_leaf_cells = $(count_leaf_cells(mesh.tree))")
    println(file, "# minimum_level = $(minimum_level(mesh.tree))")
    println(file, "# maximum_level = $(maximum_level(mesh.tree))")
    println(file, "# center_level_0 = $(mesh.tree.center_level_0)")
    println(file, "# length_level_0 = $(mesh.tree.length_level_0)")

    # Write column names, put in quotation marks to account for whitespace in names
    n_columns = (1                                # parent ids
                 + n_children_per_cell(mesh.tree) # child ids
                 + n_directions(mesh.tree)        # neighbor ids
                 + 1                              # levels
                 + ndim)                          # coordinates
    columns = Vector{String}(undef, n_columns)

    # Parent ids
    offset = 0
    columns[offset + 1] = @sprintf("%-10s", "parent_ids")
    offset += 1

    # Child ids
    for i in 1:n_children_per_cell(mesh.tree)
      columns[offset + i] = @sprintf("%-11s", "child_ids_$i")
    end
    offset += n_children_per_cell(mesh.tree)

    # Neighbor ids
    for i in 1:n_directions(mesh.tree)
      columns[offset + i] = @sprintf("%-14s", "neighbor_ids_$i")
    end
    offset += n_directions(mesh.tree)

    # Levels
    columns[offset + 1] = @sprintf("%-6s", "levels")
    offset += 1

    # Coordinates
    for i = 1:ndim
      columns[offset + i] = @sprintf("%-15s", "coordinates_$i")
    end
    offset += ndim

    # Print to file
    println(file, strip(join(columns, "  ")))

    # Write data
    for cell_id = 1:n_cells
      data_out = Vector{String}(undef, n_columns)

      # Parent ids
      offset = 0
      data_out[offset + 1] = @sprintf("%10d", mesh.tree.parent_ids[cell_id])
      offset += 1

      # Child ids
      for i = 1:n_children_per_cell(mesh.tree)
        data_out[offset + i] = @sprintf("%11d", mesh.tree.child_ids[i, cell_id])
      end
      offset += n_children_per_cell(mesh.tree)

      # Neighbor ids
      for i = 1:n_directions(mesh.tree)
        data_out[offset + i] = @sprintf("%14d", mesh.tree.neighbor_ids[i, cell_id])
      end
      offset += n_directions(mesh.tree)

      # Levels
      data_out[offset + 1] = @sprintf("%6d", mesh.tree.levels[cell_id])
      offset += 1

      # Coordinates
      for i = 1:ndim
        data_out[offset + i] = @sprintf("%+10.8e", mesh.tree.coordinates[i, cell_id])
      end
      offset += ndim

      # Print to file
      println(file, join(data_out, "  "))
    end
  end

  return filename * ".dat"
end


end # module
