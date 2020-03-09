module Io

using ..Trixi
using ..Solvers: AbstractSolver, polydeg, equations, Dg
using ..Solvers.DgSolver: polydeg
using ..Equations: nvariables, cons2prim
using ..Auxiliary: parameter
using ..Mesh: TreeMesh
using ..Mesh.Trees: Tree, count_leaf_cells, minimum_level, maximum_level,
                    n_children_per_cell, n_directions
using ..Parallel: is_mpi_root

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
      dg.elements.u[v, :, :, :] = read(file["variables_$v"])
    end
  end

  return time, step
end


# Save current DG solution with some context information as a HDF5 file for
# restarting.
function save_restart_file(dg::Dg, mesh::TreeMesh, time::Real, dt::Real, timestep::Integer)
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
    data = dg.elements.u
    varnames = equation.varnames_cons

    # Store each variable of the solution
    for v = 1:nvariables(dg)
      # Convert to 1D array
      file["variables_$v"] = data[v, :, :, :][:]

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = varnames[v]
    end
  end
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(dg::Dg, mesh::TreeMesh, time::Real, dt::Real, timestep::Integer)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("solution_%06d", timestep))

  # Convert time and time step size to floats
  time = convert(Float64, time)
  dt = convert(Float64, dt)

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
    file["x"] = dg.elements.node_coordinates[1, :, :, :][:]
    file["y"] = dg.elements.node_coordinates[2, :, :, :][:]

    # Convert to primitive variables if requested
    solution_variables = parameter("solution_variables", "conservative",
                                   valid=["conservative", "primitive"])
    if solution_variables == "conservative"
      data = dg.elements.u
      varnames = equation.varnames_cons
    else
      data = cons2prim(equation, dg.elements.u)
      varnames = equation.varnames_prim
    end

    # Store each variable of the solution
    for v = 1:nvariables(dg)
      # Convert to 1D array
      file["variables_$v"] = data[v, :, :, :][:]

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = varnames[v]
    end
  end
end


# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(mesh::TreeMesh, timestep::Integer=-1)
  # Determine output directory
  output_directory = parameter("output_directory", "out")

  # Determine file name based on existence of meaningful time step
  if timestep >= 0
    filename = joinpath(output_directory, @sprintf("mesh_%06d", timestep))
  else
    filename = joinpath(output_directory, "mesh")
  end

  # Since mesh should be replicated *exactly* on all domains, only write from root
  if is_mpi_root()
    # Create output directory (if it does not exist)
    mkpath(output_directory)

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
  end

  return filename * ".h5"
end


end # module
