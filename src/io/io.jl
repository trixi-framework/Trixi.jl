module Io

using ..Jul1dge
using ..Solvers: AbstractSolver, polydeg, equations, Dg
using ..Solvers.DgSolver: polydeg
using ..Equations: nvariables, cons2prim
using ..Auxiliary: parameter
using ..Mesh: TreeMesh
using ..Mesh.Trees: Tree, count_leaf_cells, minimum_level, maximum_level,
                    n_children_per_cell, n_directions

using HDF5: h5open, attrs
using Printf: @sprintf

export save_solution_file
export save_mesh_file


# Save current solution by forming a timestep-based filename and then
# dispatching on the 'output_format' parameter.
function save_solution_file(solver::AbstractSolver, timestep::Integer)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("solution_%06d", timestep))

  # Dispatch on format property
  output_format = parameter("output_format", "hdf5", valid=["hdf5", "text"])
  save_solution_file(Val(Symbol(output_format)), solver, filename::String)
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(::Val{:hdf5}, dg::Dg, filename::String)
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
function save_solution_file(::Val{:text}, dg::Dg, filename::String)
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
  filename = save_mesh_file(Val(Symbol(output_format)), mesh, filename::String)

  return filename
end


# Save current mesh with some context information as an HDF5 file.
function save_mesh_file(::Val{:hdf5}, mesh::TreeMesh, filename::String)
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
function save_mesh_file(::Val{:text}, mesh::TreeMesh, filename::String)
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
    columns[offset + 1] = @sprintf("%-12s", "\"parent_ids\"")
    offset += 1

    # Child ids
    for i in 1:n_children_per_cell(mesh.tree)
      columns[offset + i] = @sprintf("%-13s", "\"child_ids_$i\"")
    end
    offset += n_children_per_cell(mesh.tree)

    # Neighbor ids
    for i in 1:n_directions(mesh.tree)
      columns[offset + i] = @sprintf("%-16s", "\"neighbor_ids_$i\"")
    end
    offset += n_directions(mesh.tree)

    # Levels
    columns[offset + 1] = @sprintf("%-8s", "\"levels\"")
    offset += 1

    # Coordinates
    for i = 1:ndim
      columns[offset + i] = @sprintf("%-15s", "\"coordinates_$i\"")
    end
    offset += ndim

    # Print to file
    println(file, strip(join(columns, " ")))

    # Write data
    for cell_id = 1:n_cells
      data_out = Vector{String}(undef, n_columns)

      # Parent ids
      offset = 0
      data_out[offset + 1] = @sprintf("%12d", mesh.tree.parent_ids[cell_id])
      offset += 1

      # Child ids
      for i = 1:n_children_per_cell(mesh.tree)
        data_out[offset + i] = @sprintf("%13d", mesh.tree.child_ids[i, cell_id])
      end
      offset += n_children_per_cell(mesh.tree)

      # Neighbor ids
      for i = 1:n_directions(mesh.tree)
        data_out[offset + i] = @sprintf("%16d", mesh.tree.neighbor_ids[i, cell_id])
      end
      offset += n_directions(mesh.tree)

      # Levels
      data_out[offset + 1] = @sprintf("%8d", mesh.tree.levels[cell_id])
      offset += 1

      # Coordinates
      for i = 1:ndim
        data_out[offset + i] = @sprintf("%+10.8e", mesh.tree.coordinates[i, cell_id])
      end
      offset += ndim

      # Print to file
      println(file, join(data_out, " "))
    end
  end

  return filename * ".dat"
end


end # module
