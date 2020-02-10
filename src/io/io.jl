module Io

using ..Jul1dge
using ..Solvers: AbstractSolver, polydeg, equations, Dg
using ..Solvers.DgSolver: polydeg
using ..Equations: nvariables, cons2prim
using ..Auxiliary: parameter

using HDF5: h5open, attrs
using Printf: @sprintf

export save_solution_file


# Save current DG solution by forming a timestep-based filename and then
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
    for cell_id = 1:dg.n_elements, i = 1:n_nodes
      data_out = Vector{String}(undef, ndim + nvariables(dg))
      data_out[1] = @sprintf("%+10.8e", dg.node_coordinates[i, cell_id])
      for v = 1:nvariables(dg)
        data_out[v+1] = @sprintf("%+10.8e", data[v, i, cell_id])
      end
      println(file, join(data_out, " "))
    end
  end
end

end
