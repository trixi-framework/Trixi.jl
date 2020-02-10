module Io

using ..Jul1dge
using ..Solvers.DgMod: polydeg, syseqn
using ..Equations: nvars, cons2prim
using ..Auxiliary: parameter

using HDF5: h5open, attrs
using Printf: @sprintf

export save_solution_file


# Save current DG solution by forming a timestep-based filename and then
# dispatching on the 'output_format' parameter.
function save_solution_file(dg, timestep::Integer)
  # Create output directory (if it does not exist)
  output_directory = parameter("output_directory", "out")
  mkpath(output_directory)

  # Filename without extension based on current time step
  filename = joinpath(output_directory, @sprintf("solution_%06d", timestep))

  # Dispatch on format property
  output_format = parameter("output_format", "hdf5", valid=["hdf5", "text"])
  save_solution_file(Val(Symbol(output_format)), dg, filename::String)
end


# Save current DG solution with some context information as a HDF5 file for
# postprocessing.
function save_solution_file(::Val{:hdf5}, dg, filename::String)
  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    s = syseqn(dg)
    N = polydeg(dg)
    nvars_ = nvars(dg)

    # Add context information as attributes
    attrs(file)["ndim"] = ndim
    attrs(file)["syseqn"] = s.name
    attrs(file)["N"] = N
    attrs(file)["nvars"] = nvars_
    attrs(file)["ncells"] = dg.ncells

    # Add coordinates as 1D arrays
    file["x"] = dg.nodecoordinate[:]

    # Convert to primitive variables if requested
    solution_variables = parameter("solution_variables", "conservative",
                                   valid=["conservative", "primitive"])
    if solution_variables == "conservative"
      data = dg.u
      varnames = s.varnames_cons
    else
      data = cons2prim(s, dg.u)
      varnames = s.varnames_prim
    end

    # Store each variable of the solution
    for v = 1:nvars_
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
function save_solution_file(::Val{:text}, dg, filename::String)
  # Open file (clobber existing content)
  open(filename * ".dat", "w") do file
    s = syseqn(dg)
    N = polydeg(dg)
    nnodes = N + 1
    nvars_ = nvars(dg)

    # Convert to primitive variables if requested
    output_variables = parameter("output_variables",
                                "conservative",
                                valid=["conservative", "primitive"])
    if output_variables == "conservative"
      data = dg.u
      varnames = s.varnames_cons
    else
      data = cons2prim(s, dg.u)
      varnames = s.varnames_prim
    end

    # Add context information as comments in the first lines of the file
    println(file, "# ndim = $ndim")
    println(file, "# syseqn = \"$(s.name)\"")
    println(file, "# N = $N")
    println(file, "# nvars = $nvars_")
    println(file, "# ncells = $(dg.ncells)")

    # Write column names, put in quotation marks to account for whitespace in names
    columns = Vector{String}(undef, ndim + nvars_)
    columns[1] = @sprintf("%-15s", "\"x\"")
    for v = 1:nvars_
      columns[v+1] = @sprintf("%-15s", "\"$(varnames[v])\"")
    end
    println(file, strip(join(columns, " ")))

    # Write data
    for cell_id = 1:dg.ncells, i = 1:nnodes
      data_out = Vector{String}(undef, ndim + nvars_)
      data_out[1] = @sprintf("%+10.8e", dg.nodecoordinate[i, cell_id])
      for v = 1:nvars_
        data_out[v+1] = @sprintf("%+10.8e", data[v, i, cell_id])
      end
      println(file, join(data_out, " "))
    end
  end
end

end
