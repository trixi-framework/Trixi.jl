module Io

using ..Jul1dge
using ..DgMod: polydeg, syseqn
using ..Equation: nvars
using ..Auxiliary: parameter

using HDF5: h5open, attrs
using Printf: @sprintf

export save_solution_file


"""
    save_solution_file(dg, timestep::Integer)

Save current DG solution by forming a timestep-based filename and then
dispatching on the 'output_format' parameter.
"""
function save_solution_file(dg, timestep::Integer)
  # Filename without extension based on current time step
  filename = @sprintf("solution_%06d", timestep) 

  # Dispatch on property
  output_format = parameter("output_format", "hdf5", valid=["hdf5", "text"])
  save_solution_file(Val(Symbol(output_format)), dg, filename::String)
end


"""
    save_solution_file(::Val{:hdf5}, dg, filename::String)

Save current DG solution with some context information as a HDF5 file for
postprocessing.
"""
function save_solution_file(::Val{:hdf5}, dg, filename::String)
  # Open file (clobber existing content)
  h5open(filename * ".h5", "w") do file
    s = syseqn(dg)
    N = polydeg(dg)
    nvars_ = nvars(dg)

    # Add context information as attributes
    attrs(file)["ndim"] = ndim
    attrs(file)["syseqn"] = s.name
    attrs(file)["polydeg"] = N
    attrs(file)["nvars"] = nvars_
    attrs(file)["ncells"] = dg.ncells

    # Add coordinates as 1D arrays
    file["coordinates_0"] = dg.nodecoordinate[:]

    # Store each variable of the solution
    for v = 1:nvars_
      # Convert to 1D array
      file["variables_$v"] = dg.u[v, :, :][:]

      # Add variable name as attribute
      var = file["variables_$v"]
      attrs(var)["name"] = s.varnames[v]
    end
  end
end


"""
    save_solution_file(::Val{:text}, dg, filename::String)

Save current DG solution as a plain text file with fixed-width space-separated
values, with the first line containing the column names.
"""
function save_solution_file(::Val{:text}, dg, filename::String)
  # Open file (clobber existing content)
  open(filename * ".dat", "w") do file
    s = syseqn(dg)
    N = polydeg(dg)
    nnodes = N + 1
    nvars_ = nvars(dg)

    # Write column names, put in quotation marks to account for whitespace in names
    columns = Vector{String}()
    for d = 1:ndim
      push!(columns, "\"coordinates_$d\"")
    end
    for v = 1:nvars_
      push!(columns, "\"$(s.varnames[v])\"")
    end
    println(file, join(columns, " "))

    # Write data
    for cell_id = 1:dg.ncells, i = 1:nnodes
      data = Vector{String}()
      push!(data, @sprintf("% 10.8e", dg.nodecoordinate[i, cell_id]))
      for v = 1:nvars_
        push!(data, @sprintf("% 10.8e", dg.u[v, i, cell_id]))
      end
      println(file, join(data, " "))
    end
  end
end




####################################################################################################
# Original version with direct solution-to-image output
#
# using Plots
# import GR
# 
# export plot2file
# 
# function plot2file(dg, filename)
#   gr()
#   GR.inline("png")
#   x = dg.nodecoordinate[:]
#   y = zeros(length(x), nvars(dg))
#   s = syseqn(dg)
#   nnodes = polydeg(dg) + 1
#   for v = 1:nvars(dg)
#     for c in 1:dg.ncells
#       for i = 1:nnodes
#         y[(c - 1) * nnodes + i, v] = dg.u[v, i, c]
#       end
#     end
#     plot(x, y, label=s.varnames[:], xlims=(-10.5, 10.5), ylims=(-1, 2),
#          size=(1600,1200), thickness_scaling=3)
#   end
#   savefig(filename)
# end
####################################################################################################

end
