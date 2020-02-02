module Io

using ..Jul1dge
using ..DgMod
import ..Equation

using HDF5
using Printf

export save_solution_file


"""
    save_solution_file(dg, timestep::Integer)

Save current DG solution with some context information as a HDF5 file for postprocessing.
"""
function save_solution_file(dg, timestep::Integer)
  # Filename based on current time step
  filename = @sprintf("solution_%06d.h5", timestep) 

  # Open file (clobber existing content)
  h5open(filename, "w") do file
    s = syseqn(dg)
    N = polydeg(dg)
    nvars_ = Equation.nvars(dg)

    # Get root group
    root = file["/"]

    # Add coordinates as 1D arrays
    for d = 1:ndim
      root["coordinates_$d"] = dg.nodecoordinate[:]
    end

    # Store each variable of the solution
    for v = 1:nvars_
      # Convert to 1D array
      root["variables_$v"] = dg.u[v, :, :][:]

      # Add variable name as attribute
      var = root["variables_$v"]
      attrs(var)["name"] = s.varnames[v]
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
#   y = zeros(length(x), Equation.nvars(dg))
#   s = syseqn(dg)
#   nnodes = polydeg(dg) + 1
#   for v = 1:Equation.nvars(dg)
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
