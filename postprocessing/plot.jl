#!/usr/bin/env julia

# Get useful bits and pieces from jul1dge
include("../src/auxiliary/auxiliary.jl")

using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5: h5open

function main()
  # Parse command line arguments
  args = parse_commandline_arguments()

  input_format = get_input_format(args["datafile"])
  output_format = get_output_format(args["output"])

  labels, data = read_datafile(Val(input_format), args["datafile"])

  println(labels)
  println(data)
end


function read_datafile(::Val{:hdf5}, filename)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract basic information
    N = attrs(file)["polydeg"]
    ncells = attrs(file)["ncells"]
    nvars = attrs(file)["nvars"]

    # Extract labels for legend
    labels = Vector{String}[]
    for v = 1:nvars
      push!(labels, attrs(file["variables_$v"])["name"])
    end

    # Extract data arrays
    nnodes = N + 1
    num_datapoints = nnodes * ncells
    data = Array{Float64}(undef, nnodes, nvars)
    for v = 1:nvars
      data[:, v] .= file["variables_$v"]
    end
  end

  return labels, data
end


function get_input_format(filename::String)
  _, ext = splitext(filename)
  if ext == ".h5"
    return :hdf5
  elseif ext == ".dat"
    return :text
  else
    error("unrecognized input file extension '$ext' (must be '.h5' or '.dat')")
  end
end


function get_output_format(filename::String)
  _, ext = splitext(filename)
  if ext == ".png"
    return :png
  elseif ext == ".pdf"
    return :pdf
  else
    error("unrecognized output file extension '$ext' (must be '.png' or '.pdf')")
  end
end


function parse_commandline_arguments()
  s = ArgParseSettings()
  @add_arg_table s begin
    "datafile"
      help = "Name of Jul1dge data file to plot (allowed extensions: .h5, .dat)"
      arg_type = String
      required = true
    "--output", "-o"
      help = "Name of output file (allowed extensions: .png, .pdf)"
      arg_type = String
      required = true
    "--resolution", "-r"
      help = "Resolution of output file in pixels. Two values expected, x and y."
      nargs = 2
  end

  return parse_args(s)
end

@Auxiliary.interruptable main()


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
