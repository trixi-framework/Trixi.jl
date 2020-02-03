#!/usr/bin/env julia

# Get useful bits and pieces from jul1dge
include("../src/auxiliary/auxiliary.jl")

using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5: h5open, attrs
using Plots: plot, gr, savefig
import GR

function main()
  # Parse command line arguments
  args = parse_commandline_arguments()

  # Determine input and output file formats
  input_format = get_input_format(args["datafile"])
  output_format = get_output_format(args["output"])

  # Read data from file
  labels, x, y = read_datafile(Val(input_format), args["datafile"])

  # Set up plotting
  gr()
  if output_format == :pdf
    GR.inline("pdf")
  elseif output_format == :png
    GR.inline("png")
  else
    error("unknown output format '$output_format'")
  end

  # Create plot
  plot(x, y, label=labels, size=(1600,1200), thickness_scaling=3)
  savefig(args["output"])
end


function read_datafile(::Val{:hdf5}, filename)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract basic information
    N = read(attrs(file)["polydeg"])
    ncells = read(attrs(file)["ncells"])
    nvars = read(attrs(file)["nvars"])

    # Extract labels for legend
    labels = Array{String}(undef, 1, nvars)
    for v = 1:nvars
      labels[1, v] = read(attrs(file["variables_$v"])["name"])
    end

    # Extract coordinates
    nnodes = N + 1
    num_datapoints = nnodes * ncells
    coordinates = Array{Float64}(undef, num_datapoints)
    coordinates .= read(file["coordinates_1"])

    # Extract data arrays
    data = Array{Float64}(undef, num_datapoints, nvars)
    for v = 1:nvars
      data[:, v] .= read(file["variables_$v"])
    end

    return labels, coordinates, data
  end
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
