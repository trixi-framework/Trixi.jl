#!/usr/bin/env julia

# Get useful bits and pieces from jul1dge
include("../src/auxiliary/auxiliary.jl")
include("../src/dg/interpolation.jl")

using .Interpolation: gausslobatto, polynomialinterpolationmatrix, interpolate_nodes
using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5: h5open, attrs
using Plots: plot, gr, savefig
using TimerOutputs
import GR

function main()
  reset_timer!()

  # Parse command line arguments
  args = parse_commandline_arguments()

  # Determine input and output file formats
  input_format = get_input_format(args["datafile"])
  output_format = get_output_format(args["output"])

  # Read data from file
  print("Reading data... ")
  @timeit "read data" labels, x_raw, y_raw, nnodes = read_datafile(Val(input_format), args["datafile"])
  println("done")

  # Interpolate DG data to visualization nodes
  nvisnodes = (args["nvisnodes"] == nothing ? 4 * nnodes : args["nvisnodes"])
  println("Interpolating data... ")
  @timeit "interpolate data" begin
    if nvisnodes == 0
      x = x_raw
      y = y_raw
    else
      x = interpolate_data(x_raw, nnodes, nvisnodes)
      y = interpolate_data(y_raw, nnodes, nvisnodes)
    end
  end
  println("done")

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
  print("Creating plot... ")
  @timeit "create plot" plot(x, y, label=labels, size=(1600,1200), thickness_scaling=3)
  println("done")
  print("Saving plot... ")
  @timeit "save plot" savefig(args["output"])
  println("done")

  print_timer()
  println()
end


function interpolate_data(data_in::AbstractArray, nnodes_in::Integer, nnodes_out::Integer)
  # Get node coordinates for input and output locations on reference element
  nodes_in, _ = gausslobatto(nnodes_in)
  dx = 2/nnodes_out
  nodes_out = collect(range(-1 + dx/2, 1 - dx/2, length=nnodes_out))

  # Get interpolation matrix
  vandermonde = polynomialinterpolationmatrix(nodes_in, nodes_out)

  # Create output data structure
  ncells = div(size(data_in, 1), nnodes_in)
  nvars = size(data_in, 2)
  data_out = Array{eltype(data_in)}(undef, nnodes_out, ncells, nvars)

  # Interpolate each variable separately
  for v = 1:nvars
    # Reshape data to fit expected format for interpolation function
    # FIXME: this reshape here, reshape later funny business should be implemented properly
    reshaped = reshape(data_in[:, v], 1, nnodes_in, ncells)

    # Interpolate data for each cell
    for cell_id = 1:ncells
      data_out[:, cell_id, v] = interpolate_nodes(reshaped[:, :, cell_id], vandermonde, 1)
    end
  end

  return reshape(data_out, nnodes_out * ncells, nvars)
end


function read_datafile(::Val{:hdf5}, filename::String)
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

    return labels, coordinates, data, nnodes
  end
end


function read_datafile(::Val{:text}, filename::String)
  error("not implemented")
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
    "--nvisnodes"
      help = ("Number of visualization nodes per cell "
              * "(default: four times the number of DG nodes). "
              * "A value of zero prevents any interpolation of data.")
      arg_type = Int
      default = nothing
  end

  return parse_args(s)
end

@Auxiliary.interruptable main()
