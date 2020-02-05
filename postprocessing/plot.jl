#!/usr/bin/env julia

# Get useful bits and pieces from jul1dge
include("../src/auxiliary/auxiliary.jl")
include("../src/dg/interpolation.jl")

using .Interpolation: gausslobatto, polynomialinterpolationmatrix, interpolate_nodes
using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5: h5open, attrs
using Plots: plot, gr, savefig
using TimerOutputs
using Pkg.TOML: parse
using DelimitedFiles: readdlm
import GR

function main()
  reset_timer!()

  # Parse command line arguments
  args = parse_commandline_arguments()

  # Determine output file format
  output_format = get_output_format(args["format"])

  # Iterate over input files
  for f = 1:length(args["datafile"])
    # User info
    datafile = args["datafile"][f]
    print("Processing '$datafile'... ")

    # Determine input file format
    input_format = get_input_format(datafile)

    # Read data from file
    @timeit "read data" labels, x_raw, y_raw, nnodes = read_datafile(
        Val(input_format), datafile)

    # Interpolate DG data to visualization nodes
    nvisnodes = (args["nvisnodes"] == nothing ? 4 * nnodes : args["nvisnodes"])
    @timeit "interpolate data" begin
      if nvisnodes == 0
        x = x_raw
        y = y_raw
      else
        x = interpolate_data(x_raw, nnodes, nvisnodes)
        y = interpolate_data(y_raw, nnodes, nvisnodes)
      end
    end

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
    @timeit "create plot" plot(x, y, label=labels, size=(1600,1200), thickness_scaling=3)

    # Determine output file name
    base, _ = splitext(splitdir(datafile)[2])
    output_filename = joinpath(args["output-directory"], base * "." * string(output_format))

    # Create output directory if it does not exist
    mkpath(args["output-directory"])

    # Save file
    @timeit "save plot" savefig(output_filename)

    # User info
    println("done")
  end

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
    N = read(attrs(file)["N"])
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
    coordinates .= read(file["x"])

    # Extract data arrays
    data = Array{Float64}(undef, num_datapoints, nvars)
    for v = 1:nvars
      data[:, v] .= read(file["variables_$v"])
    end

    return labels, coordinates, data, nnodes
  end
end


function read_datafile(::Val{:text}, filename::String)
  # Open file for reading
  open(filename, "r") do file
    # We assume that the input file has the following structure:
    # - zero or more lines starting with hashtag '#', marking comments with context information
    # - one line with the column names (quoted in '"')
    # - the data, columnwise, separated by one or more space characters

    context_raw = ""
    labels_raw = Vector{String}()

    # First, read file line by line
    while true
      line = readline(file)
      if !startswith(line, "#")
        # If a line does not start with a hashtag, the line must contain the column headers
        for m in eachmatch(r"\"([^\"]+)\"", line)
          push!(labels_raw, m.captures[1])
        end
        break
      else
        # Otherwise, save line to process later
        context_raw *= strip(lstrip(line, '#')) * "\n"
      end
    end

    # Extract basic information
    context = parse(context_raw)
    N = context["N"]
    nnodes = N + 1

    # Create data structure for labels (the "-1" since we omit the coordinate)
    labels = Array{String}(undef, 1, length(labels_raw) - 1)
    labels[1, :] .= labels_raw[2:end]

    # Read all data
    data_in = readdlm(file, ' ', Float64)

    # Extract coordinates and data
    coordinates = data_in[:, 1]
    data = data_in[:, 2:end]

    return labels, coordinates, data, nnodes
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


function get_output_format(format::String)
  if format == "png"
    return :png
  elseif format == "pdf"
    return :pdf
  else
    error("unrecognized output file format '$format' (must be 'png' or 'pdf')")
  end
end


function parse_commandline_arguments()
  s = ArgParseSettings()
  @add_arg_table s begin
    "datafile"
      help = "Name of Jul1dge data file to plot (allowed extensions: .h5, .dat)"
      arg_type = String
      required = true
      nargs = '+'
    "--format", "-f"
      help = "Output file format (allowed: png, pdf)"
      arg_type = String
      default = "pdf"
    "--output-directory", "-o"
      help = "Output directory where generated images are stored (default: \".\")"
      arg_type = String
      default = "."
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


if abspath(PROGRAM_FILE) == @__FILE__
  @Auxiliary.interruptable main()
end
