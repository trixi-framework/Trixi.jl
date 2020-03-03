#!/usr/bin/env julia

module TrixiPlot

# Get useful bits and pieces from trixi
include("../src/auxiliary/auxiliary.jl")
include("../src/solvers/interpolation.jl")

module Mesh
include("../src/mesh/trees.jl")
end

const ndim = 2

using .Interpolation: gauss_lobatto_nodes_weights,
                      polynomial_interpolation_matrix, interpolate_nodes
using .Mesh.Trees: n_children_per_cell
using ArgParse: ArgParseSettings, @add_arg_table, parse_args
using HDF5: h5open, attrs
using Plots: plot, plot!, gr, savefig, scatter!, text, contourf, contourf!, heatmap, heatmap!
using TimerOutputs
using Pkg.TOML: parse
using DelimitedFiles: readdlm
import GR

function main()
  reset_timer!()

  # Parse command line arguments
  args = parse_commandline_arguments()

  # Iterate over input files
  for datafile in 1:length(args["datafile"])
    # Determine input file format
    input_format = get_input_format(datafile)
    @assert input_format == :hdf5 "Only HDF5 files are currently supported"

    # Get mesh file name
    meshfile = extract_mesh_filename(Val(input_format), datafile)

    # Read mesh
    @timeit "read mesh" center_level_0, length_level_0, leaf_cells, coordinates, levels =
      read_meshfile(Val(input_format), meshfile)

    # Read data
    @timeit "read data" labels, node_coordinates_raw, data_raw, n_nodes = read_datafile(
        Val(input_format), datafile)

    # Interpolate DG data to visualization nodes
    nvisnodes = (args["nvisnodes"] == nothing ? 4 * n_nodes : args["nvisnodes"])
    @timeit "interpolate data" begin
      if nvisnodes == 0
        node_coordinates = node_coordinates_raw
        data = data_raw
      else
        node_coordinates = interpolate_data(node_coordinates_raw, n_nodes, nvisnodes)
        data = interpolate_data(data_raw, n_nodes, nvisnodes)
      end
    end

    # Reshape data arrays for convenience
    n_elements = length(levels)
    n_variables = length(labels)
    n_visnodes = nvisnodes == 0 ? n_nodes : nvisnodes
    node_coordinates = reshape(node_coordinates, n_visnodes, n_visnodes, n_elements, ndim)
    data = reshape(data, n_visnodes, n_visnodes, n_elements, n_variables)

    # Set up plotting
    output_format = get_output_format(args["format"])
    gr()
    if output_format == :pdf
      GR.inline("pdf")
    elseif output_format == :png
      GR.inline("png")
    else
      error("unknown output format '$output_format'")
    end

    # Create output directory if it does not exist
    mkpath(args["output-directory"])

    for variable_id in 1:n_variables
      # Create plot
      @timeit "create plot" plot(size=(2000,2000), thickness_scaling=1,
                                aspectratio=:equal, legend=:none)

      # Add elements
      @timeit "add elements" for element_id in 1:n_elements
        # Plot element outline
        length = length_level_0 / 2^levels[element_id]
        vertices = cell_vertices(coordinates[:, element_id], length)
        plot!([vertices[1,:]..., vertices[1, 1]], [vertices[2,:]..., vertices[2, 1]],
              linecolor=:black,
              annotate=(coordinates[1, element_id],
                        coordinates[2, element_id],
                        text("$(leaf_cells[element_id])", 4)),
              grid=false)

        # Plot contours
        x = node_coordinates[:, 1, element_id, 1]
        y = node_coordinates[1, :, element_id, 2]
        z = transpose(data[:, :, element_id, variable_id])
        contourf!(x, y, z, levels=20, c=:bluesreds)
      end

      # Determine output file name
      base, _ = splitext(splitdir(datafile)[2])
      output_filename = joinpath(args["output-directory"],
                                 "$(base)_$(labels[variable_id])." * string(output_format))

      # Save file
      @timeit "save plot" savefig(output_filename)
    end
  end

  print_timer()
  println()
end


function interpolate_data(data_in::AbstractArray, n_nodes_in::Integer, n_nodes_out::Integer)
  # Get node coordinates for input and output locations on reference element
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)
  dx = 2/n_nodes_out
  #=nodes_out = collect(range(-1 + dx/2, 1 - dx/2, length=n_nodes_out))=#
  nodes_out = collect(range(-1, 1, length=n_nodes_out))

  # Get interpolation matrix
  vandermonde = polynomial_interpolation_matrix(nodes_in, nodes_out)

  # Create output data structure
  n_elements = div(size(data_in, 1), n_nodes_in^ndim)
  n_variables = size(data_in, 2)
  data_out = Array{eltype(data_in)}(undef, n_nodes_out, n_nodes_out, n_elements, n_variables)

  # Interpolate each variable separately
  for v = 1:n_variables
    # Reshape data to fit expected format for interpolation function
    # FIXME: this "reshape here, reshape later" funny business should be implemented properly
    reshaped = reshape(data_in[:, v], 1, n_nodes_in, n_nodes_in, n_elements)

    # Interpolate data for each cell
    for element_id = 1:n_elements
      data_out[:, :, element_id, v] = interpolate_nodes(reshaped[:, :, :, element_id],
                                                        vandermonde, 1)
    end
  end

  return reshape(data_out, n_nodes_out^ndim * n_elements, n_variables)
end


function cell_vertices(coordinates::AbstractArray{Float64, 1}, length::Float64)
  @assert ndim == 2 "Algorithm currently only works in 2D"
  vertices = zeros(ndim, 2^ndim)
  vertices[1, 1] = coordinates[1] - 1/2 * length
  vertices[2, 1] = coordinates[2] - 1/2 * length
  vertices[1, 2] = coordinates[1] + 1/2 * length
  vertices[2, 2] = coordinates[2] - 1/2 * length
  vertices[1, 3] = coordinates[1] + 1/2 * length
  vertices[2, 3] = coordinates[2] + 1/2 * length
  vertices[1, 4] = coordinates[1] - 1/2 * length
  vertices[2, 4] = coordinates[2] + 1/2 * length

  return vertices
end


function extract_mesh_filename(::Val{:hdf5}, filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract filename relative to data file
    mesh_file = read(attrs(file)["mesh_file"])

    return joinpath(dirname(filename), mesh_file)
  end
end


function read_meshfile(::Val{:hdf5}, filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract basic information
    ndim = read(attrs(file)["ndim"])
    n_cells = read(attrs(file)["n_cells"])
    n_leaf_cells = read(attrs(file)["n_leaf_cells"])
    center_level_0 = read(attrs(file)["center_level_0"])
    length_level_0 = read(attrs(file)["length_level_0"])

    # Extract coordinates, levels, child cells
    coordinates = Array{Float64}(undef, ndim, n_cells)
    coordinates .= read(file["coordinates"])
    levels = Array{Int}(undef, n_cells)
    levels .= read(file["levels"])
    child_ids = Array{Int}(undef, n_children_per_cell(ndim), n_cells)
    child_ids .= read(file["child_ids"])

    # Extract leaf cells (= cells to be plotted) and contract all other arrays accordingly
    leaf_cells = similar(levels)
    n_cells = 0
    for cell_id in 1:length(levels)
      if sum(child_ids[:, cell_id]) > 0
        continue
      end

      n_cells += 1
      leaf_cells[n_cells] = cell_id
    end
    leaf_cells = leaf_cells[1:n_cells]

    coordinates = coordinates[:, leaf_cells]
    levels = levels[leaf_cells]

    return center_level_0, length_level_0, leaf_cells, coordinates, levels
  end
end


function read_datafile(::Val{:hdf5}, filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract basic information
    N = read(attrs(file)["N"])
    n_elements = read(attrs(file)["n_elements"])
    n_variables = read(attrs(file)["n_vars"])

    # Extract labels for legend
    labels = Array{String}(undef, 1, n_variables)
    for v = 1:n_variables
      labels[1, v] = read(attrs(file["variables_$v"])["name"])
    end

    # Extract coordinates
    n_nodes = N + 1
    num_datapoints = n_nodes^ndim * n_elements
    coordinates = Array{Float64}(undef, num_datapoints, ndim)
    coordinates[:, 1] .= read(file["x"])
    coordinates[:, 2] .= read(file["y"])

    # Extract data arrays
    data = Array{Float64}(undef, num_datapoints, n_variables)
    for v = 1:n_variables
      data[:, v] .= read(file["variables_$v"])
    end

    return labels, coordinates, data, n_nodes
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
    n_nodes = N + 1

    # Create data structure for labels (the "-1" since we omit the coordinate)
    labels = Array{String}(undef, 1, length(labels_raw) - 1)
    labels[1, :] .= labels_raw[2:end]

    # Read all data
    data_in = readdlm(file, ' ', Float64)

    # Extract coordinates and data
    coordinates = data_in[:, 1]
    data = data_in[:, 2:end]

    return labels, coordinates, data, n_nodes
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
      help = "Name of Trixi data file to plot (allowed extensions: .h5, .dat)"
      arg_type = String
      required = true
      nargs = '+'
    "--format", "-f"
      help = "Output file format (allowed: png, pdf)"
      arg_type = String
      default = "png"
    "--output-directory", "-o"
      help = "Output directory where generated images are stored (default: \".\")"
      arg_type = String
      default = "."
    "--nvisnodes"
      help = ("Number of visualization nodes per cell "
              * "(default: four times the number of DG nodes). "
              * "A value of zero prevents any interpolation of data.")
      arg_type = Int
      default = nothing
    "--no-mesh"
      help =  ("Do not plot mesh in addition to solution data.")
      action = :store_true
  end

  return parse_args(s)
end

end # module TrixiPlot


if abspath(PROGRAM_FILE) == @__FILE__
  @TrixiPlot.Auxiliary.interruptable TrixiPlot.main()
end
