#
#  To use do the following
#    using Revise
#    includet("utils/trixi2txt.jl")
#    using .Trixi2Txt
#    Trixi2Txt.trixi2txt("out/file_name")
#
#  It may be that you need to install the Glob package by running
#  `import Pkg; Pkg.add("Glob")`.
#
#  After the HDF5 files have been converted to `.txt` the 1D solution can be
#  visualized in ParaView with the following steps:
#
#    1) Open the set of `solution_*.txt` files
#    2) Change the "Field Delimiter Characters" field to be a space instead of a comma
#    3) Check the box for "Merge Consecutive Delimiters"
#    4) Create a plot using Filters -> Data Analysis -> Plot Data
#    5) Within Plot Data, uncheck "Use Index For XAxis"
#    6) Select the `x` values for "X Array Name"
#    7) Now you can adjust the plots and make movies or save screenshots of the solution
module Trixi2Txt

using EllipsisNotation
using Glob: glob
using Printf: @printf
using HDF5: h5open, attributes, haskey
using MuladdMacro: @muladd
# using Tullio: @tullio
using LoopVectorization
using StaticArrays
using UnPack: @unpack

include("../src/basic_types.jl")
include("../src/solvers/dgsem/basis_lobatto_legendre.jl")
include("../src/solvers/dgsem/interpolation.jl")

function trixi2txt(filename::AbstractString...;
                   variables=[], output_directory=".", nvisnodes=nothing, max_supported_level=11)
  # Convert filenames to a single list of strings
  if isempty(filename)
    error("no input file was provided")
  end
  filenames = String[]
  for pattern in filename
    append!(filenames, glob(pattern))
  end

  # Iterate over input files
  for (index, filename) in enumerate(filenames)
    # Check if data file exists
    if !isfile(filename)
      error("file '$filename' does not exist")
    end

    # Make sure it is a data file
    if !is_solution_restart_file(filename)
      error("file '$filename' is not a data file")
    end

    # Get mesh file name
    meshfile = extract_mesh_filename(filename)

    # Check if mesh file exists
    if !isfile(meshfile)
      error("mesh file '$meshfile' does not exist")
    end

    # Read mesh
    center_level_0, length_level_0, leaf_cells, coordinates, levels = read_meshfile(meshfile)

    # Read data
    labels, data, n_elements, n_nodes, element_variables, time = read_datafile(filename)

    # Check if dimensions match
    if length(leaf_cells) != n_elements
      error("number of elements in '$(filename)' do not match number of leaf cells in " *
            "'$(meshfile)' " *
            "(did you forget to clean your 'out/' directory between different runs?)")
    end

    # Determine resolution for data interpolation
    max_level = maximum(levels)
    if max_level > max_supported_level
      error("Maximum refinement level in data file $max_level is higher than " *
            "maximum supported level $max_supported_level")
    end
    max_available_nodes_per_finest_element = 2^(max_supported_level - max_level)
    if nvisnodes == nothing
      max_nvisnodes = 2 * n_nodes
    elseif nvisnodes == 0
      max_nvisnodes = n_nodes
    else
      max_nvisnodes = nvisnodes
    end
    nvisnodes_at_max_level = min(max_available_nodes_per_finest_element, max_nvisnodes)
    resolution = nvisnodes_at_max_level * 2^max_level
    nvisnodes_per_level = [2^(max_level - level)*nvisnodes_at_max_level for level in 0:max_level]

    # Interpolate data
    structured_data = unstructured2structured(data, levels, resolution, nvisnodes_per_level)

    # Interpolate cell-centered values to node-centered values
    node_centered_data = cell2node(structured_data)

    # Determine x coordinates
    xs = collect(range(-1, 1, length=resolution+1)) .* length_level_0/2 .+ center_level_0[1]

    # Check that all variables exist in data file
    if isempty(variables)
      append!(variables, labels)
    else
      for var in variables
        if !(var in labels)
          error("variable '$var' does not exist in the data file $filename")
        end
      end
    end

    # Create output directory if it does not exist
    mkpath(output_directory)

    # Determine output file name
    base, _ = splitext(splitdir(filename)[2])
    output_filename = joinpath(output_directory, "$(base).txt")

    # Write to file
    open(output_filename, "w") do io
      # Header
      print(io, "x             ")
      for label in variables
        @printf(io, "  %-14s", label)
      end
      println(io)

      # Data
      for idx in 1:length(xs)
        @printf(io, "%+10.8e", xs[idx])
        for variable_id in 1:length(variables)
          @printf(io, " %+10.8e ", node_centered_data[idx, variable_id])
        end
        println(io)
      end
    end
  end
end


# Check if file is a data file
function is_solution_restart_file(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # If attribute "mesh_file" exists, this must be a data file
    return haskey(attributes(file), "mesh_file")
  end
end


# Use data file to extract mesh filename from attributes
function extract_mesh_filename(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract filename relative to data file
    mesh_file = read(attributes(file)["mesh_file"])

    return joinpath(dirname(filename), mesh_file)
  end
end


# Read in mesh file and return relevant data
function read_meshfile(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Check dimension - only 1D supported
    if haskey(attributes(file), "ndims")
      ndims_ = read(attributes(file)["ndims"])
    else
      ndims_ = read(attributes(file)["ndim"]) # FIXME once Trixi.jl's 3D branch is merged & released
    end
    if ndims_ != 1
      error("currently only 1D files can be processed, but '$filename' is $(ndims_)D")
    end

    # Extract basic information
    n_cells = read(attributes(file)["n_cells"])
    n_leaf_cells = read(attributes(file)["n_leaf_cells"])
    center_level_0 = read(attributes(file)["center_level_0"])
    length_level_0 = read(attributes(file)["length_level_0"])

    # Extract coordinates, levels, child cells
    coordinates = Array{Float64}(undef, ndims_, n_cells)
    coordinates .= read(file["coordinates"])
    levels = Array{Int}(undef, n_cells)
    levels .= read(file["levels"])
    child_ids = Array{Int}(undef, 2^ndims_, n_cells)
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


# Read in data file and return all relevant information
function read_datafile(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract basic information
    if haskey(attributes(file), "ndims")
      ndims_ = read(attributes(file)["ndims"])
    else
      ndims_ = read(attributes(file)["ndim"])
    end
    if haskey(attributes(file), "polydeg")
      polydeg = read(attributes(file)["polydeg"])
    else
      polydeg = read(attributes(file)["N"])
    end
    n_elements = read(attributes(file)["n_elements"])
    n_variables = read(attributes(file)["n_vars"])
    time = read(attributes(file)["time"])

    # Extract labels for legend
    labels = Array{String}(undef, 1, n_variables)
    for v = 1:n_variables
      labels[1, v] = read(attributes(file["variables_$v"])["name"])
    end

    # Extract data arrays
    n_nodes = polydeg + 1

    if ndims_ == 1
      data = Array{Float64}(undef, n_nodes, n_elements, n_variables)
      for v = 1:n_variables
        vardata = read(file["variables_$v"])
        @views data[:, :, v][:] .= vardata
      end
    else
      error("Unsupported number of spatial dimensions: ", ndims_)
    end

    # Extract element variable arrays
    element_variables = Dict{String, Union{Vector{Float64}, Vector{Int}}}()
    index = 1
    while haskey(file, "element_variables_$index")
      varname = read(attributes(file["element_variables_$index"])["name"])
      element_variables[varname] = read(file["element_variables_$index"])
      index +=1
    end

    return labels, data, n_elements, n_nodes, element_variables, time
  end
end


# Interpolate unstructured DG data to structured data (cell-centered)
function unstructured2structured(unstructured_data::AbstractArray{Float64},
                                 levels::AbstractArray{Int}, resolution::Int,
                                 nvisnodes_per_level::AbstractArray{Int})
  # Extract data shape information
  n_nodes_in, n_elements, n_variables = size(unstructured_data)

  # Get node coordinates for DG locations on reference element
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)

  # Calculate interpolation vandermonde matrices for each level
  max_level = length(nvisnodes_per_level) - 1
  vandermonde_per_level = []
  for l in 0:max_level
    n_nodes_out = nvisnodes_per_level[l + 1]
    dx = 2 / n_nodes_out
    nodes_out = collect(range(-1 + dx/2, 1 - dx/2, length=n_nodes_out))
    push!(vandermonde_per_level, polynomial_interpolation_matrix(nodes_in, nodes_out))
  end

  # Create output data structure
  structured = Array{Float64}(undef, resolution, n_variables)

  # For each variable, interpolate element data and store to global data structure
  for v in 1:n_variables
    first = 1

    # Reshape data array for use in interpolate_nodes function
    @views reshaped_data = reshape(unstructured_data[:, :, v], 1, n_nodes_in, n_elements)

    for element_id in 1:n_elements
      # Extract level for convenience
      level = levels[element_id]

      # Determine target indices
      n_nodes_out = nvisnodes_per_level[level + 1]
      last = first + (n_nodes_out - 1)

      # Interpolate data
      vandermonde = vandermonde_per_level[level + 1]
      @views structured[first:last, v] .= (
           reshape(multiply_dimensionwise_naive(reshaped_data[:, :, element_id], vandermonde),
                   n_nodes_out))

      # Update first index for next iteration
      first += n_nodes_out
    end
  end

  return structured
end


# Convert cell-centered values to node-centered values by averaging over all
# four neighbors and making use of the periodicity of the solution
function cell2node(cell_centered_data::AbstractArray{Float64})
  # Create temporary data structure to make the averaging algorithm as simple
  # as possible (by using a ghost layer)
  tmp = similar(cell_centered_data, size(cell_centered_data) .+ (2, 0))

  # Fill center with original data
  tmp[2:end-1, :] .= cell_centered_data

  # # Fill sides with opposite data (periodic domain)
  # # x-direction
  # tmp[1,   :] .= cell_centered_data[end, :]
  # tmp[end, :] .= cell_centered_data[1,   :]

  # Fill sides with duplicate information
  # x-direction
  tmp[1,   :] .= cell_centered_data[1,   :]
  tmp[end, :] .= cell_centered_data[end, :]

  # Create output data structure
  resolution_in, n_variables = size(cell_centered_data)
  resolution_out = resolution_in + 1
  node_centered_data = Array{Float64}(undef, resolution_out, n_variables)

  # Obtain node-centered value by averaging over neighboring cell-centered values
  for i in 1:resolution_out
    node_centered_data[i, :] = (tmp[i, :] + tmp[i+1, :]) / 2
  end

  return node_centered_data
end

end
