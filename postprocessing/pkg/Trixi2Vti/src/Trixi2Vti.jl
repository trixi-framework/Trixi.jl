module Trixi2Vti

# Get useful bits and pieces from trixi
include("../../../../src/solvers/interpolation.jl")
include("pointlocators.jl")

# Number of spatial dimensions
const ndim = 2

using .Interpolation: gauss_lobatto_nodes_weights,
                      polynomial_interpolation_matrix, interpolate_nodes
using .PointLocators: PointLocator, insert!, Point

using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using HDF5: h5open, attrs, exists
using WriteVTK: vtk_grid, MeshCell, VTKCellTypes, vtk_save, paraview_collection
using TimerOutputs
using ProgressMeter: @showprogress, Progress, next!


function run(;args=nothing, kwargs...)
  # Reset timer
  reset_timer!()

  # Handle command line or keyword arguments
  args = get_arguments(args, kwargs...)

  # Store for convenience
  verbose = args["verbose"]
  hide_progress = args["hide_progress"]
  filenames = args["filename"]

  # If verbose mode is enabled, always hide progress bar
  if verbose
    hide_progress = true
  end

  # Variable to avoid writing PVD files if only a single file is converted
  is_single_file = length(filenames) == 1

  # Get pvd filenames and open files
  if !is_single_file
    pvd_filename, pvd_celldata_filename = pvd_filenames(args)
    verbose && println("Opening PVD files '$(pvd_filename).pvd' + '$(pvd_celldata_filename).pvd'...")
    @timeit "open PVD file" begin
      pvd = paraview_collection(pvd_filename)
      pvd_celldata = paraview_collection(pvd_celldata_filename)
    end
  end

  # Variable to avoid writing PVD file if only mesh files were converted
  has_data = false

  # Show progress bar if not disabled
  if !hide_progress
    progress = Progress(length(filenames), 0.5, "Converting .h5 to .vti...", 40)
  end

  # Iterate over input files
  for (index, filename) in enumerate(filenames)
    verbose && println("Processing file $filename ($(index)/$(length(filenames)))...")

    # Check if data file exists
    if !isfile(filename)
      error("data file '$filename' does not exist")
    end

    # Check if it is a data file at all
    is_datafile = is_solution_restart_file(filename)

    # If file is solution/restart file, extract mesh file name
    if is_datafile
      # Get mesh file name
      meshfile = extract_mesh_filename(filename)

      # Check if mesh file exists
      if !isfile(meshfile)
        error("mesh file '$meshfile' does not exist")
      end
    else
      meshfile = filename
    end

    # Read mesh
    verbose && println("| Reading mesh file...")
    @timeit "read mesh" (center_level_0, length_level_0,
                         leaf_cells, coordinates, levels) = read_meshfile(meshfile)

    # Read data only if it is a data file
    if is_datafile
      verbose && println("| Reading data file...")
      @timeit "read data" (labels, data, n_elements, n_nodes,
                           element_variables, time) = read_datafile(filename)

      # Check if dimensions match
      if length(leaf_cells) != n_elements
        error("number of elements in '$(filename)' do not match number of leaf cells in " *
              "'$(meshfile)' " *
              "(did you forget to clean your 'out/' directory between different runs?)")
      end

      # Determine resolution for data interpolation
      if args["nvisnodes"] == nothing
        n_visnodes = 2 * n_nodes
      elseif args["nvisnodes"] == 0
        n_visnodes = n_nodes
      else
        n_visnodes = args["nvisnodes"]
      end
    else
      # If file is a mesh file, do not interpolate data
      n_visnodes = 1
    end

    # Create output directory if it does not exist
    mkpath(args["output_directory"])

    # Build VTK grids
    vtk_nodedata, vtk_celldata = build_vtk_grids(coordinates, levels, center_level_0,
                                                 length_level_0, n_visnodes, verbose,
                                                 args["output_directory"], is_datafile, filename)

    # Interpolate data
    if is_datafile
      verbose && println("| Interpolating data...")
      @timeit "interpolate data" interpolated_data = interpolate_data(data, coordinates, levels,
                                                                      center_level_0,
                                                                      length_level_0,
                                                                      n_visnodes, verbose)
    end

    # Add data to file
    verbose && println("| Adding data to VTK file...")
    @timeit "add data to VTK file" begin
      # Add cell/element data to celldata VTK file
      verbose && println("| | cell_ids...")
      @timeit "cell_ids" vtk_celldata["cell_ids"] = leaf_cells
      verbose && println("| | element_ids...")
      @timeit "element_ids" vtk_celldata["element_ids"] = collect(1:length(leaf_cells))
      verbose && println("| | levels...")
      @timeit "levels" vtk_celldata["levels"] = levels

      # Only add data if it is a data file
      if is_datafile
        # Add solution variables
        for (variable_id, label) in enumerate(labels)
          verbose && println("| | Variable: $label...")
          @timeit label vtk_nodedata[label] = @views interpolated_data[:, variable_id]
        end

        # Add element variables
        for (label, variable) in element_variables
          verbose && println("| | Element variable: $label...")
          @timeit label vtk_celldata[label] = variable
        end
      end
    end

    # Save VTK file
    if is_datafile
      verbose && println("| Saving VTK file '$(vtk_filename).vti'...")
      @timeit "save VTK file" vtk_save(vtk_nodedata)
    end

    verbose && println("| Saving VTK file '$(vtk_celldata_filename).vti'...")
    @timeit "save VTK file" vtk_save(vtk_celldata)

    # Add to PVD file only if it is a datafile
    if !is_single_file
      if is_datafile
        verbose && println("| Adding to PVD file...")
        @timeit "add VTK to PVD file" begin
          pvd[time] = vtk_nodedata
          pvd_celldata[time] = vtk_celldata
        end
        has_data = true
      else
        println("WARNING: file '$(filename)' will not be added to PVD file since it is a mesh file")
      end
    end

    # Update progress bar
    if !hide_progress
      next!(progress, showvalues=[(:finished, filename)])
    end
  end

  if !is_single_file
    # Save PVD file only if at least one data file was added
    if has_data
      verbose && println("| Saving PVD file '$(pvd_filename).pvd'...")
      @timeit "save PVD files" vtk_save(pvd)
    end
    verbose && println("| Saving PVD file '$(pvd_celldata_filename).pvd'...")
    @timeit "save PVD files" vtk_save(pvd_celldata)
  end

  verbose && println("| done.\n")
  print_timer()
  println()
end


# Interpolate data from input format to desired output format
function interpolate_data(input_data, coordinates, levels, center_level_0, length_level_0,
                          n_visnodes, verbose)
  # Normalize element coordinates: move center to (0, 0) and domain size to [-1, 1]²
  normalized_coordinates = similar(coordinates)
  for element_id in axes(coordinates, 2)
    @views normalized_coordinates[:, element_id] .= (
        (coordinates[:, element_id] .- center_level_0) ./ (length_level_0 / 2 ))
  end

  # Determine level-wise resolution
  max_level = maximum(levels)
  resolution = n_visnodes * 2^max_level

  # nvisnodes_per_level is an array (accessed by "level + 1" to accommodate
  # level-0-cell) that contains the number of visualization nodes for any
  # refinement level to visualize on an equidistant grid
  nvisnodes_per_level = [2^(max_level - level)*n_visnodes for level in 0:max_level]

  # Interpolate unstructured DG data to structured data
  structured_data = unstructured2structured(input_data, normalized_coordinates, levels,
                                            resolution, nvisnodes_per_level)

  return structured_data
end


# Create and return VTK grids that are ready to be filled with data
function build_vtk_grids(coordinates, levels, center_level_0, length_level_0,
                         n_visnodes, verbose, output_directory, is_datafile, filename)
    # Prepare VTK points and cells for celldata file
    @timeit "prepare VTK cells" vtk_celldata_points, vtk_celldata_cells = calc_vtk_points_cells(
        coordinates, levels, center_level_0, length_level_0, 1)

    # Determine output file names
    base, _ = splitext(splitdir(filename)[2])
    vtk_filename = joinpath(output_directory, base)
    vtk_celldata_filename = vtk_filename * "_celldata"

    # Open VTK files
    verbose && println("| Building VTK grid...")
    if is_datafile
      # Determine level-wise resolution
      max_level = maximum(levels)
      resolution = n_visnodes * 2^max_level

      Nx = Ny = resolution + 1
      dx = dy = length_level_0/resolution
      origin = center_level_0 .- 1/2 * length_level_0
      spacing = [dx, dy]
      @timeit "build VTK grid (node data)" vtk_nodedata = vtk_grid(vtk_filename, Nx, Ny,
                                                          origin=origin,
                                                          spacing=spacing)
    else
      vtk_nodedata = nothing
    end
    @timeit "build VTK grid (cell data)" vtk_celldata = vtk_grid(vtk_celldata_filename,
                                                        vtk_celldata_points,
                                                        vtk_celldata_cells)

  return vtk_nodedata, vtk_celldata
end


# Handle command line arguments (if given) or interpret keyword arguments
function get_arguments(args; kwargs...)
  # Handle command line arguments
  if !isnothing(args)
    # If args are given explicitly, parse command line arguments
    args = parse_commandline_arguments(args)
  else
    # Otherwise interpret keyword arguments as command line arguments
    args = Dict{String, Any}()
    for (key, value) in kwargs
      args[string(key)] = value
    end

    # Clean up some of the arguments and provide defaults
    # FIXME: This is redundant to parse_commandline_arguments
    # If filename is a single string, convert it to array
    if !haskey(args, "filename")
      error("no input file was provided")
    end
    if isa(args["filename"], String)
      args["filename"] = [args["filename"]]
    end
    if !haskey(args, "verbose")
      args["verbose"] = false
    end
    if !haskey(args, "hide_progress")
      args["hide_progress"] = false
    end
    if !haskey(args, "pvd")
      args["pvd"] = nothing
    end
    if !haskey(args, "output_directory")
      args["output_directory"] = "."
    end
    if !haskey(args, "nvisnodes")
      args["nvisnodes"] = nothing
    end
  end

  return args
end


# Determine and return filenames for PVD fiels
function pvd_filenames(args)
  # Determine pvd filename
  if !isnothing(args["pvd"])
    # Use filename if given on command line
    filename = args["pvd"]

    # Strip of directory/extension
    filename, _ = splitext(splitdir(filename)[2])
  else
    filename = get_pvd_filename(args["filename"])

    # If filename is empty, it means we were not able to determine an
    # appropriate file thus the user has to supply one
    if filename == ""
      error("could not auto-detect PVD filename (input file names have no common prefix): " *
            "please provide a PVD filename name with `--pvd`")
    end
  end

  # Get full filenames
  pvd_filename = joinpath(args["output_directory"], filename)
  pvd_celldata_filename = pvd_filename * "_celldata"

  return pvd_filename, pvd_celldata_filename
end


# Interpolate unstructured DG data to structured data (cell-centered)
function unstructured2structured(unstructured_data::AbstractArray{Float64},
                                 normalized_coordinates::AbstractArray{Float64},
                                 levels::AbstractArray{Int}, resolution::Int,
                                 nvisnodes_per_level::AbstractArray{Int})
  # Extract data shape information
  n_nodes_in, _, n_elements, n_variables = size(unstructured_data)

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

  # For each element, calculate index position at which to insert data in global data structure
  lower_left_index = element2index(normalized_coordinates, levels, resolution, nvisnodes_per_level)

  # Create output data structure
  structured = Array{Float64}(undef, resolution, resolution, n_variables)

  # For each variable, interpolate element data and store to global data structure
  for v in 1:n_variables
    # Reshape data array for use in interpolate_nodes function
    reshaped_data = reshape(unstructured_data[:, :, :, v], 1, n_nodes_in, n_nodes_in, n_elements)

    for element_id in 1:n_elements
      # Extract level for convenience
      level = levels[element_id]

      # Determine target indices
      n_nodes_out = nvisnodes_per_level[level + 1]
      first = lower_left_index[:, element_id]
      last = first .+ (n_nodes_out - 1)

      # Interpolate data
      vandermonde = vandermonde_per_level[level + 1]
      structured[first[1]:last[1], first[2]:last[2], v] .= (
          reshape(interpolate_nodes(reshaped_data[:, :, :, element_id], vandermonde, 1),
                  n_nodes_out, n_nodes_out))
    end
  end

  # Return as one 1D array for each variable
  return reshape(structured, resolution^ndim, n_variables)
end


# For a given normalized element coordinate, return the index of its lower left
# contribution to the global data structure
function element2index(normalized_coordinates::AbstractArray{Float64}, levels::AbstractArray{Int},
                       resolution::Int, nvisnodes_per_level::AbstractArray{Int})
  n_elements = length(levels)

  # First, determine lower left coordinate for all cells
  dx = 2 / resolution
  lower_left_coordinate = Array{Float64}(undef, ndim, n_elements)
  for element_id in 1:n_elements
    nvisnodes = nvisnodes_per_level[levels[element_id] + 1]
    lower_left_coordinate[1, element_id] = (
        normalized_coordinates[1, element_id] - (nvisnodes - 1)/2 * dx)
    lower_left_coordinate[2, element_id] = (
        normalized_coordinates[2, element_id] - (nvisnodes - 1)/2 * dx)
  end

  # Then, convert coordinate to global index
  indices = coordinate2index(lower_left_coordinate, resolution)

  return indices
end


# Find 2D array index for a 2-tuple of normalized, cell-centered coordinates (i.e., in [-1,1])
function coordinate2index(coordinate, resolution::Integer)
  # Calculate 1D normalized coordinates
  dx = 2/resolution
  mesh_coordinates = collect(range(-1 + dx/2, 1 - dx/2, length=resolution))

  # Find index
  id_x = searchsortedfirst.(Ref(mesh_coordinates), coordinate[1, :], lt=(x,y)->x .< y .- dx/2)
  id_y = searchsortedfirst.(Ref(mesh_coordinates), coordinate[2, :], lt=(x,y)->x .< y .- dx/2)
  return transpose(hcat(id_x, id_y))
end


# Determine filename for PVD file based on common name
function get_pvd_filename(filenames::AbstractArray)
  filenames = getindex.(splitdir.(filenames), 2)
  bases = getindex.(splitext.(filenames), 1)
  pvd_filename = longest_common_prefix(bases)
  return pvd_filename
end


# Determine longest common prefix
function longest_common_prefix(strings::AbstractArray)
  # Return early if array is empty
  if isempty(strings)
    return ""
  end

  # Count length of common prefix, by ensuring that all strings are long enough
  # and then comparing the next character
  len = 0
  while all(length.(strings) .> len) && all(getindex.(strings, len+1) .== strings[1][len+1])
    len +=1
  end

  return strings[1][1:len]
end


# Convert coordinates and level information to a list of points and VTK cells
function calc_vtk_points_cells(coordinates::AbstractMatrix{Float64},
                               levels::AbstractVector{Int},
                               center_level_0::AbstractVector{Float64},
                               length_level_0::Float64,
                               n_visnodes::Int=1)
  @assert ndim == 2 "Algorithm currently only works in 2D"

  # Create point locator
  pl = PointLocator(center_level_0, length_level_0, 1e-12)

  # Create arrays for points and cells
  n_elements = length(levels)
  points = Vector{Point}()
  vtk_cells = Vector{MeshCell}(undef, n_elements * n_visnodes^ndim)
  point_ids = Vector{Int}(undef, 2^ndim)

  # Reshape cell array for easy-peasy access
  reshaped = reshape(vtk_cells, n_visnodes, n_visnodes, n_elements)

  # Create VTK cell for each Trixi element
  for element_id in 1:n_elements
    # Extract cell values
    cell_x = coordinates[1, element_id]
    cell_y = coordinates[2, element_id]
    cell_dx = length_level_0 / 2^levels[element_id]

    # Adapt to visualization nodes for easy-to-understand loops
    dx = cell_dx / n_visnodes
    x_lowerleft = cell_x - cell_dx/2 - dx/2
    y_lowerleft = cell_y - cell_dx/2 - dx/2

    # Create cell for each visualization node
    for j = 1:n_visnodes
      for i = 1:n_visnodes
        # Determine x and y
        x = x_lowerleft + i * dx
        y = y_lowerleft + j * dx

        # Get point id for each vertex
        point_ids[1] = insert!(pl, points, x - dx/2, y - dx/2)
        point_ids[2] = insert!(pl, points, x + dx/2, y - dx/2)
        point_ids[3] = insert!(pl, points, x - dx/2, y + dx/2)
        point_ids[4] = insert!(pl, points, x + dx/2, y + dx/2)

        # Add cell
        reshaped[i, j, element_id] = MeshCell(VTKCellTypes.VTK_PIXEL, copy(point_ids)) 
      end
    end
  end

  # Convert array-of-points to two-dimensional array
  vtk_points = Matrix{Float64}(undef, ndim, length(points))
  for point_id in 1:length(points)
    vtk_points[1, point_id] = points[point_id].x
    vtk_points[2, point_id] = points[point_id].y
  end

  return vtk_points, vtk_cells
end


# Check if file is a data file
function is_solution_restart_file(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # If attribute "mesh_file" exists, this must be a data file
    return exists(attrs(file), "mesh_file")
  end
end


# Use data file to extract mesh filename from attributes
function extract_mesh_filename(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract filename relative to data file
    mesh_file = read(attrs(file)["mesh_file"])

    return joinpath(dirname(filename), mesh_file)
  end
end


# Read in mesh file and return relevant data
function read_meshfile(filename::String)
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


# Read in data file and return all relevant information
function read_datafile(filename::String)
  # Open file for reading
  h5open(filename, "r") do file
    # Extract basic information
    N = read(attrs(file)["N"])
    n_elements = read(attrs(file)["n_elements"])
    n_variables = read(attrs(file)["n_vars"])
    time = read(attrs(file)["time"])

    # Extract labels for legend
    labels = Array{String}(undef, 1, n_variables)
    for v = 1:n_variables
      labels[1, v] = read(attrs(file["variables_$v"])["name"])
    end

    # Extract data arrays
    n_nodes = N + 1
    data = Array{Float64}(undef, n_nodes, n_nodes, n_elements, n_variables)
    for v = 1:n_variables
      vardata = read(file["variables_$v"])
      @views data[:, :, :, v][:] .= vardata
    end

    # Extract element variable arrays
    element_variables = Dict{String, Union{Vector{Float64}, Vector{Int}}}()
    index = 1
    while exists(file, "element_variables_$index")
      varname = read(attrs(file["element_variables_$index"])["name"])
      element_variables[varname] = read(file["element_variables_$index"])
      index +=1
    end

    return labels, data, n_elements, n_nodes, element_variables, time
  end
end


# Parse command line arguments and return result
function parse_commandline_arguments(args=ARGS)
  # If anything is changed here, it should also be checked at the beginning of run()
  # FIXME: Refactor the code to avoid this redundancy
  s = ArgParseSettings()
  s.autofix_names = true
  @add_arg_table! s begin
    "filename"
      help = "Name of Trixi solution/restart/mesh file to convert to a .vti file."
      arg_type = String
      required = true
      nargs = '+'
    "--verbose", "-v"
      help = "Enable verbose output to avoid despair over long plot times 😉"
      action = :store_true
    "--hide-progress"
      help = "Hide progress bar (will be hidden automatically if `--verbose` is given)"
      action = :store_true
    "--pvd"
      help = ("Use this filename to store PVD file (instead of auto-detecting name). Note that " *
              "only the name will be used (directory and file extension are ignored).")
      arg_type = String
    "--output-directory", "-o"
      help = "Output directory where generated images are stored"
      arg_type = String
      default = "."
    "--nvisnodes"
      help = ("Number of visualization nodes per element "
              * "(default: twice the number of DG nodes). "
              * "A value of zero uses the number of nodes in the DG elements.")
      arg_type = Int
      default = nothing
  end

  return parse_args(args, s)
end


####################################################################################################
# From auxiliary/auxiliary.jl
####################################################################################################
# Allow an expression to be terminated gracefully by Ctrl-c.
#
# On Unix-like operating systems, gracefully handle user interrupts (SIGINT), also known as
# Ctrl-c, while evaluation expression `ex`.
macro interruptable(ex)
  @static Sys.isunix() && quote
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 0)

    try
      # Try to run code
      $(esc(ex))
    catch e
      # Only catch interrupt exceptions and end with a nice message
      isa(e, InterruptException) || rethrow(e)
      println(stderr, "\nExecution interrupted by user (Ctrl-c)")
    end

    # Disable interrupt handling again
    ccall(:jl_exit_on_sigint, Cvoid, (Cint,), 1)
  end
end



####################################################################################################
# From mesh/trees.jl
####################################################################################################
# Auxiliary methods for often-required calculations
# Number of potential child cells
n_children_per_cell() = n_children_per_cell(ndim)
n_children_per_cell(dims::Integer) = 2^dims

end # module Trixi2Vti
