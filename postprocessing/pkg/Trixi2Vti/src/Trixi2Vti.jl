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
    # If datafile is a single string, convert it to array
    if !haskey(args, "datafile")
      println(stderr, "error: no datafile was provided")
      return
    end
    if isa(args["datafile"], String)
      args["datafile"] = [args["datafile"]]
    end
    if !haskey(args, "verbose")
      args["verbose"] = false
    end
    if !haskey(args, "hide-progress")
      args["hide-progress"] = false
    end
    if !haskey(args, "save-pvd")
      args["save-pvd"] = "auto"
    end
    if !haskey(args, "separate-celldata")
      args["separate-celldata"] = false
    end
    if !haskey(args, "pvd-filename")
      args["save-pvd"] = nothing
    end
    if !haskey(args, "output_directory")
      args["output_directory"] = "."
    end
    if !haskey(args, "nvisnodes")
      args["nvisnodes"] = nothing
    end
  end

  # Store for convenience
  verbose = args["verbose"]
  hide_progress = args["hide_progress"]
  separate_celldata = true # Cannot be false since we do not interpolate cell data to image data
  datafiles = args["datafile"]

  # If verbose mode is enabled, always hide progress bar
  if verbose
    hide_progress = true
  end

  # Initialize PVD file if desired
  if args["save-pvd"] == "yes" || (args["save-pvd"] == "auto" && length(datafiles) > 1)
    # Determine pvd filename
    if !isnothing(args["pvd-filename"])
      # Use filename if given on command line
      filename = args["pvd-filename"]

      # Strip of directory/extension
      filename, _ = splitext(splitdir(filename)[2])
    else
      filename = get_pvd_filename(datafiles)

      # If filename is empty, it means we were not able to determine an
      # appropriate file thus the user has to supply one
      if filename == ""
        error("could not auto-detect PVD filename (input file names have no common prefix): " *
              "please provide a PVD filename name with `--pvd-filename` " *
              "or disable saving a PVD file with `--save-pvd=no`")
      end
    end

    # Get full filenae
    pvd_filename = joinpath(args["output_directory"], filename)

    # Opening PVD file
    verbose && println("Opening PVD file '$(pvd_filename).pvd'...")
    @timeit "open PVD file" pvd = paraview_collection(pvd_filename)

    # Open separate PVD file for celldata information
    if separate_celldata
      # Get full filename
      pvd_celldata_filename = joinpath(args["output_directory"], filename * "_celldata")

      # Opening PVD file
      verbose && println("Opening PVD file '$(pvd_celldata_filename).pvd'...")
      @timeit "open PVD file" pvd_celldata = paraview_collection(pvd_celldata_filename)
    end

    # Add variable to avoid writing PVD file if only mesh files were converted
    has_data = false

    # Enable saving to PVD
    save_pvd = true
  else
    # Disable saving to PVD
    save_pvd = false
  end

  # Show progress bar if not disabled
  if !hide_progress
    progress = Progress(length(datafiles), 0.5, "Converting .h5 to .vtu...", 40)
  end

  # Iterate over input files
  for (index, datafile) in enumerate(datafiles)
    verbose && println("Processing file $datafile ($(index)/$(length(datafiles)))...")

    # Check if data file exists
    if !isfile(datafile)
      error("data file '$datafile' does not exist")
    end

    # Check if it is a data file at all
    is_datafile = is_solution_restart_file(datafile)

    # If file is solution/restart file, extract mesh file name
    if is_datafile
      # Get mesh file name
      meshfile = extract_mesh_filename(datafile)

      # Check if mesh file exists
      if !isfile(meshfile)
        error("mesh file '$meshfile' does not exist")
      end
    else
      meshfile = datafile
    end

    # Read mesh
    verbose && println("| Reading mesh file...")
    @timeit "read mesh" (center_level_0, length_level_0,
                         leaf_cells, coordinates, levels) = read_meshfile(meshfile)

    if is_datafile
      # Read data only if it is a data file
      verbose && println("| Reading data file...")
      @timeit "read data" (labels, unstructured_data, n_elements, n_nodes,
                           element_variables, time) = read_datafile(datafile)

      # Check if dimensions match
      if length(leaf_cells) != n_elements
        error("number of elements in '$(datafile)' do not match number of leaf cells in " *
              "'$(meshfile)' " *
              "(did you forget to clean your 'out/' directory between different runs?)")
      end

      # Determine resolution for data interpolation
      if args["nvisnodes"] == nothing
        nvisnodes_at_max_level = 2 * n_nodes
      elseif args["nvisnodes"] == 0
        nvisnodes_at_max_level = n_nodes
      else
        nvisnodes_at_max_level = args["nvisnodes"]
      end

      # Determine level-wise resolution
      max_level = maximum(levels)
      resolution = nvisnodes_at_max_level * 2^max_level

      # nvisnodes_per_level is an array (accessed by "level + 1" to accommodate
      # level-0-cell) that contains the number of visualization nodes for any
      # refinement level to visualize on an equidistant grid
      nvisnodes_per_level = [2^(max_level - level)*nvisnodes_at_max_level for level in 0:max_level]
    else
      # If file is a mesh file, do not interpolate data
      n_visnodes = 1
    end

    # Prepare VTK points and cells for celldata file
    if separate_celldata
      @timeit "prepare VTK cells" vtk_celldata_points, vtk_celldata_cells = calc_vtk_points_cells(
          coordinates, levels, center_level_0, length_level_0, 1)
    end

    # Create output directory if it does not exist
    mkpath(args["output_directory"])

    # Determine output file name
    base, _ = splitext(splitdir(datafile)[2])
    vtk_filename = joinpath(args["output_directory"], base)

    # Open VTK file
    verbose && println("| Building VTK grid...")
    if is_datafile
      Nx = Ny = resolution + 1
      dx = dy = length_level_0/resolution
      origin = center_level_0 .- 1/2 * length_level_0
      spacing = [dx, dy]
      @timeit "build VTK grid (node data)" vtk = vtk_grid(vtk_filename, Nx, Ny,
                                                          origin=origin,
                                                          spacing=spacing)

      # Normalize element coordinates: move center to (0, 0) and domain size to [-1, 1]²
      normalized_coordinates = similar(coordinates)
      for element_id in 1:n_elements
        @views normalized_coordinates[:, element_id] .= (
            (coordinates[:, element_id] .- center_level_0) ./ (length_level_0 / 2 ))
      end

      # Interpolate unstructured DG data to structured data
      verbose && println("| Interpolating data...")
      @timeit "interpolate data" (structured_data =
          unstructured2structured(unstructured_data, normalized_coordinates,
                                  levels, resolution, nvisnodes_per_level))
    end

    # Open VTK celldata file
    if separate_celldata
      # Determine output file name
      vtk_celldata_filename = joinpath(args["output_directory"], base * "_celldata")

      # Open VTK file
      @timeit "build VTK grid (cell data)" vtk_celldata = vtk_grid(vtk_celldata_filename,
                                                          vtk_celldata_points,
                                                          vtk_celldata_cells)
    end

    # Add data to file
    verbose && println("| Adding data to VTK file...")
    @timeit "add data to VTK file" begin
      # Add cell/element data to celldata VTK file if it exists, otherwise to regular VTK file
      if separate_celldata
        verbose && println("| | cell_ids...")
        @timeit "cell_ids" vtk_celldata["cell_ids"] = leaf_cells
        verbose && println("| | element_ids...")
        @timeit "element_ids" vtk_celldata["element_ids"] = collect(1:length(leaf_cells))
        verbose && println("| | levels...")
        @timeit "levels" vtk_celldata["levels"] = levels
      else
        verbose && println("| | cell_ids...")
        @timeit "cell_ids" vtk["cell_ids"] = cell2visnode(leaf_cells, n_visnodes)
        verbose && println("| | element_ids...")
        @timeit "element_ids" vtk["element_ids"] = cell2visnode(collect(1:length(leaf_cells)),
                                                                n_visnodes)
        verbose && println("| | levels...")
        @timeit "levels" vtk["levels"] = cell2visnode(levels, n_visnodes)
      end

      # Only add data if it is a data file
      if is_datafile
        # Add solution variables
        for (variable_id, label) in enumerate(labels)
          verbose && println("| | Variable: $label...")
          @timeit label vtk[label] = @views vec(structured_data[:, :, variable_id])
        end

        # Add element variables
        if separate_celldata
          for (label, variable) in element_variables
            verbose && println("| | Element variable: $label...")
            @timeit label vtk_celldata[label] = variable
          end
        else
          for (label, variable) in element_variables
            verbose && println("| | Element variable: $label...")
            @timeit label vtk[label] = cell2visnode(variable, n_visnodes)
          end
        end
      end
    end

    # Save VTK file
    verbose && println("| Saving VTK file '$(vtk_filename).vtu'...")
    @timeit "save VTK file" vtk_save(vtk)

    # Add to PVD file only if it is a datafile
    if save_pvd
      if is_datafile
        verbose && println("| Adding to PVD file...")
        @timeit "add VTK to PVD file" pvd[time] = vtk
        has_data = true
      else
        println("WARNING: file '$(datafile)' will not be added to PVD file since it is a mesh file")
      end
    end

    if separate_celldata
      # Save VTK file
      verbose && println("| Saving VTK celldata file '$(vtk_celldata_filename).vtu'...")
      @timeit "save VTK file" vtk_save(vtk_celldata)

      # Add to PVD file only if it is a datafile
      if save_pvd && is_datafile
        verbose && println("| Adding to PVD file...")
        @timeit "add VTK to PVD file" pvd_celldata[time] = vtk_celldata
        has_data = true
      end
    end

    # Update progress bar
    if !hide_progress
      next!(progress, showvalues=[(:finished, datafile)])
    end
  end

  # Save PVD file only if at least one data file was added
  if save_pvd && has_data
    verbose && println("| Saving PVD file '$(pvd_filename).pvd'...")
    @timeit "save PVD file" vtk_save(pvd)
  end
  if save_pvd && separate_celldata
    verbose && println("| Saving PVD file '$(pvd_celldata_filename).pvd'...")
    @timeit "save PVD file" vtk_save(pvd_celldata)
  end

  verbose && println("| done.\n")
  print_timer()
  println()
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

  #=# Calculate node coordinates for structured locations on reference element=#
  #=max_level = length(nvisnodes_per_level) - 1=#
  #=visnodes_per_level = []=#
  #=for l in 0:max_level=#
  #=  n_nodes_out = nvisnodes_per_level[l + 1]=#
  #=  dx = 2 / n_nodes_out=#
  #=  push!(visnodes_per_level, collect(range(-1 + dx/2, 1 - dx/2, length=n_nodes_out)))=#
  #=end=#

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

  return structured
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


# Convert cell data to visnode data
function cell2visnode(cell_data::Vector, n_visnodes::Int)
  cellsize = n_visnodes^ndim
  visnode_data = Vector{eltype(cell_data)}(undef, length(cell_data) * cellsize)
  for cell_id in 1:length(cell_data)
    for node_id in 1:cellsize
      visnode_data[(cell_id - 1)*cellsize + node_id] = cell_data[cell_id]
    end
  end
  return visnode_data
end


# Determine filename for PVD file based on common name
function get_pvd_filename(datafiles::AbstractArray)
  filenames = getindex.(splitdir.(datafiles), 2)
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


# Interpolate to visualization nodes
function raw2visnodes(data_gl::AbstractArray{Float64}, n_visnodes::Int, variable_id::Int)
  # Extract data shape information
  n_nodes_in, _, n_elements, n_variables = size(data_gl)

  # Get node coordinates for DG locations on reference element
  nodes_in, _ = gauss_lobatto_nodes_weights(n_nodes_in)

  # Calculate Vandermonde matrix
  dx = 2 / n_visnodes
  nodes_out = collect(range(-1 + dx/2, 1 - dx/2, length=n_visnodes))
  vandermonde = polynomial_interpolation_matrix(nodes_in, nodes_out)

  # Create output data structure
  data_vis = Array{Float64}(undef, n_visnodes, n_visnodes, n_elements)

  # Reshape data array for use in interpolate_nodes function
  @views reshaped_data = reshape(data_gl[:, :, :, variable_id], 1, n_nodes_in,
                                 n_nodes_in, n_elements)

  # Interpolate data to visualization nodes
  for element_id in 1:n_elements
    @views data_vis[:, :, element_id] .= reshape(
        interpolate_nodes(reshaped_data[:, :, :, element_id], vandermonde, 1),
        n_visnodes, n_visnodes)
  end

  return data_vis
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


# Check if file is a datafile
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
  @add_arg_table! s begin
    "datafile"
      help = "Name of Trixi solution/restart/mesh file to convert to a .vtu file."
      arg_type = String
      required = true
      nargs = '+'
    "--verbose", "-v"
      help = "Enable verbose output to avoid despair over long plot times 😉"
      action = :store_true
    "--hide-progress"
      help = "Hide progress bar (will be hidden automatically if `--verbose` is given)"
      action = :store_true
    "--separate-celldata", "-s"
      help = ("Save cell data in separate file. This is slightly slower since it requires " *
              "building two sets of VTK grids for each data file. However, it allows to view " *
              "cell data on the original mesh (and not on the visualization nodes).")
      action = :store_true
    "--save-pvd"
      help = ("In addition to a VTK file, write a PVD file that contains time information. " *
              "Possible values are 'yes', 'no', or 'auto'. If set to 'auto', a PVD file is only " *
              "created if multiple files are converted.")
      default = "auto"
      arg_type = String
      range_tester = (x->x in ("yes", "auto", "no"))
    "--pvd-filename"
      help = ("Use this filename to store PVD file (instead of auto-detecting name). Note that " *
              "only the name will be used (directory and extension are ignored).")
      arg_type = String
    "--output-directory", "-o"
      help = "Output directory where generated images are stored"
      dest_name = "output_directory"
      arg_type = String
      default = "."
    "--nvisnodes"
      help = ("Number of visualization nodes per cell "
              * "(default: four times the number of DG nodes). "
              * "A value of zero prevents any interpolation of data.")
      arg_type = Int
      default = nothing
  end

  return parse_args(s)
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
