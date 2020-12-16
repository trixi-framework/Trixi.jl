struct PlotData2D{Coordinates, Data, VariableNames, Vertices}
  x::Coordinates
  y::Coordinates
  data::Data
  variable_names::VariableNames
  mesh_vertices_x::Vertices
  mesh_vertices_y::Vertices
end

# Convenience type to allow dispatch on ODESolution objects that were created by Trixi
const TrixiODESolution = ODESolution{T, N, uType, uType2, DType, tType, rateType, P} where 
    {T, N, uType, uType2, DType, tType, rateType, P<:ODEProblem{uType_, tType_, isinplace, P_} where 
     {uType_, tType_, isinplace, P_<:AbstractSemidiscretization}}

function PlotData2D(u, semi;
                    grid_lines=true, max_supported_level=11, nvisnodes=nothing,
                    slice_axis=:z, slice_axis_intercept=0,
                    solution_variables=cons2cons)
  mesh, equations, solver, _ = mesh_equations_solver_cache(semi)
  @assert ndims(mesh) in (2, 3) "unsupported number of dimensions $ndims (must be 2 or 3)"

  # Extract mesh info
  center_level_0 = mesh.tree.center_level_0
  length_level_0 = mesh.tree.length_level_0
  leaf_cell_ids = leaf_cells(mesh.tree)
  coordinates = mesh.tree.coordinates[:, leaf_cell_ids]
  levels = mesh.tree.levels[leaf_cell_ids]

  unstructured_data = get_unstructured_data(u, semi, solution_variables)
  x, y, data, mesh_vertices_x, mesh_vertices_y = get_data_2d(center_level_0, length_level_0,
                                                             leaf_cell_ids, coordinates, levels,
                                                             ndims(mesh), unstructured_data,
                                                             nnodes(solver), grid_lines,
                                                             max_supported_level, nvisnodes,
                                                             slice_axis, slice_axis_intercept)
  variable_names = SVector(varnames(solution_variables, equations))

  return PlotData2D(x, y, data, variable_names, mesh_vertices_x, mesh_vertices_y)
end

PlotData2D(u_ode::AbstractVector, semi; kwargs...) = PlotData2D(wrap_array(u_ode, semi), semi; kwargs...)
PlotData2D(sol::TrixiODESolution; kwargs...) = PlotData2D(sol.u[end], sol.prob.p; kwargs...)


struct PlotDataSeries2D{PD<:PlotData2D}
  plot_data::PD
  variable::String
end

Base.getindex(pd::PlotData2D, variable) = PlotDataSeries2D(pd, variable)

@recipe function f(pds::PlotDataSeries2D)
  pd = pds.plot_data
  variable = pds.variable

  if variable != "mesh"
    variable_id = findfirst(isequal(variable), pd.variable_names)
    if isnothing(variable_id)
      error("variable '$variable' was not found in data set (existing: $(join(pd.variable_names, ", ")))")
    end
  end
  
  xlims --> (pd.x[begin], pd.x[end])
  ylims --> (pd.y[begin], pd.y[end])
  aspect_ratio --> :equal
  legend -->  :none

  # Add data series
  if pds.variable == "mesh"
    @series begin
      seriestype := :path
      linecolor := :black
      linewidth := 1
      pd.mesh_vertices_x, pd.mesh_vertices_y
    end
  else
    title --> pds.variable
    colorbar --> :true

    @series begin
      seriestype := :heatmap
      fill --> true
      linewidth --> 0
      pd.x, pd.y, transpose(view(pd.data, :, :, variable_id))
    end
  end

  ()
end

# Plot all available variables in a grid
@recipe function f(pd::PlotData2D)
  # Create layout that is as square as possible, with a preference for more columns than rows if not
  cols = ceil(Int, sqrt(length(pd.variable_names)))
  rows = ceil(Int, length(pd.variable_names)/cols)
  layout := (rows, cols)

  for (i, variable) in enumerate(pd.variable_names)
    @series begin
      subplot := i
      pd[variable]
    end
  end
end

# Create a PlotData2D plot from a solution for convenience
@recipe f(::Type{TrixiODESolution}, sol::TrixiODESolution) = PlotData2D(sol)


function get_data_2d(center_level_0, length_level_0, leaf_cells, coordinates, levels, ndims,
                     unstructured_data, n_nodes,
                     grid_lines=false, max_supported_level=11, nvisnodes=nothing,
                     slice_axis=:z, slice_axis_intercept=0)
  # Determine resolution for data interpolation
  max_level = maximum(levels)
  if max_level > max_supported_level
    error("Maximum refinement level in data file $max_level is higher than " *
          "maximum supported level $max_supported_level")
  end
  max_available_nodes_per_finest_element = 2^(max_supported_level - max_level)
  if nvisnodes === nothing
    max_nvisnodes = 2 * n_nodes
  elseif nvisnodes == 0
    max_nvisnodes = n_nodes
  else
    max_nvisnodes = nvisnodes
  end
  nvisnodes_at_max_level = min(max_available_nodes_per_finest_element, max_nvisnodes)
  resolution = nvisnodes_at_max_level * 2^max_level
  nvisnodes_per_level = [2^(max_level - level)*nvisnodes_at_max_level for level in 0:max_level]
  # nvisnodes_per_level is an array (accessed by "level + 1" to accommodate
  # level-0-cell) that contains the number of visualization nodes for any
  # refinement level to visualize on an equidistant grid

  if ndims == 3
    (unstructured_data, coordinates, levels,
        center_level_0) = unstructured_2d_to_3d(unstructured_data,
        coordinates, levels, length_level_0, center_level_0, slice_axis,
        slice_axis_intercept)
  end

  # Normalize element coordinates: move center to (0, 0) and domain size to [-1, 1]²
  n_elements = length(levels)
  normalized_coordinates = similar(coordinates)
  for element_id in 1:n_elements
    @views normalized_coordinates[:, element_id] .= (
          (coordinates[:, element_id] .- center_level_0) ./ (length_level_0 / 2 ))
  end

  # Interpolate unstructured DG data to structured data
  (structured_data =
      unstructured2structured(unstructured_data, normalized_coordinates,
                              levels, resolution, nvisnodes_per_level))

  # Interpolate cell-centered values to node-centered values
  node_centered_data = cell2node(structured_data)

  # Determine axis coordinates for contour plot
  xs = collect(range(-1, 1, length=resolution+1)) .* length_level_0/2 .+ center_level_0[1]
  ys = collect(range(-1, 1, length=resolution+1)) .* length_level_0/2 .+ center_level_0[2]

  # Determine element vertices to plot grid lines
  if grid_lines
    mesh_vertices_x, mesh_vertices_y = calc_vertices(coordinates, levels, length_level_0)
  else
    mesh_vertices_x = mesh_vertices_y = nothing
  end

  return xs, ys, node_centered_data, mesh_vertices_x, mesh_vertices_y
end

function get_unstructured_data(u, semi, solution_variables)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  if solution_variables === cons2cons
    raw_data = u
    n_vars = size(raw_data, 1)
  else
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    raw_data = Array(reinterpret(eltype(u),
           solution_variables.(reinterpret(SVector{nvariables(equations),eltype(u)}, u),
                      Ref(equations))))
    n_vars = size(raw_data, 1)
  end

  unstructured_data = Array{Float64}(undef,
                                     ntuple((d) -> nnodes(solver), ndims(equations))...,
                                     nelements(solver, cache), n_vars)
  for variable in 1:n_vars
    @views unstructured_data[.., :, variable] .= raw_data[variable, .., :]
  end

  return unstructured_data
end

