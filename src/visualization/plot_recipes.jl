struct ContourPlot{Coordinates, Data, Variables, Limits, Vertices}
  x::Coordinates
  y::Coordinates
  data::Data
  variables::Variables
  xlims::Limits
  ylims::Limits
  vertices_x::Vertices
  vertices_y::Vertices
end

function ContourPlot(sol::ODESolution;
                     grid_lines=true, max_supported_level=11, nvisnodes=nothing,
                     slice_axis=:z, slice_axis_intercept=0,
                     solution_variables=Trixi.cons2cons)
  # Extract basic information about Trixi's solution
  semi = sol.prob.p
  u = Trixi.wrap_array(sol.u[end], semi)
  mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
  ndims = Trixi.ndims(mesh)
  @assert ndims in (2, 3) "unsupported number of dimensions $ndims (must be 2 or 3)"

  # Extract mesh info
  center_level_0 = mesh.tree.center_level_0
  length_level_0 = mesh.tree.length_level_0
  leaf_cells = Trixi.leaf_cells(mesh.tree)
  coordinates = mesh.tree.coordinates[:, leaf_cells]
  levels = mesh.tree.levels[leaf_cells]

  unstructured_data = get_unstructured_data(u, semi, solution_variables)
  x, y, data, vertices_x, vertices_y = get_contour_data(center_level_0, length_level_0, leaf_cells,
                                                        coordinates, levels, ndims,
                                                        unstructured_data, Trixi.nnodes(solver),
                                                        grid_lines, max_supported_level, nvisnodes,
                                                        slice_axis, slice_axis_intercept)
  variables = SVector(Trixi.varnames(solution_variables, equations))
  xlims = (-1, 1) .* (length_level_0/2 + center_level_0[1])
  ylims = (-1, 1) .* (length_level_0/2 + center_level_0[2])

  return ContourPlot(x, y, data, variables, xlims, ylims, vertices_x, vertices_y)
end


struct ContourPlotSeries{CP<:ContourPlot}
  contour_plot::CP
  variable::String
end

Base.getindex(cp::ContourPlot, variable) = ContourPlotSeries(cp, variable)

@recipe function f(cps::ContourPlotSeries)
  cp = cps.contour_plot
  variable = cps.variable

  if variable != "mesh"
    variable_id = findfirst(isequal(variable), cp.variables)
    if isnothing(variable_id)
      error("variable '$variable' was not found in data set (existing: $(join(cp.variables, ", ")))")
    end
  end
  
  xlims --> cp.xlims
  ylims --> cp.ylims
  aspect_ratio --> :equal
  legend -->  :none

  # Add data series
  if cps.variable == "mesh"
    @series begin
      seriestype := :path
      linecolor := :black
      linewidth := 1
      cp.vertices_x, cp.vertices_y
    end
  else
    title --> cps.variable
    colorbar --> :true

    @series begin
      seriestype := :heatmap
      fill --> true
      linewidth --> 0
      cp.x, cp.y, transpose(view(cp.data, :, :, variable_id))
    end
  end

  ()
end

@recipe function f(cp::ContourPlot)
  cols = ceil(Int, sqrt(length(cp.variables)))
  rows = ceil(Int, length(cp.variables)/cols)
  layout := (rows, cols)
  for (i, variable) in enumerate(cp.variables)
    @series begin
      subplot := i
      cp[variable]
    end
  end
end


function get_contour_data(center_level_0, length_level_0, leaf_cells, coordinates, levels, ndims,
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
    vertices_x, vertices_y = calc_vertices(coordinates, levels, length_level_0)
  else
    vertices_x = vertices_y = nothing
  end

  return xs, ys, node_centered_data, vertices_x, vertices_y
end

function get_unstructured_data(u, semi, solution_variables)
  mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

  if solution_variables === Trixi.cons2cons
    raw_data = u
    n_vars = size(raw_data, 1)
  else
    # Reinterpret the solution array as an array of conservative variables,
    # compute the solution variables via broadcasting, and reinterpret the
    # result as a plain array of floating point numbers
    raw_data = Array(reinterpret(eltype(u),
           solution_variables.(reinterpret(SVector{Trixi.nvariables(equations),eltype(u)}, u),
                      Ref(equations))))
    n_vars = size(raw_data, 1)
  end

  unstructured_data = Array{Float64}(undef,
                                     ntuple((d) -> Trixi.nnodes(solver), ndims(equations))...,
                                     Trixi.nelements(solver, cache), n_vars)
  for variable in 1:n_vars
    @views unstructured_data[.., :, variable] .= raw_data[variable, .., :]
  end

  return unstructured_data
end

