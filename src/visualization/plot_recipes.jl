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

Base.getindex(pd::PlotData2D, variable_name) = PlotDataSeries2D(pd, variable_name)

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
      pd.x, pd.y, pd.data[variable_id]
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
@recipe f(sol::TrixiODESolution) = PlotData2D(sol)


