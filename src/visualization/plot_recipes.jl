# Convenience type to allow dispatch on ODESolution objects that were created by Trixi
#
# Note: This is an experimental feature and may be changed in future releases without notice.
const TrixiODESolution = ODESolution{T, N, uType, uType2, DType, tType, rateType, P} where
    {T, N, uType, uType2, DType, tType, rateType, P<:ODEProblem{uType_, tType_, isinplace, P_} where
     {uType_, tType_, isinplace, P_<:AbstractSemidiscretization}}


"""
    PlotData2D

Holds all relevant data for creating 2D plots of multiple solution variables and to visualize the
mesh.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct PlotData2D{Coordinates, Data, VariableNames, Vertices}
  x::Coordinates
  y::Coordinates
  data::Data
  variable_names::VariableNames
  mesh_vertices_x::Vertices
  mesh_vertices_y::Vertices
end


"""
    PlotData2D(u, semi;
               solution_variables=cons2cons,
               grid_lines=true, max_supported_level=11, nvisnodes=nothing,
               slice_axis=:z, slice_axis_intercept=0)

Create a new `PlotData2D` object that can be used for visualizing 2D/3D DGSEM solution data array
`u` with `Plots.jl`. All relevant geometrical information is extracted from the semidiscretization
`semi`. By default, the conservative variables from the solution are used for plotting. This can be
changed by passing an appropriate conversion function to `solution_variables`.

If `grid_lines` is `true`, also extract grid vertices for visualizing the mesh. The output
resolution is indirectly set via `max_supported_level`: all data is interpolated to
`2^max_supported_level` uniformly distributed points in each spatial direction, also setting the
maximum allowed refinement level in the solution. `nvisnodes` specifies the number of visualization
nodes to be used. If it is `nothing`, twice the number of solution DG nodes are used for
visualization, and if set to `0`, exactly the number of nodes in the DG elements are used.

When visualizing data from a three-dimensional simulation, a 2D slice is extracted for plotting.
`slice_axis` specifies the axis orthogonal to the slice plane and may be `:x`, `:y`, or `:z`. The
point on the slice axis where it intersects with the slice plane is given in `slice_axis_intercept`.
Both of these values are ignored when visualizing 2D data.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.

# Examples
```julia
julia> using Trixi, Plots

julia> trixi_include(default_example())
[...]

julia> pd = PlotData2D(sol)
PlotData2D(...)

julia> plot(pd) # To plot all available variables

julia> plot(pd["scalar"]) # To plot only a single variable

julia> plot!(getmesh(pd)) # To add grid lines to the plot
"""
function PlotData2D(u, semi;
                    solution_variables=cons2cons,
                    grid_lines=true, max_supported_level=11, nvisnodes=nothing,
                    slice_axis=:z, slice_axis_intercept=0)
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

"""
    PlotData2D(u_ode::AbstractVector, semi; kwargs...)

Create a `PlotData2D` object from a one-dimensional ODE solution `u_ode` and the semidiscretization
`semi`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData2D(u_ode::AbstractVector, semi; kwargs...) = PlotData2D(wrap_array(u_ode, semi), semi; kwargs...)

"""
    PlotData2D(sol::TrixiODESolution; kwargs...)

Create a `PlotData2D` object from an `ODESolution` created by Trixi.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData2D(sol::TrixiODESolution; kwargs...) = PlotData2D(sol.u[end], sol.prob.p; kwargs...)

# Auxiliary data structure for visualizing a single variable
#
# Note: This is an experimental feature and may be changed in future releases without notice.
struct PlotDataSeries2D{PD<:PlotData2D}
  plot_data::PD
  variable_name::String
  variable_id::Int
end

# Show only a truncated output for convenience (the full data does not make sense)
Base.show(io::IO, ::PlotDataSeries2D) = print(io, "PlotDataSeries2D(...)")

"""
    Base.getindex(pd::PlotData2D, variable_name)

Extract a single variable `variable_name` from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function Base.getindex(pd::PlotData2D, variable_name)
  variable_id = findfirst(isequal(variable_name), pd.variable_names)

  if isnothing(variable_id)
    throw(KeyError(variable_name))
  end

  return PlotDataSeries2D(pd, variable_name, variable_id)
end

# Define additional methods for convenience
Base.firstindex(pd::PlotData2D) = first(pd.variable_names)
Base.lastindex(pd::PlotData2D) = last(pd.variable_names)
Base.length(pd::PlotData2D) = length(pd.variable_names)
Base.size(pd::PlotData2D) = (length(pd),)
Base.keys(pd::PlotData2D) = tuple(pd.variable_names...)
Base.eltype(pd::PlotData2D) = Pair{String, PlotDataSeries2D}
function Base.iterate(pd::PlotData2D, state=1)
  if state > length(pd)
    return nothing
  else
    return (pd.variable_names[state] => pd[pd.variable_names[state]], state + 1)
  end
end

# Show only a truncated output for convenience (the full data does not make sense)
Base.show(io::IO, pd::PlotData2D) = print(io, "PlotData2D(...)")

"""
    getmesh(pd::PlotData2D)

Extract grid lines from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
getmesh(pd::PlotData2D) = PlotDataSeries2D(pd, "mesh", 0)


# Visualize a single variable in a 2D plot (default: heatmap)
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pds::PlotDataSeries2D)
  pd = pds.plot_data
  variable_name = pds.variable_name
  variable_id = pds.variable_id


  xlims --> (pd.x[begin], pd.x[end])
  ylims --> (pd.y[begin], pd.y[end])
  aspect_ratio --> :equal
  legend -->  :none

  # Add data series
  if pds.variable_name == "mesh"
    @series begin
      seriestype := :path
      linecolor := :black
      linewidth := 1
      pd.mesh_vertices_x, pd.mesh_vertices_y
    end
  else
    title --> pds.variable_name
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


# Plot all available variables at once for convenience
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pd::PlotData2D)
  # Create layout that is as square as possible, with a preference for more columns than rows if not
  cols = ceil(Int, sqrt(length(pd.variable_names)))
  rows = ceil(Int, length(pd.variable_names)/cols)
  layout := (rows, cols)

  # Plot all existing variables
  for (i, variable_name) in enumerate(pd.variable_names)
    @series begin
      subplot := i
      pd[variable_name]
    end
  end

  # Fill remaining subplots with empty plot
  for i in (length(pd.variable_names)+1):(rows*cols)
    @series begin
      subplot := i
      axis := false
      ticks := false
      legend := false
      [], []
    end
  end
end


# Create a PlotData2D plot directly from an ODESolution for convenience
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(sol::TrixiODESolution;
                   solution_variables=cons2cons,
                   grid_lines=true, max_supported_level=11, nvisnodes=nothing, slice_axis=:z,
                   slice_axis_intercept=0)
  return PlotData2D(sol;
                    solution_variables=solution_variables,
                    grid_lines=grid_lines, max_supported_level=max_supported_level, nvisnodes=nvisnodes, slice_axis=slice_axis,
                    slice_axis_intercept=slice_axis_intercept)
end

