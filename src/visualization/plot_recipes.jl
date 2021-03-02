# Convenience type to allow dispatch on solution objects that were created by Trixi
#
# This is a union of a Trixi-specific DiffEqBase.ODESolution and of Trixi's own
# TimeIntegratorSolution.
#
# Note: This is an experimental feature and may be changed in future releases without notice.
const TrixiODESolution = Union{ODESolution{T, N, uType, uType2, DType, tType, rateType, P} where
    {T, N, uType, uType2, DType, tType, rateType, P<:ODEProblem{uType_, tType_, isinplace, P_, F_} where
     {uType_, tType_, isinplace, P_<:AbstractSemidiscretization, F_<:ODEFunction{true, typeof(rhs!)}}}, TimeIntegratorSolution}

# This abstract type is used to derive PlotData types of different dimensions; but still allows to share some functions for them.
abstract type AbstractPlotData end

# Define additional methods for convenience.
# These are defined for AbstractPlotData, so they can be used for all types of plot data.
Base.firstindex(pd::AbstractPlotData) = first(pd.variable_names)
Base.lastindex(pd::AbstractPlotData) = last(pd.variable_names)
Base.length(pd::AbstractPlotData) = length(pd.variable_names)
Base.size(pd::AbstractPlotData) = (length(pd),)
Base.keys(pd::AbstractPlotData) = tuple(pd.variable_names...)

"""
    PlotData2D

Holds all relevant data for creating 2D plots of multiple solution variables and to visualize the
mesh.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct PlotData2D{Coordinates, Data, VariableNames, Vertices} <: AbstractPlotData
  x::Coordinates
  y::Coordinates
  data::Data
  variable_names::VariableNames
  mesh_vertices_x::Vertices
  mesh_vertices_y::Vertices
end


"""
    PlotData2D(u, semi;
               solution_variables=cons2prim,
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
```
"""
function PlotData2D(u, semi;
                    solution_variables=cons2prim,
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
    PlotData2D(sol::Union{DiffEqBase.ODESolution,TimeIntegratorSolution}; kwargs...)

Create a `PlotData2D` object from a solution object created by either `OrdinaryDiffEq.solve!` (which
returns a `DiffEqBase.ODESolution`) or Trixi's own `solve!` (which returns a
`TimeIntegratorSolution`).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData2D(sol::TrixiODESolution; kwargs...) = PlotData2D(sol.u[end], sol.prob.p; kwargs...)

# Auxiliary data structure for visualizing a single variable
#
# Note: This is an experimental feature and may be changed in future releases without notice.
struct PlotDataSeries2D{PD<:PlotData2D}
  plot_data::PD
  variable_id::Int
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pds::PlotDataSeries2D)
  @nospecialize pds # reduce precompilation time

  print(io, "PlotDataSeries2D{", typeof(pds.plot_data), "}(<plot_data::PlotData2D>, ",
        pds.variable_id, ")")
end

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

  return PlotDataSeries2D(pd, variable_id)
end

Base.eltype(pd::PlotData2D) = Pair{String, PlotDataSeries2D}
function Base.iterate(pd::PlotData2D, state=1)
  if state > length(pd)
    return nothing
  else
    return (pd.variable_names[state] => pd[pd.variable_names[state]], state + 1)
  end
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pd::PlotData2D)
  @nospecialize pd # reduce precompilation time

  print(io, "PlotData2D{",
            typeof(pd.x), ",",
            typeof(pd.data), ",",
            typeof(pd.variable_names), ",",
            typeof(pd.mesh_vertices_x),
            "}(<x>, <y>, <data>, <variable_names>, <mesh_vertices_x>, <mesh_vertices_y>)")
end

# Auxiliary data structure for visualizing the mesh
#
# Note: This is an experimental feature and may be changed in future releases without notice.
struct PlotMesh2D{PD<:PlotData2D}
  plot_data::PD
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pm::PlotMesh2D)
  @nospecialize pm # reduce precompilation time

  print(io, "PlotMesh2D{", typeof(pm.plot_data), "}(<plot_data::PlotData2D>)")
end

"""
    getmesh(pd::PlotData2D)

Extract grid lines from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
getmesh(pd::PlotData2D) = PlotMesh2D(pd)


# Visualize a single variable in a 2D plot (default: heatmap)
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pds::PlotDataSeries2D)
  @unpack plot_data, variable_id = pds
  @unpack x, y, data, variable_names = plot_data

  # Set geometric properties
  xlims --> (x[begin], x[end])
  ylims --> (y[begin], y[end])
  aspect_ratio --> :equal

  # Set annotation properties
  legend -->  :none
  title --> variable_names[variable_id]
  colorbar --> :true

  # Set series properties
  seriestype --> :heatmap

  # Return data for plotting
  x, y, data[variable_id]
end


# Visualize the mesh in a 2D plot
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pm::PlotMesh2D)
  @unpack plot_data = pm
  @unpack x, y, mesh_vertices_x, mesh_vertices_y = plot_data

  # Set geometric and annotation properties
  xlims --> (x[begin], x[end])
  ylims --> (y[begin], y[end])
  aspect_ratio --> :equal
  legend -->  :none

  # Set series properties
  seriestype := :path
  linecolor := :black
  linewidth := 1

  # Return data for plotting
  mesh_vertices_x, mesh_vertices_y
end


# Plot all available variables at once for convenience
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pd::PlotData2D)
  # Create layout that is as square as possible, when there are more than 3 subplots.
  # This is done with a preference for more columns than rows if not.

  if length(pd) <= 3
    cols = length(pd)
    rows = 1
  else
    cols = ceil(Int, sqrt(length(pd)))
    rows = ceil(Int, length(pd)/cols)
  end

  layout := (rows, cols)

  # Plot all existing variables
  for (i, (variable_name, series)) in enumerate(pd)
    @series begin
      subplot := i
      series
    end
  end

  # Fill remaining subplots with empty plot
  for i in (length(pd)+1):(rows*cols)
    @series begin
      subplot := i
      axis := false
      ticks := false
      legend := false
      [], []
    end
  end
end



"""
    PlotData1D

Holds all relevant data for creating 1D plots of multiple solution variables and to visualize the
mesh.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct PlotData1D{Coordinates, Data, VariableNames, Vertices} <:AbstractPlotData
  x::Coordinates
  data::Data
  variable_names::VariableNames
  mesh_vertices_x::Vertices
end

"""
    PlotData1D(u, semi)

Create a new `PlotData1D` object that can be used for visualizing 1D DGSEM solution data array
`u` with `Plots.jl`. All relevant geometrical information is extracted from the semidiscretization
`semi`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function PlotData1D(u, semi)

  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @assert ndims(mesh) in (1) "unsupported number of dimensions $ndims (must be 1)"

  x = cache.elements.node_coordinates

  # TODO cons2prim is hardcoded here and needs to be changed later.
  variable_names = SVector(varnames(cons2prim, equations))

  return PlotData1D(vec(x), reshape(u, length(variable_names),:), variable_names, vcat(x[1, 1, :], x[1, end, end]))
end

"""
    PlotData1D(u_ode::AbstractVector, semi)

Create a `PlotData1D` object from a one-dimensional ODE solution `u_ode` and the semidiscretization
`semi`.
!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData1D(u_ode::AbstractVector, semi) = PlotData1D(wrap_array(u_ode, semi), semi)

"""
    PlotData1D(sol::Union{DiffEqBase.ODESolution,TimeIntegratorSolution})

Create a `PlotData1D` object from a solution object created by either `OrdinaryDiffEq.solve!` (which
returns a `DiffEqBase.ODESolution`) or Trixi's own `solve!` (which returns a
`TimeIntegratorSolution`).
!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData1D(sol::TrixiODESolution) = PlotData1D(sol.u[end], sol.prob.p)

# Store multiple PlotData1D objects in one PlotDataSeries1D.
# This is used for multi-variable equations.
struct PlotDataSeries1D{PD<:PlotData1D}
  plot_data::PD
  variable_id::Int
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pds::PlotDataSeries1D)
  print(io, "PlotDataSeries1D{", typeof(pds.plot_data), "}(<plot_data::PlotData1D>, ",
        pds.variable_id, ")")
end

# Extract a single variable from a PlotData1D object.
function Base.getindex(pd::PlotData1D, variable_name)
  variable_id = findfirst(isequal(variable_name), pd.variable_names)

  if isnothing(variable_id)
    throw(KeyError(variable_name))
  end

  return PlotDataSeries1D(pd, variable_id)
end

Base.eltype(pd::PlotData1D) = Pair{String, PlotDataSeries1D}
function Base.iterate(pd::PlotData1D, state=1)
  if state > length(pd)
    return nothing
  else
    return (pd.variable_names[state] => pd[pd.variable_names[state]], state + 1)
  end
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pd::PlotData1D)
  print(io, "PlotData1D{",
            typeof(pd.x), ",",
            typeof(pd.data), ",",
            typeof(pd.variable_names), ",",
            typeof(pd.mesh_vertices_x),
            "}(<x>, <data>, <variable_names>, <mesh_vertices_x>)")
end


# A struct to store all relevant information about the mesh of a 1D equations, which is needed to plot the mesh.
struct PlotMesh1D{PD<:PlotData1D}
  plot_data::PD
end

# Show a PlotMesh1D in a convenient way.
function Base.show(io::IO, pm::PlotMesh1D)
  print(io, "PlotMesh1D{", typeof(pm.plot_data), "}(<plot_data::PlotData1D>)")
end

# Extract the grid lines from a PlotData1D object to create a PlotMesh1D.
getmesh(pd::PlotData1D) = PlotMesh1D(pd)

# Plot a single variable.
@recipe function f(pds::PlotDataSeries1D)
  @unpack plot_data, variable_id = pds
  @unpack x, data, variable_names = plot_data

  # Set geometric properties
  xlims --> (x[begin], x[end])

  # Set annotation properties
  legend -->  :none
  title --> variable_names[variable_id]

  # Return data for plotting
  x, data[variable_id,:]
end

# Plot the mesh as vertical lines from a PlotMesh1D object.
@recipe function f(pm::PlotMesh1D)
  @unpack plot_data = pm
  @unpack x, mesh_vertices_x = plot_data

  # Set geometric and annotation properties
  xlims --> (x[begin], x[end])
  legend -->  :none

  # Set series properties
  seriestype --> :vline
  linecolor --> :grey
  linewidth --> 1

  # Return data for plotting
  mesh_vertices_x
end

# This plots all variables by creating a subplot for each of them.
@recipe function f(pd::PlotData1D)
  # Create layout that is as square as possible, when there are more than 3 subplots.
  # This is done with a preference for more columns than rows if not.
  if length(pd) <= 3
    cols = length(pd)
    rows = 1
  else
    cols = ceil(Int, sqrt(length(pd)))
    rows = ceil(Int, length(pd)/cols)
  end

  layout := (rows, cols)

  # Plot all existing variables
  for (i, (variable_name, series)) in enumerate(pd)
    @series begin
      subplot := i
      series
    end
  end

  # Fill remaining subplots with empty plot
  for i in (length(pd)+1):(rows*cols)
    @series begin
      subplot := i
      axis := false
      ticks := false
      legend := false
      []
    end
  end
end


# Create a plot directly from a TrixiODESolution for convenience
# The plot is created by a PlotData1D or PlotData2D object.
#
# Note: This is an experimental feature and may be changed in future releases without notice.
#
# Note: If you change the defaults values here, you need to also change them in the PlotData1D or PlotData2D
#       constructor.
@recipe function f(sol::TrixiODESolution;
                   solution_variables=cons2prim,
                   grid_lines=true, max_supported_level=11, nvisnodes=nothing, slice_axis=:z,
                   slice_axis_intercept=0)

  mesh, _, _, _ = mesh_equations_solver_cache(sol.prob.p)

  # Create a PlotData1D or PlotData2D object depending on the dimension.
  if ndims(mesh) == 1
    return PlotData1D(sol)
  else
    return PlotData2D(sol;
                      solution_variables=solution_variables,
                      grid_lines=grid_lines, max_supported_level=max_supported_level,
                      nvisnodes=nvisnodes, slice_axis=slice_axis,
                      slice_axis_intercept=slice_axis_intercept)
    end
end
