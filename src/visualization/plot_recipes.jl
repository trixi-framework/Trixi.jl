# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


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

# These methods are used internally to set the default value of the solution variables:
# - If a `cons2prim` for the given `equations` exists, use it
# - Otherwise, use `cons2cons`, which is defined for all systems of equations
digest_solution_variables(equations, solution_variables) = solution_variables
function digest_solution_variables(equations, solution_variables::Nothing)
  if hasmethod(cons2prim, Tuple{AbstractVector, typeof(equations)})
    return cons2prim
  else
    return cons2cons
  end
end

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
  orientation_x::Int
  orientation_y::Int
end


"""
    PlotData2D(u, semi [or mesh, equations, solver, cache];
               solution_variables=nothing,
               grid_lines=true, max_supported_level=11, nvisnodes=nothing,
               slice=:xy, point=(0.0, 0.0, 0.0))

Create a new `PlotData2D` object that can be used for visualizing 2D/3D DGSEM solution data array
`u` with `Plots.jl`. All relevant geometrical information is extracted from the semidiscretization
`semi`. By default, the primitive variables (if existent) or the conservative variables (otherwise)
from the solution are used for plotting. This can be changed by passing an appropriate conversion
function to `solution_variables`.

If `grid_lines` is `true`, also extract grid vertices for visualizing the mesh. The output
resolution is indirectly set via `max_supported_level`: all data is interpolated to
`2^max_supported_level` uniformly distributed points in each spatial direction, also setting the
maximum allowed refinement level in the solution. `nvisnodes` specifies the number of visualization
nodes to be used. If it is `nothing`, twice the number of solution DG nodes are used for
visualization, and if set to `0`, exactly the number of nodes in the DG elements are used.

When visualizing data from a three-dimensional simulation, a 2D slice is extracted for plotting.
`slice` specifies the plane that is being sliced and may be `:xy`, `:xz`, or `:yz`.
The slice position is specified by a `point` that lies on it, which defaults to `(0.0, 0.0, 0.0)`.
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

PlotData2D(u_ode, semi; kwargs...) = PlotData2D(wrap_array_native(u_ode, semi),
                                                mesh_equations_solver_cache(semi)...;
                                                kwargs...)

function PlotData2D(u, mesh::TreeMesh, equations, solver, cache;
                    solution_variables=nothing,
                    grid_lines=true, max_supported_level=11, nvisnodes=nothing,
                    slice=:xy, point=(0.0, 0.0, 0.0))
  @assert ndims(mesh) in (2, 3) "unsupported number of dimensions $ndims (must be 2 or 3)"
  solution_variables_ = digest_solution_variables(equations, solution_variables)

  # Extract mesh info
  center_level_0 = mesh.tree.center_level_0
  length_level_0 = mesh.tree.length_level_0
  leaf_cell_ids = leaf_cells(mesh.tree)
  coordinates = mesh.tree.coordinates[:, leaf_cell_ids]
  levels = mesh.tree.levels[leaf_cell_ids]

  unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations, solver, cache)
  x, y, data, mesh_vertices_x, mesh_vertices_y = get_data_2d(center_level_0, length_level_0,
                                                             leaf_cell_ids, coordinates, levels,
                                                             ndims(mesh), unstructured_data,
                                                             nnodes(solver), grid_lines,
                                                             max_supported_level, nvisnodes,
                                                             slice, point)
  variable_names = SVector(varnames(solution_variables_, equations))

  orientation_x, orientation_y = _get_orientations(mesh, slice)

  return PlotData2D(x, y, data, variable_names, mesh_vertices_x, mesh_vertices_y,
                    orientation_x, orientation_y)
end


function PlotData2D(u, mesh::Union{CurvedMesh,UnstructuredQuadMesh}, equations, solver, cache;
                    solution_variables=nothing, grid_lines=true, kwargs...)
  @unpack node_coordinates = cache.elements

  @assert ndims(mesh) == 2 "unsupported number of dimensions $ndims (must be 2)"
  solution_variables_ = digest_solution_variables(equations, solution_variables)

  unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations, solver, cache)

  x = vec(view(node_coordinates, 1, ..))
  y = vec(view(node_coordinates, 2, ..))

  data = [vec(unstructured_data[.., v]) for v in eachvariable(equations)]

  if grid_lines
    mesh_vertices_x, mesh_vertices_y = calc_vertices(node_coordinates, mesh)
  else
    mesh_vertices_x = Matrix{Float64}(undef, 0, 0)
    mesh_vertices_y = Matrix{Float64}(undef, 0, 0)
  end

  variable_names = SVector(varnames(solution_variables_, equations))

  orientation_x, orientation_y = _get_orientations(mesh, nothing)

  return PlotData2D(x, y, data, variable_names, mesh_vertices_x, mesh_vertices_y,
                    orientation_x, orientation_y)
end


"""
    PlotData2D(sol; kwargs...)

Create a `PlotData2D` object from a solution object created by either `OrdinaryDiffEq.solve!` (which
returns a `DiffEqBase.ODESolution`) or Trixi's own `solve!` (which returns a
`TimeIntegratorSolution`).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData2D(sol::TrixiODESolution; kwargs...) = PlotData2D(sol.u[end], sol.prob.p; kwargs...)

# Convert `slice` to orientations (1 -> `x`, 2 -> `y`, 3 -> `z`) for the two axes in a 2D plot
function _get_orientations(mesh, slice)
  if ndims(mesh) == 2 || (ndims(mesh) == 3 && slice === :xy)
    orientation_x = 1
    orientation_y = 2
  elseif ndims(mesh) == 3 && slice === :xz
    orientation_x = 1
    orientation_y = 3
  elseif ndims(mesh) == 3 && slice === :yz
    orientation_x = 2
    orientation_y = 3
  else
    orientation_x = 0
    orientation_y = 0
  end
  return orientation_x, orientation_y
end

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

# Convert `orientation` into a guide label (see also `_get_orientations`)
function _get_guide(orientation::Integer)
  if orientation == 1
    return "\$x\$"
  elseif orientation == 2
    return "\$y\$"
  elseif orientation == 3
    return "\$z\$"
  else
    return ""
  end
end

# Visualize a single variable in a 2D plot (default: heatmap)
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pds::PlotDataSeries2D)
  @unpack plot_data, variable_id = pds
  @unpack x, y, data, variable_names, orientation_x, orientation_y = plot_data

  # Set geometric properties
  xlims --> (x[begin], x[end])
  ylims --> (y[begin], y[end])
  aspect_ratio --> :equal

  # Set annotation properties
  legend -->  :none
  title --> variable_names[variable_id]
  colorbar --> :true
  xguide --> _get_guide(orientation_x)
  yguide --> _get_guide(orientation_y)

  # Set series properties
  seriestype --> :heatmap

  # Return data for plotting
  x, y, data[variable_id]
end


# Visualize a single variable in a 2D plot. Only works for `scatter` right now.
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pds::PlotDataSeries2D{<:PlotData2D{<:Any, <:AbstractVector{<:AbstractVector}}})
  @unpack plot_data, variable_id = pds
  @unpack x, y, data, variable_names, orientation_x, orientation_y = plot_data

  # Set geometric properties
  xlims --> (minimum(x), maximum(x))
  ylims --> (minimum(y), maximum(y))
  aspect_ratio --> :equal

  # Set annotation properties
  legend -->  :none
  title --> variable_names[variable_id]
  colorbar --> :true
  xguide --> _get_guide(orientation_x)
  yguide --> _get_guide(orientation_y)

  # Set series properties
  seriestype --> :scatter
  markerstrokewidth --> 0

  marker_z --> data[variable_id]

  # Return data for plotting
  x, y
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


# Visualize the mesh in a 2D plot
#
# Note: This is an experimental feature and may be changed in future releases without notice.
@recipe function f(pm::PlotMesh2D{<:PlotData2D{<:Any, <:AbstractVector{<:AbstractVector}}})
  @unpack plot_data = pm
  @unpack x, y, mesh_vertices_x, mesh_vertices_y = plot_data

  # Set geometric and annotation properties
  xlims --> (minimum(x), maximum(x))
  ylims --> (minimum(y), maximum(y))
  aspect_ratio --> :equal
  legend -->  :none

  # Set series properties
  seriestype --> :path
  linecolor --> :black
  linewidth --> 1

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
  orientation_x::Integer
end

"""
    PlotData1D(u, semi [or mesh, equations, solver, cache];
               solution_variables=nothing, nvisnodes=nothing)

Create a new `PlotData1D` object that can be used for visualizing 1D DGSEM solution data array
`u` with `Plots.jl`. All relevant geometrical information is extracted from the semidiscretization
`semi`. By default, the primitive variables (if existent) or the conservative variables (otherwise)
from the solution are used for plotting. This can be changed by passing an appropriate conversion
function to `solution_variables`.

`nvisnodes` specifies the number of visualization nodes to be used. If it is `nothing`,
twice the number of solution DG nodes are used for visualization, and if set to `0`,
exactly the number of nodes in the DG elements are used.

When visualizing data from a two-dimensional simulation, a 1D slice is extracted for plotting.
`slice` specifies the axis along which the slice is extracted and may be `:x`, or `:y`.
The slice position is specified by a `point` that lies on it, which defaults to `(0.0, 0.0)`.
Both of these values are ignored when visualizing 1D data.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData1D(u_ode, semi; kwargs...) = PlotData1D(wrap_array_native(u_ode, semi),
                                                mesh_equations_solver_cache(semi)...;
                                                kwargs...)

function PlotData1D(u, mesh, equations, solver, cache;
                    solution_variables=nothing, nvisnodes=nothing,
                    slice=:x, point=(0.0, 0.0, 0.0))

  solution_variables_ = digest_solution_variables(equations, solution_variables)
  variable_names = SVector(varnames(solution_variables_, equations))

  original_nodes = cache.elements.node_coordinates
  unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations, solver, cache)

  if ndims(mesh) == 1
    x, data, mesh_vertices_x = get_data_1d(original_nodes, unstructured_data, nvisnodes)
    orientation_x = 1
  elseif ndims(mesh) == 2
    x, data, mesh_vertices_x = unstructured_2d_to_1d(original_nodes, unstructured_data, nvisnodes, slice, point)
    orientation_x = 0
  else # ndims(mesh) == 3
    x, data, mesh_vertices_x = unstructured_3d_to_1d(original_nodes, unstructured_data, nvisnodes, slice, point)
    orientation_x = 0
  end

  return PlotData1D(x, data, variable_names, mesh_vertices_x,
                    orientation_x)
end


"""
    PlotData1D(sol; kwargs...)

Create a `PlotData1D` object from a solution object created by either `OrdinaryDiffEq.solve!` (which
returns a `DiffEqBase.ODESolution`) or Trixi's own `solve!` (which returns a
`TimeIntegratorSolution`).
!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData1D(sol::TrixiODESolution; kwargs...) = PlotData1D(sol.u[end], sol.prob.p; kwargs...)

function PlotData1D(time_series_callback::TimeSeriesCallback, point_id::Integer)
  @unpack time, variable_names, point_data = time_series_callback

  n_solution_variables = length(variable_names)
  data = Matrix{Float64}(undef, length(time), n_solution_variables)
  reshaped = reshape(point_data[point_id], n_solution_variables, length(time))
  for v in 1:n_solution_variables
    @views data[:, v] = reshaped[v, :]
  end

  mesh_vertices_x = Vector{Float64}(undef, 0)

  return PlotData1D(time, data, SVector(variable_names), mesh_vertices_x, 0)
end

function PlotData1D(cb::DiscreteCallback{<:Any, <:TimeSeriesCallback}, point_id::Integer)
  return PlotData1D(cb.affect!, point_id)
end

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
  @unpack x, data, variable_names, orientation_x = plot_data

  # Set geometric properties
  xlims --> (x[begin], x[end])

  # Set annotation properties
  legend --> :none
  title --> variable_names[variable_id]
  xguide --> _get_guide(orientation_x)

  # Return data for plotting
  x, data[:, variable_id]
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
@recipe function f(sol::TrixiODESolution)
  # Redirect everything to the recipe below
  return sol.u[end], sol.prob.p
end

# Note: If you change the defaults values here, you need to also change them in the PlotData1D or PlotData2D
#       constructor.
@recipe function f(u, semi::AbstractSemidiscretization;
                   solution_variables=nothing,
                   grid_lines=true, max_supported_level=11, nvisnodes=nothing, slice=:xy,
                   point=(0.0, 0.0, 0.0))
  # Create a PlotData1D or PlotData2D object depending on the dimension.
  if ndims(semi) == 1
    return PlotData1D(u, semi; solution_variables, nvisnodes)
  else
    return PlotData2D(u, semi;
                      solution_variables, grid_lines, max_supported_level,
                      nvisnodes, slice, point)
  end
end


@recipe function f(cb::DiscreteCallback{<:Any, <:TimeSeriesCallback}, point_id::Integer)
  return cb.affect!, point_id
end

@recipe function f(time_series_callback::TimeSeriesCallback, point_id::Integer)
  return PlotData1D(time_series_callback, point_id)
end


end # @muladd
