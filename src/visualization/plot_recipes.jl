# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


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


function PlotData2D(u, mesh::Union{StructuredMesh,UnstructuredMesh2D}, equations, solver, cache;
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


"""
    getmesh(pd::PlotData2D)

Extract grid lines from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
getmesh(pd::AbstractPlotData{2}) = PlotMesh2D(pd)

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
RecipesBase.@recipe function f(pds::PlotDataSeries2D)
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
RecipesBase.@recipe function f(pds::PlotDataSeries2D{<:PlotData2D{<:Any, <:AbstractVector{<:AbstractVector}}})
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
RecipesBase.@recipe function f(pm::PlotMesh2D)
  @unpack plot_data = pm
  @unpack x, y, mesh_vertices_x, mesh_vertices_y = plot_data

  # Set geometric and annotation properties
  xlims --> (x[begin], x[end])
  ylims --> (y[begin], y[end])
  aspect_ratio --> :equal
  legend -->  :none
  grid --> false

  # Set series properties
  seriestype --> :path
  linecolor --> :grey
  linewidth --> 1

  # Return data for plotting
  mesh_vertices_x, mesh_vertices_y
end


# Visualize the mesh in a 2D plot
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(pm::PlotMesh2D{<:PlotData2D{<:Any, <:AbstractVector{<:AbstractVector}}})
  @unpack plot_data = pm
  @unpack x, y, mesh_vertices_x, mesh_vertices_y = plot_data

  # Set geometric and annotation properties
  xlims --> (minimum(x), maximum(x))
  ylims --> (minimum(y), maximum(y))
  aspect_ratio --> :equal
  legend -->  :none
  grid --> false

  # Set series properties
  seriestype --> :path
  linecolor --> :grey
  linewidth --> 1

  # Return data for plotting
  mesh_vertices_x, mesh_vertices_y
end


# Plot all available variables at once for convenience
#
# Note: This is an experimental feature and may be changed in future releases without notice.
RecipesBase.@recipe function f(pd::Union{PlotData2D, UnstructuredPlotData2D})
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
This applies analogously to three-dimensonal simulations, where `slice` may be `xy:`, `:xz`, or `:yz`.

Another way to visualize 2D/3D data is by creating a plot along a given curve.
This is done with the keyword argument `curve`. It can be set to a list of 2D/3D points
which define the curve. When using `curve` any other input from `slice` or `point` will be ignored.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
PlotData1D(u_ode, semi; kwargs...) = PlotData1D(wrap_array_native(u_ode, semi),
                                                mesh_equations_solver_cache(semi)...;
                                                kwargs...)

function PlotData1D(u, mesh, equations, solver, cache;
                    solution_variables=nothing, nvisnodes=nothing,
                    slice=:x, point=(0.0, 0.0, 0.0), curve=nothing)

  solution_variables_ = digest_solution_variables(equations, solution_variables)
  variable_names = SVector(varnames(solution_variables_, equations))

  original_nodes = cache.elements.node_coordinates
  unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations, solver, cache)

  if ndims(mesh) == 1
    x, data, mesh_vertices_x = get_data_1d(original_nodes, unstructured_data, nvisnodes)
    orientation_x = 1
  elseif ndims(mesh) == 2
    if curve != nothing
      x, data, mesh_vertices_x = unstructured_2d_to_1d_curve(original_nodes, unstructured_data, nvisnodes, curve, mesh, solver, cache)
    else
      x, data, mesh_vertices_x = unstructured_2d_to_1d(original_nodes, unstructured_data, nvisnodes, slice, point)
    end
    orientation_x = 0
  else # ndims(mesh) == 3
    if curve != nothing
      x, data, mesh_vertices_x = unstructured_3d_to_1d_curve(original_nodes, unstructured_data, nvisnodes, curve, mesh, solver, cache)
    else
      x, data, mesh_vertices_x = unstructured_3d_to_1d(original_nodes, unstructured_data, nvisnodes, slice, point)
    end
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

# Extract the grid lines from a PlotData1D object to create a PlotMesh1D.
getmesh(pd::PlotData1D) = PlotMesh1D(pd)

# Plot a single variable.
RecipesBase.@recipe function f(pds::PlotDataSeries1D)
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
RecipesBase.@recipe function f(pm::PlotMesh1D)
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
RecipesBase.@recipe function f(pd::PlotData1D)
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
RecipesBase.@recipe function f(sol::TrixiODESolution)
  # Redirect everything to the recipe below
  return sol.u[end], sol.prob.p
end

# Note: If you change the defaults values here, you need to also change them in the PlotData1D or PlotData2D
#       constructor.
RecipesBase.@recipe function f(u, semi::AbstractSemidiscretization;
                   solution_variables=nothing,
                   grid_lines=true, max_supported_level=11, nvisnodes=nothing, slice=:xy,
                   point=(0.0, 0.0, 0.0), curve=nothing)
  # Create a PlotData1D or PlotData2D object depending on the dimension.
  if ndims(semi) == 1
    return PlotData1D(u, semi; solution_variables, nvisnodes, slice, point, curve)
  else
    return PlotData2D(u, semi;
                      solution_variables, grid_lines, max_supported_level,
                      nvisnodes, slice, point)
  end
end

# need to define this function because some keywords from the more general plot recipe
# are not supported (e.g., `max_supported_level`).
@recipe function f(u, semi::DGMultiSemidiscretizationHyperbolic;
                   solution_variables=cons2cons, grid_lines=true)
  return PlotData2D(u, semi)
end

# if u is an Array of SVectors and not a StructArray, convert it to one first.
function PlotData2D(u::Array{<:SVector, 2}, mesh::AbstractMeshData, equations, dg::DGMulti, cache;
                    solution_variables=nothing, nvisnodes=2*polydeg(dg))
  nvars = length(first(u))
  u_structarray = StructArray{eltype(u)}(ntuple(_->zeros(eltype(first(u)), size(u)), nvars))
  for (i, u_i) in enumerate(u)
    u_structarray[i] = u_i
  end

  # re-dispatch to PlotData2D with mesh, equations, dg, cache arguments
  return PlotData2D(u_structarray, mesh, equations, dg, cache;
                    solution_variables=solution_variables, nvisnodes=nvisnodes)
end

function PlotData2D(u::StructArray, mesh::AbstractMeshData, equations, dg::DGMulti, cache;
                    solution_variables=nothing, nvisnodes=2*polydeg(dg))

  rd = dg.basis
  md = mesh.md

  # interpolation matrix from nodal points to plotting points
  @unpack Vp = rd
  interpolate_to_plotting_points!(out, x) = mul!(out, Vp, x)

  solution_variables_ = digest_solution_variables(equations, solution_variables)
  variable_names = SVector(varnames(solution_variables_, equations))

  num_plotting_points = size(Vp, 1)
  nvars = nvariables(equations)
  uEltype = eltype(first(u))
  u_plot_local = StructArray{SVector{nvars, uEltype}}(ntuple(_->zeros(uEltype, num_plotting_points), nvars))
  u_plot = zeros(SVector{nvars, uEltype}, num_plotting_points, md.num_elements)
  for e in eachelement(mesh, dg, cache)

    # interpolate solution to plotting nodes element-by-element
    StructArrays.foreachfield(interpolate_to_plotting_points!, u_plot_local, view(u, :, e))

    # transform nodal values of the solution according to `solution_variables`
    for (i, u_i) in enumerate(u_plot_local)
      u_plot[i, e] = solution_variables_(u_i, equations)
    end
  end

  # interpolate nodal coordinates to plotting points
  x_plot, y_plot = map(x->Vp * x, md.xyz)

  # construct a triangulation of the reference plotting nodes
  t = reference_plotting_triangulation(rd.rstp) # rstp = reference coordinates of plotting points

  xfplot, yfplot = mesh_plotting_wireframe(rd, md, num_plotting_points=nvisnodes)

  # Set the plotting values of solution on faces to nothing - they're not used for Plots.jl since
  # only 2D heatmap plots are supported through TriplotBase/TriplotRecipes.
  ufplot = nothing

  return UnstructuredPlotData2D(x_plot, y_plot, u_plot, t, xfplot, yfplot, ufplot, variable_names)
end

@recipe function f(pds::PlotDataSeries2D{<:UnstructuredPlotData2D})

  pd = pds.plot_data
  @unpack variable_id = pds
  @unpack x, y, u, t, variable_names = pd

  # extract specific solution field to plot
  u_field = zeros(eltype(first(u)), size(u))
  for (i, u_i) in enumerate(u)
    u_field[i] = u_i[variable_id]
  end

  legend --> false
  aspect_ratio --> 1
  title --> pd.variable_names[variable_id]
  xlims --> extrema(x)
  ylims --> extrema(y)
  xguide --> _get_guide(1)
  yguide --> _get_guide(2)
  seriestype --> :heatmap
  colorbar --> :true

  return DGTriPseudocolor(global_plotting_triangulation_Triplot((x, y), u_field, t)...)
end

# Visualize a 2D mesh associated with a DGMulti solver.
@recipe function f(pm::PlotMesh2D{<:UnstructuredPlotData2D})
  pd = pm.plot_data
  @unpack xf, yf = pd

  xlims --> extrema(xf)
  ylims --> extrema(yf)
  aspect_ratio --> :equal
  legend -->  :none

  # Set series properties
  seriestype --> :path
  linecolor --> :grey
  linewidth --> 1

  return xf, yf
end

RecipesBase.@recipe function f(cb::DiscreteCallback{<:Any, <:TimeSeriesCallback}, point_id::Integer)
  return cb.affect!, point_id
end

RecipesBase.@recipe function f(time_series_callback::TimeSeriesCallback, point_id::Integer)
  return PlotData1D(time_series_callback, point_id)
end


end # @muladd
