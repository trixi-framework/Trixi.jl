# Convenience type to allow dispatch on solution objects that were created by Trixi.jl
#
# This is a union of a Trixi.jl-specific SciMLBase.ODESolution and of Trixi.jl's own
# TimeIntegratorSolution.
#
# Note: This is an experimental feature and may be changed in future releases without notice.
#! format: off
const TrixiODESolution = Union{ODESolution{T, N, uType, uType2, DType, tType, rateType, P} where
    {T, N, uType, uType2, DType, tType, rateType, P<:ODEProblem{uType_, tType_, isinplace, P_, F_} where
     {uType_, tType_, isinplace, P_<:AbstractSemidiscretization, F_}}, TimeIntegratorSolution}
#! format: on

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This file holds plotting types which can be used for both Plots.jl and Makie.jl.

# This abstract type is used to derive PlotData types of different dimensions; but still allows to share some functions for them.
abstract type AbstractPlotData{NDIMS} end

# Define additional methods for convenience.
# These are defined for AbstractPlotData, so they can be used for all types of plot data.
Base.firstindex(pd::AbstractPlotData) = first(pd.variable_names)
Base.lastindex(pd::AbstractPlotData) = last(pd.variable_names)
Base.length(pd::AbstractPlotData) = length(pd.variable_names)
Base.size(pd::AbstractPlotData) = (length(pd),)
Base.keys(pd::AbstractPlotData) = tuple(pd.variable_names...)

function Base.iterate(pd::AbstractPlotData, state = 1)
    if state > length(pd)
        return nothing
    else
        return (pd.variable_names[state] => pd[pd.variable_names[state]], state + 1)
    end
end

"""
    Base.getindex(pd::AbstractPlotData, variable_name)

Extract a single variable `variable_name` from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function Base.getindex(pd::AbstractPlotData, variable_name)
    variable_id = findfirst(isequal(variable_name), pd.variable_names)

    if isnothing(variable_id)
        throw(KeyError(variable_name))
    end

    return PlotDataSeries(pd, variable_id)
end

Base.eltype(pd::AbstractPlotData) = Pair{String, PlotDataSeries{typeof(pd)}}

"""
    PlotData2D

Holds all relevant data for creating 2D plots of multiple solution variables and to visualize the
mesh.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct PlotData2DCartesian{Coordinates, Data, VariableNames, Vertices} <:
       AbstractPlotData{2}
    x::Coordinates
    y::Coordinates
    data::Data
    variable_names::VariableNames
    mesh_vertices_x::Vertices
    mesh_vertices_y::Vertices
    orientation_x::Int
    orientation_y::Int
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pd::PlotData2DCartesian)
    @nospecialize pd # reduce precompilation time

    print(io, "PlotData2DCartesian{",
          typeof(pd.x), ",",
          typeof(pd.data), ",",
          typeof(pd.variable_names), ",",
          typeof(pd.mesh_vertices_x),
          "}(<x>, <y>, <data>, <variable_names>, <mesh_vertices_x>, <mesh_vertices_y>)")
end

# holds plotting information for UnstructuredMesh2D and DGMulti-compatible meshes
struct PlotData2DTriangulated{DataType, NodeType, FaceNodeType, FaceDataType,
                              VariableNames, PlottingTriangulation} <:
       AbstractPlotData{2}
    x::NodeType # physical nodal coordinates, size (num_plotting_nodes x num_elements)
    y::NodeType
    data::DataType
    t::PlottingTriangulation
    x_face::FaceNodeType
    y_face::FaceNodeType
    face_data::FaceDataType
    variable_names::VariableNames
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pd::PlotData2DTriangulated)
    @nospecialize pd # reduce precompilation time

    print(io, "PlotData2DTriangulated{",
          typeof(pd.x), ", ",
          typeof(pd.data), ", ",
          typeof(pd.x_face), ", ",
          typeof(pd.face_data), ", ",
          typeof(pd.variable_names),
          "}(<x>, <y>, <data>, <plot_triangulation>, <x_face>, <y_face>, <face_data>, <variable_names>)")
end

"""
    PlotData1D

Holds all relevant data for creating 1D plots of multiple solution variables and to visualize the
mesh.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct PlotData1D{Coordinates, Data, VariableNames, Vertices} <: AbstractPlotData{1}
    x::Coordinates
    data::Data
    variable_names::VariableNames
    mesh_vertices_x::Vertices
    orientation_x::Integer
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

# Auxiliary data structure for visualizing a single variable
#
# Note: This is an experimental feature and may be changed in future releases without notice.
struct PlotDataSeries{PD <: AbstractPlotData}
    plot_data::PD
    variable_id::Int
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pds::PlotDataSeries)
    @nospecialize pds # reduce precompilation time

    print(io, "PlotDataSeries{", typeof(pds.plot_data), "}(<plot_data>, ",
          pds.variable_id, ")")
end

# Generic PlotMesh wrapper type.
struct PlotMesh{PD <: AbstractPlotData}
    plot_data::PD
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pm::PlotMesh)
    @nospecialize pm # reduce precompilation time

    print(io, "PlotMesh{", typeof(pm.plot_data), "}(<plot_data>)")
end

"""
    getmesh(pd::AbstractPlotData)

Extract grid lines from `pd` for plotting with `Plots.plot`.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
getmesh(pd::AbstractPlotData) = PlotMesh(pd)

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
function PlotData2D(u_ode, semi; kwargs...)
    PlotData2D(wrap_array_native(u_ode, semi),
               mesh_equations_solver_cache(semi)...;
               kwargs...)
end

# Redirect `PlotDataTriangulated2D` constructor.
function PlotData2DTriangulated(u_ode, semi; kwargs...)
    PlotData2DTriangulated(wrap_array_native(u_ode, semi),
                           mesh_equations_solver_cache(semi)...;
                           kwargs...)
end

# Create a PlotData2DCartesian object for TreeMeshes on default.
function PlotData2D(u, mesh::TreeMesh, equations, solver, cache; kwargs...)
    PlotData2DCartesian(u, mesh::TreeMesh, equations, solver, cache; kwargs...)
end

# Create a PlotData2DTriangulated object for any type of mesh other than the TreeMesh.
function PlotData2D(u, mesh, equations, solver, cache; kwargs...)
    PlotData2DTriangulated(u, mesh, equations, solver, cache; kwargs...)
end

# Create a PlotData2DCartesian for a TreeMesh.
function PlotData2DCartesian(u, mesh::TreeMesh, equations, solver, cache;
                             solution_variables = nothing,
                             grid_lines = true, max_supported_level = 11,
                             nvisnodes = nothing,
                             slice = :xy, point = (0.0, 0.0, 0.0))
    @assert ndims(mesh) in (2, 3) "unsupported number of dimensions $ndims (must be 2 or 3)"
    solution_variables_ = digest_solution_variables(equations, solution_variables)

    # Extract mesh info
    center_level_0 = mesh.tree.center_level_0
    length_level_0 = mesh.tree.length_level_0
    leaf_cell_ids = leaf_cells(mesh.tree)
    coordinates = mesh.tree.coordinates[:, leaf_cell_ids]
    levels = mesh.tree.levels[leaf_cell_ids]

    unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations,
                                              solver, cache)
    x, y, data, mesh_vertices_x, mesh_vertices_y = get_data_2d(center_level_0,
                                                               length_level_0,
                                                               leaf_cell_ids,
                                                               coordinates, levels,
                                                               ndims(mesh),
                                                               unstructured_data,
                                                               nnodes(solver),
                                                               grid_lines,
                                                               max_supported_level,
                                                               nvisnodes,
                                                               slice, point)
    variable_names = SVector(varnames(solution_variables_, equations))

    orientation_x, orientation_y = _get_orientations(mesh, slice)

    return PlotData2DCartesian(x, y, data, variable_names, mesh_vertices_x,
                               mesh_vertices_y,
                               orientation_x, orientation_y)
end

"""
    PlotData2D(sol; kwargs...)

Create a `PlotData2D` object from a solution object created by either `OrdinaryDiffEq.solve!` (which
returns a `SciMLBase.ODESolution`) or Trixi.jl's own `solve!` (which returns a
`TimeIntegratorSolution`).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function PlotData2D(sol::TrixiODESolution; kwargs...)
    PlotData2D(sol.u[end], sol.prob.p; kwargs...)
end

# Also redirect when using PlotData2DTriangulate.
function PlotData2DTriangulated(sol::TrixiODESolution; kwargs...)
    PlotData2DTriangulated(sol.u[end], sol.prob.p; kwargs...)
end

# If `u` is an `Array{<:SVectors}` and not a `StructArray`, convert it to a `StructArray` first.
function PlotData2D(u::Array{<:SVector, 2}, mesh, equations, dg::DGMulti, cache;
                    solution_variables = nothing, nvisnodes = 2 * nnodes(dg))
    nvars = length(first(u))
    u_structarray = StructArray{eltype(u)}(ntuple(_ -> zeros(eltype(first(u)), size(u)),
                                                  nvars))
    for (i, u_i) in enumerate(u)
        u_structarray[i] = u_i
    end

    # re-dispatch to PlotData2D with mesh, equations, dg, cache arguments
    return PlotData2D(u_structarray, mesh, equations, dg, cache;
                      solution_variables = solution_variables, nvisnodes = nvisnodes)
end

# constructor which returns an `PlotData2DTriangulated` object.
function PlotData2D(u::StructArray, mesh, equations, dg::DGMulti, cache;
                    solution_variables = nothing, nvisnodes = 2 * nnodes(dg))
    rd = dg.basis
    md = mesh.md

    # Vp = the interpolation matrix from nodal points to plotting points
    @unpack Vp = rd
    interpolate_to_plotting_points!(out, x) = mul!(out, Vp, x)

    solution_variables_ = digest_solution_variables(equations, solution_variables)
    variable_names = SVector(varnames(solution_variables_, equations))

    if Vp isa UniformScaling
        num_plotting_points = size(u, 1)
    else
        num_plotting_points = size(Vp, 1)
    end
    nvars = nvariables(equations)
    uEltype = eltype(first(u))
    u_plot = StructArray{SVector{nvars, uEltype}}(ntuple(_ -> zeros(uEltype,
                                                                    num_plotting_points,
                                                                    md.num_elements),
                                                         nvars))

    for e in eachelement(mesh, dg, cache)
        # interpolate solution to plotting nodes element-by-element
        StructArrays.foreachfield(interpolate_to_plotting_points!, view(u_plot, :, e),
                                  view(u, :, e))

        # transform nodal values of the solution according to `solution_variables`
        transform_to_solution_variables!(view(u_plot, :, e), solution_variables_,
                                         equations)
    end

    # interpolate nodal coordinates to plotting points
    x_plot, y_plot = map(x -> Vp * x, md.xyz) # md.xyz is a tuple of arrays containing nodal coordinates

    # construct a triangulation of the reference plotting nodes
    t = reference_plotting_triangulation(rd.rstp) # rd.rstp = reference coordinates of plotting points

    x_face, y_face, face_data = mesh_plotting_wireframe(u, mesh, equations, dg, cache;
                                                        nvisnodes = nvisnodes)

    return PlotData2DTriangulated(x_plot, y_plot, u_plot, t, x_face, y_face, face_data,
                                  variable_names)
end

# specializes the PlotData2D constructor to return an PlotData2DTriangulated for any type of mesh.
function PlotData2DTriangulated(u, mesh, equations, dg::DGSEM, cache;
                                solution_variables = nothing,
                                nvisnodes = 2 * polydeg(dg))
    @assert ndims(mesh)==2 "Input must be two-dimensional."

    n_nodes_2d = nnodes(dg)^ndims(mesh)
    n_elements = nelements(dg, cache)

    # build nodes on reference element (seems to be the right ordering)
    r, s = reference_node_coordinates_2d(dg)

    # reference plotting nodes
    if nvisnodes == 0 || nvisnodes === nothing
        nvisnodes = polydeg(dg) + 1
    end
    plotting_interp_matrix = plotting_interpolation_matrix(dg; nvisnodes = nvisnodes)

    # create triangulation for plotting nodes
    r_plot, s_plot = (x -> plotting_interp_matrix * x).((r, s)) # interpolate dg nodes to plotting nodes

    # construct a triangulation of the plotting nodes
    t = reference_plotting_triangulation((r_plot, s_plot))

    # extract x,y coordinates and solutions on each element
    uEltype = eltype(u)
    nvars = nvariables(equations)
    x = reshape(view(cache.elements.node_coordinates, 1, :, :, :), n_nodes_2d,
                n_elements)
    y = reshape(view(cache.elements.node_coordinates, 2, :, :, :), n_nodes_2d,
                n_elements)
    u_extracted = StructArray{SVector{nvars, uEltype}}(ntuple(_ -> similar(x,
                                                                           (n_nodes_2d,
                                                                            n_elements)),
                                                              nvars))
    for element in eachelement(dg, cache)
        sk = 1
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_extracted[sk, element] = u_node
            sk += 1
        end
    end

    # interpolate to volume plotting points
    xplot, yplot = plotting_interp_matrix * x, plotting_interp_matrix * y
    uplot = StructArray{SVector{nvars, uEltype}}(map(x -> plotting_interp_matrix * x,
                                                     StructArrays.components(u_extracted)))

    xfp, yfp, ufp = mesh_plotting_wireframe(u_extracted, mesh, equations, dg, cache;
                                            nvisnodes = nvisnodes)

    # convert variables based on solution_variables mapping
    solution_variables_ = digest_solution_variables(equations, solution_variables)
    variable_names = SVector(varnames(solution_variables_, equations))

    transform_to_solution_variables!(uplot, solution_variables_, equations)
    transform_to_solution_variables!(ufp, solution_variables_, equations)

    return PlotData2DTriangulated(xplot, yplot, uplot, t, xfp, yfp, ufp, variable_names)
end

# Wrapper struct to indicate that an array represents a scalar data field. Used only for dispatch.
struct ScalarData{T}
    data::T
end

"""
    ScalarPlotData2D(u, semi::AbstractSemidiscretization; kwargs...)

Returns an `PlotData2DTriangulated` object which is used to visualize a single scalar field.
`u` should be an array whose entries correspond to values of the scalar field at nodal points.
"""
function ScalarPlotData2D(u, semi::AbstractSemidiscretization; kwargs...)
    ScalarPlotData2D(u, mesh_equations_solver_cache(semi)...; kwargs...)
end

# Returns an `PlotData2DTriangulated` which is used to visualize a single scalar field
function ScalarPlotData2D(u, mesh, equations, dg::DGMulti, cache;
                          variable_name = nothing, nvisnodes = 2 * nnodes(dg))
    rd = dg.basis
    md = mesh.md

    # Vp = the interpolation matrix from nodal points to plotting points
    @unpack Vp = rd

    # interpolate nodal coordinates and solution field to plotting points
    x_plot, y_plot = map(x -> Vp * x, md.xyz) # md.xyz is a tuple of arrays containing nodal coordinates
    u_plot = Vp * u

    # construct a triangulation of the reference plotting nodes
    t = reference_plotting_triangulation(rd.rstp) # rd.rstp = reference coordinates of plotting points

    # Ignore face data when plotting `ScalarPlotData2D`, since mesh lines can be plotted using
    # existing functionality based on `PlotData2D(sol)`.
    x_face, y_face, face_data = mesh_plotting_wireframe(ScalarData(u), mesh, equations,
                                                        dg, cache;
                                                        nvisnodes = 2 * nnodes(dg))

    # wrap solution in ScalarData struct for recipe dispatch
    return PlotData2DTriangulated(x_plot, y_plot, ScalarData(u_plot), t,
                                  x_face, y_face, face_data, variable_name)
end

function ScalarPlotData2D(u, mesh, equations, dg::DGSEM, cache; variable_name = nothing,
                          nvisnodes = 2 * nnodes(dg))
    n_nodes_2d = nnodes(dg)^ndims(mesh)
    n_elements = nelements(dg, cache)

    # build nodes on reference element (seems to be the right ordering)
    r, s = reference_node_coordinates_2d(dg)

    # reference plotting nodes
    if nvisnodes == 0 || nvisnodes === nothing
        nvisnodes = polydeg(dg) + 1
    end
    plotting_interp_matrix = plotting_interpolation_matrix(dg; nvisnodes = nvisnodes)

    # create triangulation for plotting nodes
    r_plot, s_plot = (x -> plotting_interp_matrix * x).((r, s)) # interpolate dg nodes to plotting nodes

    # construct a triangulation of the plotting nodes
    t = reference_plotting_triangulation((r_plot, s_plot))

    # extract x,y coordinates and reshape them into matrices of size (n_nodes_2d, n_elements)
    x = view(cache.elements.node_coordinates, 1, :, :, :)
    y = view(cache.elements.node_coordinates, 2, :, :, :)
    x, y = reshape.((x, y), n_nodes_2d, n_elements)

    # interpolate to volume plotting points by multiplying each column by `plotting_interp_matrix`
    x_plot, y_plot = plotting_interp_matrix * x, plotting_interp_matrix * y
    u_plot = plotting_interp_matrix * reshape(u, size(x))

    # Ignore face data when plotting `ScalarPlotData2D`, since mesh lines can be plotted using
    # existing functionality based on `PlotData2D(sol)`.
    x_face, y_face, face_data = mesh_plotting_wireframe(ScalarData(u), mesh, equations,
                                                        dg, cache;
                                                        nvisnodes = 2 * nnodes(dg))

    # wrap solution in ScalarData struct for recipe dispatch
    return PlotData2DTriangulated(x_plot, y_plot, ScalarData(u_plot), t,
                                  x_face, y_face, face_data, variable_name)
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
This applies analogously to three-dimensional simulations, where `slice` may be `:xy`, `:xz`, or `:yz`.

Another way to visualize 2D/3D data is by creating a plot along a given curve.
This is done with the keyword argument `curve`. It can be set to a list of 2D/3D points
which define the curve. When using `curve` any other input from `slice` or `point` will be ignored.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function PlotData1D(u_ode, semi; kwargs...)
    PlotData1D(wrap_array_native(u_ode, semi),
               mesh_equations_solver_cache(semi)...;
               kwargs...)
end

function PlotData1D(u, mesh::TreeMesh, equations, solver, cache;
                    solution_variables = nothing, nvisnodes = nothing,
                    slice = :x, point = (0.0, 0.0, 0.0), curve = nothing)
    solution_variables_ = digest_solution_variables(equations, solution_variables)
    variable_names = SVector(varnames(solution_variables_, equations))

    original_nodes = cache.elements.node_coordinates
    unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations,
                                              solver, cache)

    orientation_x = 0 # Set 'orientation' to zero on default.

    if ndims(mesh) == 1
        x, data, mesh_vertices_x = get_data_1d(original_nodes, unstructured_data,
                                               nvisnodes)
        orientation_x = 1

        # Special care is required for first-order FV approximations since the nodes are the
        # cell centers and do not contain the boundaries
        n_nodes = size(unstructured_data, 1)
        if n_nodes == 1
            n_visnodes = length(x) ÷ nelements(solver, cache)
            if n_visnodes != 2
                throw(ArgumentError("This number of visualization nodes is currently not supported for finite volume approximations."))
            end
            left_boundary = mesh.tree.center_level_0[1] - mesh.tree.length_level_0 / 2
            dx_2 = zero(left_boundary)
            for i in 1:div(length(x), 2)
                # Adjust plot nodes so that they are at the boundaries of each element
                dx_2 = x[2 * i - 1] - left_boundary
                x[2 * i - 1] -= dx_2
                x[2 * i] += dx_2
                left_boundary = left_boundary + 2 * dx_2

                # Adjust mesh plot nodes
                mesh_vertices_x[i] -= dx_2
            end
            mesh_vertices_x[end] += dx_2
        end
    elseif ndims(mesh) == 2
        if curve !== nothing
            x, data, mesh_vertices_x = unstructured_2d_to_1d_curve(original_nodes,
                                                                   unstructured_data,
                                                                   nvisnodes, curve,
                                                                   mesh, solver, cache)
        else
            x, data, mesh_vertices_x = unstructured_2d_to_1d(original_nodes,
                                                             unstructured_data,
                                                             nvisnodes, slice, point)
        end
    else # ndims(mesh) == 3
        if curve !== nothing
            x, data, mesh_vertices_x = unstructured_3d_to_1d_curve(original_nodes,
                                                                   unstructured_data,
                                                                   nvisnodes, curve,
                                                                   mesh, solver, cache)
        else
            x, data, mesh_vertices_x = unstructured_3d_to_1d(original_nodes,
                                                             unstructured_data,
                                                             nvisnodes, slice, point)
        end
    end

    return PlotData1D(x, data, variable_names, mesh_vertices_x,
                      orientation_x)
end

function PlotData1D(u, mesh, equations, solver, cache;
                    solution_variables = nothing, nvisnodes = nothing,
                    slice = :x, point = (0.0, 0.0, 0.0), curve = nothing)
    solution_variables_ = digest_solution_variables(equations, solution_variables)
    variable_names = SVector(varnames(solution_variables_, equations))

    original_nodes = cache.elements.node_coordinates
    unstructured_data = get_unstructured_data(u, solution_variables_, mesh, equations,
                                              solver, cache)

    orientation_x = 0 # Set 'orientation' to zero on default.

    if ndims(mesh) == 1
        x, data, mesh_vertices_x = get_data_1d(original_nodes, unstructured_data,
                                               nvisnodes)
        orientation_x = 1
    elseif ndims(mesh) == 2
        # Create a 'PlotData2DTriangulated' object so a triangulation can be used when extracting relevant data.
        pd = PlotData2DTriangulated(u, mesh, equations, solver, cache;
                                    solution_variables, nvisnodes)
        x, data, mesh_vertices_x = unstructured_2d_to_1d_curve(pd, curve, slice, point,
                                                               nvisnodes)
    else # ndims(mesh) == 3
        # Extract the information required to create a PlotData1D object.
        x, data, mesh_vertices_x = unstructured_3d_to_1d_curve(original_nodes, u, curve,
                                                               slice, point, nvisnodes)
    end

    return PlotData1D(x, data, variable_names, mesh_vertices_x,
                      orientation_x)
end

# Specializes the `PlotData1D` constructor for one-dimensional `DGMulti` solvers.
function PlotData1D(u, mesh, equations, dg::DGMulti{1}, cache;
                    solution_variables = nothing)
    solution_variables_ = digest_solution_variables(equations, solution_variables)
    variable_names = SVector(varnames(solution_variables_, equations))

    orientation_x = 0 # Set 'orientation' to zero on default.

    if u isa StructArray
        # Convert conserved variables to the given `solution_variables` and set up
        # plotting coordinates
        # This uses a "structure of arrays"
        data = map(x -> vcat(dg.basis.Vp * x, fill(NaN, 1, size(u, 2))),
                   StructArrays.components(solution_variables_.(u, equations)))
        x = vcat(dg.basis.Vp * mesh.md.x, fill(NaN, 1, size(u, 2)))

        # Here, we ensure that `DGMulti` visualization uses the same data layout and format
        # as `TreeMesh`. This enables us to reuse existing plot recipes. In particular,
        # `hcat(data...)` creates a matrix of size `num_plotting_points` by `nvariables(equations)`,
        # with data on different elements separated by `NaNs`.
        x_plot = vec(x)
        data_plot = hcat(vec.(data)...)
    else
        # Convert conserved variables to the given `solution_variables` and set up
        # plotting coordinates
        # This uses an "array of structures"
        data_tmp = dg.basis.Vp * solution_variables_.(u, equations)
        data = vcat(data_tmp, fill(NaN * zero(eltype(data_tmp)), 1, size(u, 2)))
        x = vcat(dg.basis.Vp * mesh.md.x, fill(NaN, 1, size(u, 2)))

        # Same as above - we create `data_plot` as array of size `num_plotting_points`
        # by "number of plotting variables".
        x_plot = vec(x)
        data_plot = permutedims(reinterpret(reshape, eltype(eltype(data)), vec(data)),
                                (2, 1))
    end

    return PlotData1D(x_plot, data_plot, variable_names, mesh.md.VX, orientation_x)
end

"""
    PlotData1D(sol; kwargs...)

Create a `PlotData1D` object from a solution object created by either `OrdinaryDiffEq.solve!`
(which returns a `SciMLBase.ODESolution`) or Trixi.jl's own `solve!` (which returns a
`TimeIntegratorSolution`).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function PlotData1D(sol::TrixiODESolution; kwargs...)
    PlotData1D(sol.u[end], sol.prob.p; kwargs...)
end

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

function PlotData1D(cb::DiscreteCallback{<:Any, <:TimeSeriesCallback},
                    point_id::Integer)
    return PlotData1D(cb.affect!, point_id)
end
end # @muladd
