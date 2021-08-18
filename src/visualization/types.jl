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

function Base.iterate(pd::AbstractPlotData, state=1)
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
struct PlotData2D{Coordinates, Data, VariableNames, Vertices} <: AbstractPlotData{2}
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
function Base.show(io::IO, pd::PlotData2D)
  @nospecialize pd # reduce precompilation time

  print(io, "PlotData2D{",
            typeof(pd.x), ",",
            typeof(pd.data), ",",
            typeof(pd.variable_names), ",",
            typeof(pd.mesh_vertices_x),
            "}(<x>, <y>, <data>, <variable_names>, <mesh_vertices_x>, <mesh_vertices_y>)")
end


# holds plotting information for UnstructuredMesh2D and DGMulti-compatible meshes
struct UnstructuredPlotData2D{DataType, FaceDataType, VariableNames, PlottingTriangulation, RealT} <: AbstractPlotData{2}
  x::Array{RealT, 2} # physical nodal coordinates, size (num_plotting_nodes x num_elements)
  y::Array{RealT, 2}
  data::DataType
  t::PlottingTriangulation
  x_face::Array{RealT, 2}
  y_face::Array{RealT, 2}
  face_data::FaceDataType
  variable_names::VariableNames
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pd::UnstructuredPlotData2D)
  @nospecialize pd # reduce precompilation time

  print(io, "UnstructuredPlotData2D{",
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
struct PlotDataSeries{PD<:AbstractPlotData}
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
struct PlotMesh{PD<:AbstractPlotData}
  plot_data::PD
end

# Show only a truncated output for convenience (the full data does not make sense)
function Base.show(io::IO, pm::PlotMesh)
  @nospecialize pm # reduce precompilation time

  print(io, "PlotMesh{", typeof(pm.plot_data), "}(<plot_data>)")
end

# Convenience type to allow dispatch on solution objects that were created by Trixi
#
# This is a union of a Trixi-specific DiffEqBase.ODESolution and of Trixi's own
# TimeIntegratorSolution.
#
# Note: This is an experimental feature and may be changed in future releases without notice.
const TrixiODESolution = Union{ODESolution{T, N, uType, uType2, DType, tType, rateType, P} where
    {T, N, uType, uType2, DType, tType, rateType, P<:ODEProblem{uType_, tType_, isinplace, P_, F_} where
     {uType_, tType_, isinplace, P_<:AbstractSemidiscretization, F_<:ODEFunction{true, typeof(rhs!)}}}, TimeIntegratorSolution}

# Convenience type to allow dispatch on semidiscretizations using the DGMulti solver
const DGMultiSemidiscretizationHyperbolic{Mesh, Equations} =
  SemidiscretizationHyperbolic{Mesh, Equations, InitialCondition, BoundaryCondition, SourceTerms,
  <:DGMulti, Cache} where {InitialCondition, BoundaryCondition, SourceTerms, Cache}

