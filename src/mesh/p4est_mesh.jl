"""
    P4estMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

A structured curved mesh.

Different numbers of cells per dimension are possible and arbitrary functions
can be used as domain faces.

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
mutable struct P4estMesh{NDIMS, RealT<:Real, NDIMSP2} <: AbstractMesh{NDIMS}
  p4est::Ptr{p4est_t}
  p4est_mesh::Ptr{p4est_mesh_t}
  trees_per_dimension::NTuple{NDIMS, Int}
  tree_node_coordinates::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree_id]
  nodes::Vector{RealT}
  periodicity::NTuple{NDIMS, Bool}
  current_filename::String
  unsaved_changes::Bool
end


"""
    P4estMesh(cells_per_dimension, mapping, RealT; unsaved_changes=true, mapping_as_string=mapping2string(mapping, length(cells_per_dimension)))

Create a P4estMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTupleE{NDIMS, Int}`: the number of cells in each dimension.
- `mapping`: a function of `NDIMS` variables to describe the mapping, which transforms
             the reference mesh to the physical domain.
             If no `mapping_as_string` is defined, this function must be defined with the name `mapping`
             to allow for restarts.
             This will be changed in the future, see https://github.com/trixi-framework/Trixi.jl/issues/541.
- `RealT::Type`: the type that should be used for coordinates.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
"""
function P4estMesh(cells_per_dimension, mapping, nodes::AbstractVector;
                   RealT=Float64, periodicity=true, unsaved_changes=true)
  NDIMS = length(cells_per_dimension)

  # Convert periodicity to a Tuple of a Bool for every dimension
  if all(periodicity)
    # Also catches case where periodicity = true
    periodicity = ntuple(_->true, NDIMS)
  elseif !any(periodicity)
    # Also catches case where periodicity = false
    periodicity = ntuple(_->false, NDIMS)
  else
    # Default case if periodicity is an iterable
    periodicity = Tuple(periodicity)
  end

  tree_node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS,
                                                ntuple(_ -> length(nodes), NDIMS)...,
                                                prod(cells_per_dimension))
  calc_node_coordinates!(tree_node_coordinates, mapping, cells_per_dimension, nodes)

  initial_refinement_level = 0
  conn = p4est_connectivity_new_brick(cells_per_dimension..., true, true)
  p4est = p4est_new_ext(0, conn, 0, initial_refinement_level, false, 0, C_NULL, C_NULL)

  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FACE)
  p4est_mesh = p4est_mesh_new(p4est, ghost, P4EST_CONNECT_FACE)

  return P4estMesh{NDIMS, RealT, NDIMS+2}(p4est, p4est_mesh, Tuple(cells_per_dimension),
                                          tree_node_coordinates, nodes, periodicity, "", unsaved_changes)
end


function P4estMesh(cells_per_dimension, mapping, polydeg::Integer;
                   RealT=Float64, periodicity=true, unsaved_changes=true)
  basis = LobattoLegendreBasis(RealT, polydeg)
  nodes = basis.nodes

  P4estMesh(cells_per_dimension, mapping, nodes; RealT=RealT,
            periodicity=periodicity, unsaved_changes=unsaved_changes)
end


"""
    P4estMesh(cells_per_dimension, faces, RealT; unsaved_changes=true, faces_as_string=faces2string(faces))

Create a P4estMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTupleE{NDIMS, Int}`: the number of cells in each dimension.
- `faces::NTuple{2*NDIMS}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                            Each function must take `NDIMS-1` arguments.
                            `faces[1]` describes the face onto which the face in negative x-direction
                            of the unit hypercube is mapped. The face in positive x-direction of
                            the unit hypercube will be mapped onto the face described by `faces[2]`.
                            `faces[3:4]` describe the faces in positive and negative y-direction respectively
                            (in 2D and 3D).
                            `faces[5:6]` describe the faces in positive and negative z-direction respectively (in 3D).
- `RealT::Type`: the type that should be used for coordinates.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function P4estMesh(cells_per_dimension, faces::Tuple, polydeg::Integer; RealT=Float64, periodicity=true)
  validate_faces(faces)

  # Use the transfinite mapping with the correct number of arguments
  mapping = transfinite_mapping(faces)

  return P4estMesh(cells_per_dimension, mapping, polydeg; RealT=RealT,
                   periodicity=periodicity, mapping_as_string=mapping_as_string)
end


"""
    P4estMesh(cells_per_dimension, coordinates_min, coordinates_max)

Create a P4estMesh that represents a uncurved structured mesh with a rectangular domain.

# Arguments
- `cells_per_dimension::NTuple{NDIMS, Int}`: the number of cells in each dimension.
- `coordinates_min::NTuple{NDIMS, RealT}`: coordinate of the corner in the negative direction of each dimension.
- `coordinates_max::NTuple{NDIMS, RealT}`: coordinate of the corner in the positive direction of each dimension.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function P4estMesh(cells_per_dimension, coordinates_min, coordinates_max, polydeg; periodicity=true)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  mapping = coordinates2mapping(coordinates_min, coordinates_max)

  return P4estMesh(cells_per_dimension, mapping, polydeg; RealT=RealT, periodicity=periodicity)
end


# Check if mesh is periodic
isperiodic(mesh::P4estMesh) = all(mesh.periodicity)
isperiodic(mesh::P4estMesh, dimension) = mesh.periodicity[dimension]

@inline Base.ndims(::P4estMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT
Base.size(mesh::P4estMesh) = mesh.trees_per_dimension
Base.size(mesh::P4estMesh, i) = mesh.trees_per_dimension[i]
Base.axes(mesh::P4estMesh) = map(Base.OneTo, mesh.trees_per_dimension)
Base.axes(mesh::P4estMesh, i) = Base.OneTo(mesh.trees_per_dimension[i])


function Base.show(io::IO, ::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "P4estMesh{", NDIMS, ", ", RealT, "}")
end


# function Base.show(io::IO, ::MIME"text/plain", mesh::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
#   if get(io, :compact, false)
#     show(io, mesh)
#   else
#     summary_header(io, "P4estMesh{" * string(NDIMS) * ", " * string(RealT) * "}")
#     summary_line(io, "size", size(mesh))

#     summary_line(io, "mapping", "")
#     # Print code lines of mapping_as_string
#     mapping_lines = split(mesh.mapping_as_string, ";")
#     for i in eachindex(mapping_lines)
#       summary_line(increment_indent(io), "line $i", strip(mapping_lines[i]))
#     end
#     summary_footer(io)
#   end
# end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates, mapping, cells_per_dimension, nodes::AbstractVector)
  linear_indices = LinearIndices(cells_per_dimension)

  # Get cell length in reference mesh
  dx = 2 / cells_per_dimension[1]
  dy = 2 / cells_per_dimension[2]

  for cell_y in 1:cells_per_dimension[2], cell_x in 1:cells_per_dimension[1]
    tree_id = linear_indices[cell_x, cell_y]

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x-1) * dx + dx/2
    cell_y_offset = -1 + (cell_y-1) * dy + dy/2

    for j in eachindex(nodes), i in eachindex(nodes)
      # node_coordinates are the mapped reference node coordinates
      node_coordinates[:, i, j, tree_id] .= mapping(cell_x_offset + dx/2 * nodes[i],
                                                    cell_y_offset + dy/2 * nodes[j])
    end
  end
end
