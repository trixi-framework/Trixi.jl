"""
    P4estMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees, which uses the C library p4est
to manage trees and mesh refinement.

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
mutable struct P4estMesh{NDIMS, RealT<:Real, NDIMSP2} <: AbstractMesh{NDIMS}
  p4est                 ::Ptr{p4est_t}
  p4est_mesh            ::Ptr{p4est_mesh_t}
  tree_node_coordinates ::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree_id]
  nodes                 ::Vector{RealT}
  periodicity           ::NTuple{NDIMS, Bool}
  current_filename      ::String
  unsaved_changes       ::Bool
end


"""
    P4estMesh(trees_per_dimension, mapping, nodes::AbstractVector;
              RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true)

Create a P4estMesh that is a structured curved hypercube of the specified size.

# Arguments
- `trees_per_dimension::NTupleE{NDIMS, Int}`: the number of trees in each dimension.
- `mapping`: a function of `NDIMS` variables to describe the mapping, which transforms
             the reference mesh (`[-1, 1]^n`) to the physical domain.
- `nodes::AbstractVector`: nodes which should be used to store the geometry of the mesh.
                           The mapping will be evaluated at the tensor product of the specified nodes
                           for each tree and will then be interpolated for refined cells.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
"""
function P4estMesh(trees_per_dimension, mapping, nodes::AbstractVector;
                   RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true)
  NDIMS = length(trees_per_dimension)

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
                                                prod(trees_per_dimension))
  calc_node_coordinates!(tree_node_coordinates, mapping, trees_per_dimension, nodes)

  # p4est_connectivity_new_brick has trees in Morton order, so use our own function for this
  conn = connectivity_structured_periodic(trees_per_dimension...)
  p4est = p4est_new_ext(0, conn, 0, initial_refinement_level, false, 0, C_NULL, C_NULL)

  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FACE)
  p4est_mesh = p4est_mesh_new(p4est, ghost, P4EST_CONNECT_FACE)

  function destroy_p4est_structs()
    p4est_mesh_destroy(p4est_mesh)
    p4est_ghost_destroy(ghost)
    p4est_destroy(p4est)
    p4est_connectivity_destroy(conn)
  end

  atexit(destroy_p4est_structs)

  return P4estMesh{NDIMS, RealT, NDIMS+2}(p4est, p4est_mesh, tree_node_coordinates,
                                          nodes, periodicity, "", unsaved_changes)
end


"""
    P4estMesh(trees_per_dimension, mapping; polydeg
              RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true)

Create a P4estMesh that is a structured curved hypercube of the specified size.

# Arguments
- `trees_per_dimension::NTupleE{NDIMS, Int}`: the number of trees in each dimension.
- `mapping`: a function of `NDIMS` variables to describe the mapping, which transforms
             the reference mesh (`[-1, 1]^n`) to the physical domain.
- `polydeg::Integer`: polynomial degree to used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
"""
function P4estMesh(trees_per_dimension, mapping;
                   polydeg, RealT=Float64, initial_refinement_level=0,
                   periodicity=true, unsaved_changes=true)
  basis = LobattoLegendreBasis(RealT, polydeg)
  nodes = basis.nodes

  P4estMesh(trees_per_dimension, mapping, nodes; RealT=RealT,
            initial_refinement_level=initial_refinement_level,
            periodicity=periodicity, unsaved_changes=unsaved_changes)
end


"""
P4estMesh(trees_per_dimension, faces; polydeg, RealT=Float64, initial_refinement_level=0, periodicity=true)

Create a P4estMesh that is a structured curved hypercube of the specified size.

# Arguments
- `trees_per_dimension::NTupleE{NDIMS, Int}`: the number of trees in each dimension.
- `faces::NTuple{2*NDIMS}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                            Each function must take `NDIMS-1` arguments.
                            `faces[1]` describes the face onto which the face in negative x-direction
                            of the unit hypercube is mapped. The face in positive x-direction of
                            the unit hypercube will be mapped onto the face described by `faces[2]`.
                            `faces[3:4]` describe the faces in positive and negative y-direction respectively
                            (in 2D and 3D).
                            `faces[5:6]` describe the faces in positive and negative z-direction respectively (in 3D).
- `polydeg::Integer`: polynomial degree to used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function P4estMesh(trees_per_dimension, faces::Tuple; polydeg, RealT=Float64, initial_refinement_level=0, periodicity=true)
  validate_faces(faces)

  # Use the transfinite mapping with the correct number of arguments
  mapping = transfinite_mapping(faces)

  return P4estMesh(trees_per_dimension, mapping; polydeg=polydeg, RealT=RealT,
                   initial_refinement_level=initial_refinement_level,
                   periodicity=periodicity, mapping_as_string=mapping_as_string)
end


"""
    P4estMesh(trees_per_dimension, coordinates_min, coordinates_max;
              polydeg, initial_refinement_level=0, periodicity=true)

Create a P4estMesh that represents an uncurved structured mesh with a rectangular domain.

# Arguments
- `trees_per_dimension::NTuple{NDIMS, Int}`: the number of trees in each dimension.
- `coordinates_min::NTuple{NDIMS, RealT}`: coordinate of the corner in the negative direction of each dimension.
- `coordinates_max::NTuple{NDIMS, RealT}`: coordinate of the corner in the positive direction of each dimension.
- `polydeg::Integer`: polynomial degree to used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function P4estMesh(trees_per_dimension, coordinates_min, coordinates_max;
                   polydeg, initial_refinement_level=0, periodicity=true)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  mapping = coordinates2mapping(coordinates_min, coordinates_max)

  return P4estMesh(trees_per_dimension, mapping, polydeg=polydeg; RealT=RealT,
                   initial_refinement_level=initial_refinement_level, periodicity=periodicity)
end


# Create a new p4est_connectivity, which represents a structured rectangle with periodic boundaries.
# Similar to p4est_connectivity_new_brick, but doesn't use Morton ordering
# TODO non-periodic
function connectivity_structured_periodic(cells_x, cells_y)
  linear_indices = LinearIndices((cells_x, cells_y))

  # Vertices represent the coordinates of the forest. This is used by p4est
  # to write VTK files.
  # Trixi doesn't use p4est's coordinates, so the vertices can be empty.
  num_vertices = 0
  num_trees = cells_x * cells_y
  # No corner connectivity is needed
  num_corners = 0
  vertices = C_NULL
  tree_to_vertex = C_NULL

  # Periodic boundaries
  tree_to_tree = Matrix{Int32}(undef, 4, num_trees)
  for cell_y in 1:cells_y, cell_x in 1:cells_x
    tree = linear_indices[cell_x, cell_y]
    tree_to_tree[1, tree] = linear_indices[mod(cell_x - 2, cells_x) + 1, cell_y] - 1
    tree_to_tree[2, tree] = linear_indices[mod(cell_x, cells_x) + 1, cell_y] - 1
    tree_to_tree[3, tree] = linear_indices[cell_x, mod(cell_y - 2, cells_y) + 1] - 1
    tree_to_tree[4, tree] = linear_indices[cell_x, mod(cell_y, cells_y) + 1] - 1
  end

  tree_to_face = Matrix{Int8}(undef, 4, num_trees)
  tree_to_face[1, :] .= 1
  tree_to_face[2, :] .= 0
  tree_to_face[3, :] .= 3
  tree_to_face[4, :] .= 2

  tree_to_corner = C_NULL
  ctt_offset = Array{Int32}([0])

  corner_to_tree = C_NULL
  corner_to_corner = C_NULL

  p4est_connectivity_new_copy(num_vertices, num_trees, num_corners,
                              vertices, tree_to_vertex,
                              tree_to_tree, tree_to_face,
                              tree_to_corner, ctt_offset,
                              corner_to_tree, corner_to_corner)
end


# Check if mesh is periodic
isperiodic(mesh::P4estMesh) = all(mesh.periodicity)
isperiodic(mesh::P4estMesh, dimension) = mesh.periodicity[dimension]

@inline Base.ndims(::P4estMesh{NDIMS}) where NDIMS = NDIMS
@inline Base.real(::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline ntrees(mesh::P4estMesh) = mesh.p4est.trees.elem_count
@inline ncells(mesh::P4estMesh) = mesh.p4est_mesh.local_num_quadrants


function Base.show(io::IO, ::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "P4estMesh{", NDIMS, ", ", RealT, "}")
end


function Base.show(io::IO, ::MIME"text/plain", mesh::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
  if get(io, :compact, false)
    show(io, mesh)
  else
    setup = [
             "#trees" => ntrees(mesh),
             "current #cells" => ncells(mesh),
             "periodicity" => mesh.periodicity,
             "polydeg" => length(mesh.nodes),
            ]
    summary_box(io, "P4estMesh{" * string(NDIMS) * ", " * string(RealT) * "}", setup)
  end
end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates, mapping, trees_per_dimension, nodes::AbstractVector)
  linear_indices = LinearIndices(trees_per_dimension)

  # Get cell length in reference mesh
  dx = 2 / trees_per_dimension[1]
  dy = 2 / trees_per_dimension[2]

  for cell_y in 1:trees_per_dimension[2], cell_x in 1:trees_per_dimension[1]
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
