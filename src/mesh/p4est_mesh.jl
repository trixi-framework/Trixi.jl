"""
    P4estMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the C library p4est
to manage trees and mesh refinement.

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
mutable struct P4estMesh{NDIMS, RealT<:Real, NDIMSP2} <: AbstractMesh{NDIMS}
  p4est                 ::Ptr{p4est_t}
  p4est_mesh            ::Ptr{p4est_mesh_t}
  tree_node_coordinates ::Array{RealT, NDIMSP2} # [dimension, i, j, k, n_tree]
  nodes                 ::Vector{RealT}
  boundary_names        ::Array{Symbol, 2}      # [face direction, n_tree]
  current_filename      ::String
  unsaved_changes       ::Bool
end


"""
    P4estMesh(trees_per_dimension; polydeg,
              mapping=nothing, faces=nothing, coordinates_min=nothing, coordinates_max=nothing,
              RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true)

Create a structured curved P4estMesh of the specified size.

There are three ways to map the mesh to the physical domain.
1. Define a `mapping` that maps the hypercube `[-1, 1]^n`.
2. Specify a `Tuple` `faces` of functions that parametrize each face.
3. Create a rectangular mesh by specifying `coordinates_min` and `coordinates_max`.

# Arguments
- `trees_per_dimension::NTupleE{NDIMS, Int}`: the number of trees in each dimension.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the reference mesh (`[-1, 1]^n`) to the physical domain.
             Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `faces::NTuple{2*NDIMS}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                            Each function must take `NDIMS-1` arguments.
                            `faces[1]` describes the face onto which the face in negative x-direction
                            of the unit hypercube is mapped. The face in positive x-direction of
                            the unit hypercube will be mapped onto the face described by `faces[2]`.
                            `faces[3:4]` describe the faces in positive and negative y-direction respectively
                            (in 2D and 3D).
                            `faces[5:6]` describe the faces in positive and negative z-direction respectively (in 3D).
                            Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `coordinates_min`: vector or tuple of the coordinates of the corner in the negative direction of each dimension
                     to create a rectangular mesh.
                     Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `coordinates_max`: vector or tuple of the coordinates of the corner in the positive direction of each dimension
                     to create a rectangular mesh.
                     Use only one of `mapping`, `faces` and `coordinates_min`/`coordinates_max`.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
"""
function P4estMesh(trees_per_dimension; polydeg,
                   mapping=nothing, faces=nothing, coordinates_min=nothing, coordinates_max=nothing,
                   RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true)

  @assert (
    (coordinates_min === nothing) === (coordinates_max === nothing)
  ) "Either both or none of coordinates_min and coordinates_max must be specified"

  @assert count(i -> i !== nothing,
    (mapping, faces, coordinates_min)
  ) == 1 "Exactly one of mapping, faces and coordinates_min/max must be specified"

  # Extract mapping
  if faces !== nothing
    validate_faces(faces)
    mapping = transfinite_mapping(faces)
  elseif coordinates_min !== nothing
    mapping = coordinates2mapping(coordinates_min, coordinates_max)
  end

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

  basis = LobattoLegendreBasis(RealT, polydeg)
  nodes = basis.nodes
  tree_node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS,
                                                ntuple(_ -> length(nodes), NDIMS)...,
                                                prod(trees_per_dimension))
  calc_node_coordinates!(tree_node_coordinates, mapping, trees_per_dimension, nodes)

  # p4est_connectivity_new_brick has trees in Morton order, so use our own function for this
  conn = connectivity_structured(trees_per_dimension..., periodicity)
  p4est = p4est_new_ext(0, conn, 0, initial_refinement_level, false, 0, C_NULL, C_NULL)

  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FACE)
  p4est_mesh = p4est_mesh_new(p4est, ghost, P4EST_CONNECT_FACE)

  # Destroy p4est structs at exit of Julia
  function destroy_p4est_structs()
    p4est_mesh_destroy(p4est_mesh)
    p4est_ghost_destroy(ghost)
    p4est_destroy(p4est)
    p4est_connectivity_destroy(conn)
  end

  atexit(destroy_p4est_structs)

  # Non-periodic boundaries
  boundary_names = fill(Symbol("---"), 4, prod(trees_per_dimension))
  linear_indices = LinearIndices(trees_per_dimension)

  # Boundaries in x-direction
  if !periodicity[1]
    for cell_y in 1:trees_per_dimension[2]
      tree = linear_indices[1, cell_y]
      boundary_names[1, tree] = :x_neg

      tree = linear_indices[end, cell_y]
      boundary_names[2, tree] = :x_pos
    end
  end

  # Boundaries in y-direction
  if !periodicity[2]
    for cell_x in 1:trees_per_dimension[1]
      tree = linear_indices[cell_x, 1]
      boundary_names[3, tree] = :y_neg

      tree = linear_indices[cell_x, end]
      boundary_names[4, tree] = :y_pos
    end
  end

  return P4estMesh{NDIMS, RealT, NDIMS+2}(p4est, p4est_mesh, tree_node_coordinates, nodes,
                                          boundary_names, "", unsaved_changes)
end


function P4estMesh(meshfile::String;
                   RealT=Float64, initial_refinement_level=0, unsaved_changes=true)
  # TODO P4EST
  NDIMS = 2

  conn = p4est_connectivity_read_inp(meshfile)
  n_trees = conn.num_trees

  vertices        = unsafe_wrap(Array, conn.vertices, (3, conn.num_vertices))
  tree_to_vertex  = unsafe_wrap(Array, conn.tree_to_vertex, (4, n_trees))

  # Uncurved geometry, only save corner coordinates
  nodes = [-1.0, 1.0]
  tree_node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS,
                                                ntuple(_ -> length(nodes), NDIMS)...,
                                                n_trees)
  calc_node_coordinates!(tree_node_coordinates, vertices, tree_to_vertex)

  p4est = p4est_new_ext(0, conn, 0, initial_refinement_level, false, 0, C_NULL, C_NULL)
  ghost = p4est_ghost_new(p4est, P4EST_CONNECT_FACE)
  p4est_mesh = p4est_mesh_new(p4est, ghost, P4EST_CONNECT_FACE)

  boundary_names = fill(:all, 4, prod(trees_per_dimension))

  return P4estMesh{NDIMS, RealT, NDIMS+2}(p4est, p4est_mesh, tree_node_coordinates, nodes,
                                          boundary_names, "", unsaved_changes)
end


# Create a new p4est_connectivity that represents a structured rectangle.
# Similar to p4est_connectivity_new_brick, but doesn't use Morton order.
# This order makes `calc_node_coordinates!` below and the calculation
# of `boundary_names` above easier but is irrelevant otherwise.
# TODO P4EST non-periodic
function connectivity_structured(cells_x, cells_y, periodicity)
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

  tree_to_tree = Matrix{p4est_topidx_t}(undef, 4, num_trees)
  tree_to_face = Matrix{Int8}(undef, 4, num_trees)

  for cell_y in 1:cells_y, cell_x in 1:cells_x
    tree = linear_indices[cell_x, cell_y]

    # Subtract 1 because p4est uses zero-based indexing
    # Negative x-direction
    if cell_x > 1
      tree_to_tree[1, tree] = linear_indices[cell_x - 1, cell_y] - 1
      tree_to_face[1, tree] = 1
    elseif periodicity[1]
      tree_to_tree[1, tree] = linear_indices[cells_x, cell_y] - 1
      tree_to_face[1, tree] = 1
    else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
      tree_to_tree[1, tree] = tree - 1
      tree_to_face[1, tree] = 0
    end

    # Positive x-direction
    if cell_x < cells_x
      tree_to_tree[2, tree] = linear_indices[cell_x + 1, cell_y] - 1
      tree_to_face[2, tree] = 0
    elseif periodicity[1]
      tree_to_tree[2, tree] = linear_indices[1, cell_y] - 1
      tree_to_face[2, tree] = 0
    else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
      tree_to_tree[2, tree] = tree - 1
      tree_to_face[2, tree] = 1
    end

    # Negative x-direction
    if cell_y > 1
      tree_to_tree[3, tree] = linear_indices[cell_x, cell_y - 1] - 1
      tree_to_face[3, tree] = 3
    elseif periodicity[2]
      tree_to_tree[3, tree] = linear_indices[cell_x, cells_y] - 1
      tree_to_face[3, tree] = 3
    else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
      tree_to_tree[3, tree] = tree - 1
      tree_to_face[3, tree] = 2
    end

    # Positive x-direction
    if cell_y < cells_y
      tree_to_tree[4, tree] = linear_indices[cell_x, cell_y + 1] - 1
      tree_to_face[4, tree] = 2
    elseif periodicity[2]
      tree_to_tree[4, tree] = linear_indices[cell_x, 1] - 1
      tree_to_face[4, tree] = 2
    else # Non-periodic boundary, tree and face point to themselves (zero-based indexing)
      tree_to_tree[4, tree] = tree - 1
      tree_to_face[4, tree] = 3
    end
  end

  tree_to_corner = C_NULL
  ctt_offset = Array{p4est_topidx_t}([0])

  corner_to_tree = C_NULL
  corner_to_corner = C_NULL

  p4est_connectivity_new_copy(num_vertices, num_trees, num_corners,
                              vertices, tree_to_vertex,
                              tree_to_tree, tree_to_face,
                              tree_to_corner, ctt_offset,
                              corner_to_tree, corner_to_corner)
end

@inline Base.ndims(::P4estMesh{NDIMS}) where NDIMS = NDIMS
@inline Base.real(::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline ntrees(mesh::P4estMesh) = mesh.p4est.trees.elem_count
@inline ncells(mesh::P4estMesh) = mesh.p4est_mesh.local_num_quadrants


function Base.show(io::IO, mesh::P4estMesh)
  print(io, "P4estMesh{", ndims(mesh), ", ", real(mesh), "}")
end


function Base.show(io::IO, ::MIME"text/plain", mesh::P4estMesh)
  if get(io, :compact, false)
    show(io, mesh)
  else
    setup = [
             "#trees" => ntrees(mesh),
             "current #cells" => ncells(mesh),
             "polydeg" => length(mesh.nodes),
            ]
    summary_box(io, "P4estMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) * "}", setup)
  end
end


# Calculate physical coordinates to which every node of the reference element is mapped
# This function assumes a structured mesh with trees in row order.
# 2D version
function calc_node_coordinates!(node_coordinates::AbstractArray{RealT, 4},
                                mapping, trees_per_dimension, nodes) where RealT
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


# Calculate physical coordinates to which every node of the reference element is mapped
# This function assumes a structured mesh with trees in row order.
# 2D version
function calc_node_coordinates!(node_coordinates::AbstractArray{RealT, 4},
                                vertices::AbstractArray, tree_to_vertex) where RealT
  for tree in 1:size(tree_to_vertex, 2)
    # Tree vertices are stored in Z-order, ignore z-coordinate in 2D
    tree_node_coordinates[:, 1, 1, tree] = vertices[1:2, tree_to_vertex[1, tree]]
    tree_node_coordinates[:, 2, 1, tree] = vertices[1:2, tree_to_vertex[2, tree]]
    tree_node_coordinates[:, 1, 2, tree] = vertices[1:2, tree_to_vertex[3, tree]]
    tree_node_coordinates[:, 2, 2, tree] = vertices[1:2, tree_to_vertex[4, tree]]
  end
end
