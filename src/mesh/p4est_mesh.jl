# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    P4estMesh{NDIMS} <: AbstractMesh{NDIMS}

An unstructured curved mesh based on trees that uses the C library p4est
to manage trees and mesh refinement.

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
mutable struct P4estMesh{NDIMS, RealT<:Real, NDIMSP2, NNODES} <: AbstractMesh{NDIMS}
  p4est                 ::Ptr{p4est_t}
  # Coordinates at the nodes specified by the tensor product of `nodes` (NDIMS times).
  # This specifies the geometry interpolation for each tree.
  tree_node_coordinates ::Array{RealT, NDIMSP2} # [dimension, i, j, k, tree]
  nodes                 ::SVector{NNODES, RealT}
  boundary_names        ::Array{Symbol, 2}      # [face direction, tree]
  current_filename      ::String
  unsaved_changes       ::Bool

  function P4estMesh{NDIMS}(p4est, tree_node_coordinates, nodes, boundary_names,
                            current_filename, unsaved_changes) where NDIMS
    mesh = new{NDIMS, eltype(tree_node_coordinates), NDIMS+2, length(nodes)}(
      p4est, tree_node_coordinates,
      nodes, boundary_names, current_filename, unsaved_changes)

    # Destroy p4est structs when the mesh is garbage collected
    finalizer(destroy_mesh, mesh)

    return mesh
  end
end


function destroy_mesh(mesh)
  conn = mesh.p4est.connectivity
  p4est_destroy(mesh.p4est)
  p4est_connectivity_destroy(conn)
end


"""
    P4estMesh(trees_per_dimension; polydeg,
              mapping=nothing, faces=nothing, coordinates_min=nothing, coordinates_max=nothing,
              RealT=Float64, initial_refinement_level=0, periodicity=true, unsaved_changes=true)

Create a structured curved `P4estMesh` of the specified size.

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
  calc_tree_node_coordinates!(tree_node_coordinates, nodes, mapping, trees_per_dimension)

  # p4est_connectivity_new_brick has trees in Morton order, so use our own function for this
  conn = connectivity_structured(trees_per_dimension..., periodicity)
  # Use Int-Vector of size 2 as quadrant user data
  p4est = p4est_new_ext(0, conn, 0, initial_refinement_level, true, 2 * sizeof(Int), C_NULL, C_NULL)

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

  return P4estMesh{NDIMS}(p4est, tree_node_coordinates, nodes,
                          boundary_names, "", unsaved_changes)
end


"""
    P4estMesh(meshfile::String;
              mapping=nothing, polydeg=1, RealT=Float64,
              initial_refinement_level=0, unsaved_changes=true)

Import an uncurved, unstructured, conforming mesh from an Abaqus mesh file (`.inp`),
map the mesh with the specified mapping, and create a `P4estMesh` from the curved mesh.

Cells in the mesh file will be imported as trees in the `P4estMesh`.

# Arguments
- `meshfile::String`: an uncurved Abaqus mesh file that can be imported by p4est.
- `mapping`: a function of `NDIMS` variables to describe the mapping that transforms
             the imported mesh to the physical domain. Use `nothing` for the identity map.
- `polydeg::Integer`: polynomial degree used to store the geometry of the mesh.
                      The mapping will be approximated by an interpolation polynomial
                      of the specified degree for each tree.
                      The default of `1` creates an uncurved geometry. Use a higher value if the mapping
                      will curve the imported uncurved mesh.
- `RealT::Type`: the type that should be used for coordinates.
- `initial_refinement_level::Integer`: refine the mesh uniformly to this level before the simulation starts.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
"""
function P4estMesh(meshfile::String;
                   mapping=nothing, polydeg=1, RealT=Float64,
                   initial_refinement_level=0, unsaved_changes=true)
  # TODO P4EST
  NDIMS = 2

  # Prevent p4est from crashing Julia if the file doesn't exist
  @assert isfile(meshfile)

  conn = p4est_connectivity_read_inp(meshfile)

  # These need to be of the type Int for unsafe_wrap below to work
  n_trees::Int = conn.num_trees
  n_vertices::Int = conn.num_vertices

  vertices        = unsafe_wrap(Array, conn.vertices, (3, n_vertices))
  tree_to_vertex  = unsafe_wrap(Array, conn.tree_to_vertex, (4, n_trees))

  basis = LobattoLegendreBasis(RealT, polydeg)
  nodes = basis.nodes

  tree_node_coordinates = Array{RealT, NDIMS+2}(undef, NDIMS,
                                                ntuple(_ -> length(nodes), NDIMS)...,
                                                n_trees)
  calc_tree_node_coordinates!(tree_node_coordinates, nodes, mapping, vertices, tree_to_vertex)

  # Use Int-Vector of size 2 as quadrant user data
  p4est = p4est_new_ext(0, conn, 0, initial_refinement_level, true, 2 * sizeof(Int), C_NULL, C_NULL)

  # There's no simple and generic way to distinguish boundaries. Name all of them :all.
  boundary_names = fill(:all, 4, n_trees)

  return P4estMesh{NDIMS}(p4est, tree_node_coordinates, nodes,
                          boundary_names, "", unsaved_changes)
end


# Create a new p4est_connectivity that represents a structured rectangle.
# Similar to p4est_connectivity_new_brick, but doesn't use Morton order.
# This order makes `calc_tree_node_coordinates!` below and the calculation
# of `boundary_names` above easier but is irrelevant otherwise.
function connectivity_structured(cells_x, cells_y, periodicity)
  linear_indices = LinearIndices((cells_x, cells_y))

  # Vertices represent the coordinates of the forest. This is used by p4est
  # to write VTK files.
  # Trixi doesn't use p4est's coordinates, so the vertices can be empty.
  n_vertices = 0
  n_trees = cells_x * cells_y
  # No corner connectivity is needed
  n_corners = 0
  vertices = C_NULL
  tree_to_vertex = C_NULL

  tree_to_tree = Matrix{p4est_topidx_t}(undef, 4, n_trees)
  tree_to_face = Matrix{Int8}(undef, 4, n_trees)

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
  # p4est docs: "in trivial cases it is just a pointer to a p4est_topix value of 0."
  # We don't need corner connectivity, so this is a trivial case.
  ctt_offset = Array{p4est_topidx_t}([0])

  corner_to_tree = C_NULL
  corner_to_corner = C_NULL

  p4est_connectivity_new_copy(n_vertices, n_trees, n_corners,
                              vertices, tree_to_vertex,
                              tree_to_tree, tree_to_face,
                              tree_to_corner, ctt_offset,
                              corner_to_tree, corner_to_corner)
end

@inline Base.ndims(::P4estMesh{NDIMS}) where NDIMS = NDIMS
@inline Base.real(::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

@inline ntrees(mesh::P4estMesh) = mesh.p4est.trees.elem_count
@inline ncells(mesh::P4estMesh) = mesh.p4est.global_num_quadrants


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
             "polydeg" => length(mesh.nodes) - 1,
            ]
    summary_box(io, "P4estMesh{" * string(ndims(mesh)) * ", " * string(real(mesh)) * "}", setup)
  end
end


# Calculate physical coordinates of each node.
# This function assumes a structured mesh with trees in row order.
# 2D version
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4},
                                     nodes, mapping, trees_per_dimension)
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


# Calculate physical coordinates of each node.
# Extract corners of each tree, interpolate to requested interpolation nodes,
# map the resulting coordinates with the specified mapping.
# 2D version
function calc_tree_node_coordinates!(node_coordinates::AbstractArray{RealT, 4},
                                     nodes, mapping,
                                     vertices, tree_to_vertex) where RealT
  nodes_in = [-1.0, 1.0]
  matrix = polynomial_interpolation_matrix(nodes_in, nodes)
  data_in = Array{RealT, 3}(undef, 2, 2, 2)
  tmp1 = zeros(RealT, 2, length(nodes), length(nodes_in))

  for tree in 1:size(tree_to_vertex, 2)
    # Tree vertices are stored in Z-order, ignore z-coordinate in 2D, zero-based indexing
    data_in[:, 1, 1] .= vertices[1:2, tree_to_vertex[1, tree] + 1]
    data_in[:, 2, 1] .= vertices[1:2, tree_to_vertex[2, tree] + 1]
    data_in[:, 1, 2] .= vertices[1:2, tree_to_vertex[3, tree] + 1]
    data_in[:, 2, 2] .= vertices[1:2, tree_to_vertex[4, tree] + 1]

    multiply_dimensionwise!(
      view(node_coordinates, :, :, :, tree),
      matrix, matrix,
      data_in,
      tmp1
    )
  end

  map_node_coordinates!(node_coordinates, mapping)
end

function map_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4}, mapping)
  for tree in axes(node_coordinates, 4),
      j in axes(node_coordinates, 3),
      i in axes(node_coordinates, 2)

    node_coordinates[:, i, j, tree] .= mapping(node_coordinates[1, i, j, tree],
                                               node_coordinates[2, i, j, tree])
  end

  return node_coordinates
end

function map_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4}, mapping::Nothing)
  return node_coordinates
end


function init_fn(p4est, which_tree, quadrant)
  # Unpack quadrant's user data ([global quad ID, controller_value])
  ptr = Ptr{Int}(quadrant.p.user_data)

  # Initialize quad ID as -1 and controller_value as 0 (don't refine or coarsen)
  unsafe_store!(ptr, -1, 1)
  unsafe_store!(ptr, 0, 2)

  return nothing
end

function refine_fn(p4est, which_tree, quadrant)
  # Controller value has been copied to the quadrant's user data storage before.
  # Unpack quadrant's user data ([global quad ID, controller_value]).
  ptr = Ptr{Int}(quadrant.p.user_data)
  controller_value = unsafe_load(ptr, 2)

  if controller_value > 0
    # return true (refine)
    return Cint(1)
  else
    # return false (don't refine)
    return Cint(0)
  end
end

# Refine marked cells and rebalance forest.
# Return a list of all cells that have been refined during refinement or rebalancing.
function refine!(mesh::P4estMesh)
  # Copy original element IDs to quad user data storage
  original_n_cells = ncells(mesh)
  save_original_ids(mesh)

  init_fn_c = @cfunction(init_fn, Cvoid,
    (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))

  # Refine marked cells
  refine_fn_c = @cfunction(refine_fn, Cint,
    (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))
  @trixi_timeit timer() "refine" p4est_refine(mesh.p4est, false, refine_fn_c, init_fn_c)

  @trixi_timeit timer() "rebalance" p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn_c)
  # Due to a bug in p4est, the forest needs to be rebalanced twice sometimes
  # See https://github.com/cburstedde/p4est/issues/112
  @trixi_timeit timer() "rebalance" p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn_c)

  return collect_changed_cells(mesh, original_n_cells)
end


function coarsen_fn(p4est, which_tree, quadrants_ptr)
  quadrants = unsafe_wrap(Array, quadrants_ptr, 4)

  # Controller value has been copied to the quadrant's user data storage before.
  # Load controller value from quadrant's user data ([global quad ID, controller_value]).
  controller_value(i) = unsafe_load(Ptr{Int}(quadrants[i].p.user_data), 2)

  # p4est calls this function for each 2^ndims quads that could be coarsened to a single one.
  # Only coarsen if all these 2^ndims quads have been marked for coarsening.
  if all(i -> controller_value(i) < 0, eachindex(quadrants))
    # return true (coarsen)
    return Cint(1)
  else
    # return false (don't coarsen)
    return Cint(0)
  end
end

# Coarsen marked cells if the forest will stay balanced.
# Return a list of all cells that have been coarsened.
function coarsen!(mesh::P4estMesh)
  # Copy original element IDs to quad user data storage
  original_n_cells = ncells(mesh)
  save_original_ids(mesh)

  # Coarsen marked cells
  coarsen_fn_c = @cfunction(coarsen_fn, Cint,
    (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{Ptr{p4est_quadrant_t}}))
  init_fn_c = @cfunction(init_fn, Cvoid,
    (Ptr{p4est_t}, Ptr{p4est_topidx_t}, Ptr{p4est_quadrant_t}))

  @trixi_timeit timer() "coarsen!" p4est_coarsen(mesh.p4est, false, coarsen_fn_c, init_fn_c)

  # IDs of newly created cells (one-based)
  new_cells = collect_new_cells(mesh)
  # Old IDs of cells that have been coarsened (one-based)
  coarsened_cells_vec = collect_changed_cells(mesh, original_n_cells)
  # 2^ndims changed cells should have been coarsened to one new cell.
  # This matrix will store the IDs of all cells that have been coarsened to cell new_cells[i]
  # in the i-th column.
  coarsened_cells = reshape(coarsened_cells_vec, 2^ndims(mesh), length(new_cells))

  # Save new original IDs to find out what changed after balancing
  intermediate_n_cells = ncells(mesh)
  save_original_ids(mesh)

  @trixi_timeit timer() "rebalance" p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn_c)
  # Due to a bug in p4est, the forest needs to be rebalanced twice sometimes
  # See https://github.com/cburstedde/p4est/issues/112
  @trixi_timeit timer() "rebalance" p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, init_fn_c)

  refined_cells = collect_changed_cells(mesh, intermediate_n_cells)

  # Some cells may have been coarsened even though they unbalanced the forest.
  # These cells have now been refined again by p4est_balance.
  # refined_cells contains the intermediate IDs (ID of coarse cell
  # between coarsening and balancing) of these cells.
  # Find original ID of each cell that has been coarsened and then refined again.
  for refined_cell in refined_cells
    # i-th cell of the ones that have been created by coarsening has been refined again
    i = findfirst(==(refined_cell), new_cells)

    # Remove IDs of the 2^NDIMS cells that have been coarsened to this cell
    coarsened_cells[:, i] .= -1
  end

  # Return all IDs of cells that have been coarsened but not refined again by balancing
  return coarsened_cells_vec[coarsened_cells_vec .>= 0]
end


# Copy global quad ID to quad's user data storage, will be called below
function save_original_id_iter_volume(info, user_data)
  # Load tree from global trees array, one-based indexing
  tree = unsafe_load_sc(p4est_tree_t, info.p4est.trees, info.treeid + 1)
  # Quadrant numbering offset of this quadrant
  offset = tree.quadrants_offset
  # Global quad ID
  quad_id = offset + info.quadid

  # Unpack quadrant's user data ([global quad ID, controller_value])
  ptr = Ptr{Int}(info.quad.p.user_data)
  # Save global quad ID
  unsafe_store!(ptr, quad_id, 1)

  return nothing
end

# Copy old element IDs to each quad's user data storage
function save_original_ids(mesh::P4estMesh)
  iter_volume_c = @cfunction(save_original_id_iter_volume, Cvoid, (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))

  p4est_iterate(mesh.p4est,
                C_NULL, # ghost layer
                C_NULL, # user data
                iter_volume_c, # iter_volume
                C_NULL, # iter_face
                C_NULL) # iter_corner
end


# Extract information about which cells have been changed
function collect_changed_iter_volume(info, user_data)
  # The original element ID has been saved to user_data before.
  # Load original quad ID from quad's user data ([global quad ID, controller_value]).
  quad_data_ptr = Ptr{Int}(info.quad.p.user_data)
  original_id = unsafe_load(quad_data_ptr, 1)

  # original_id of cells that have been newly created is -1
  if original_id >= 0
    # Unpack user_data = original_cells
    user_data_ptr = Ptr{Int}(user_data)

    # If quad has an original_id, it existed before refinement/coarsening,
    # and therefore wasn't changed.
    # Mark original_id as "not changed during refinement/coarsening" in original_cells
    unsafe_store!(user_data_ptr, 0, original_id + 1)
  end

  return nothing
end

function collect_changed_cells(mesh, original_n_cells)
  original_cells = collect(1:original_n_cells)

  # Iterate over all quads and set original cells that haven't been changed to zero
  iter_volume_c = @cfunction(collect_changed_iter_volume, Cvoid, (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))

  GC.@preserve original_cells begin
    p4est_iterate(mesh.p4est,
                  C_NULL, # ghost layer
                  pointer(original_cells),
                  iter_volume_c, # iter_volume
                  C_NULL, # iter_face
                  C_NULL) # iter_corner
  end

  # Changed cells are all that haven't been set to zero above
  changed_original_cells = original_cells[original_cells .> 0]

  return changed_original_cells
end


# Extract newly created cells
function collect_new_iter_volume(info, user_data)
  # The original element ID has been saved to user_data before.
  # Unpack quadrant's user data ([global quad ID, controller_value]).
  quad_data_ptr = Ptr{Int}(info.quad.p.user_data)
  original_id = unsafe_load(quad_data_ptr, 1)

  # original_id of cells that have been newly created is -1
  if original_id < 0
    # Load tree from global trees array, one-based indexing
    tree = unsafe_load_sc(p4est_tree_t, info.p4est.trees, info.treeid + 1)
    # Quadrant numbering offset of this quadrant
    offset = tree.quadrants_offset
    # Global quad ID
    quad_id = offset + info.quadid

    # Unpack user_data = original_cells
    user_data_ptr = Ptr{Int}(user_data)

    # Mark cell as "newly created during refinement/coarsening/balancing"
    unsafe_store!(user_data_ptr, 1, quad_id + 1)
  end

  return nothing
end

function collect_new_cells(mesh)
  cell_is_new = zeros(Int, ncells(mesh))

  # Iterate over all quads and set original cells that have been changed to one
  iter_volume_c = @cfunction(collect_new_iter_volume, Cvoid, (Ptr{p4est_iter_volume_info_t}, Ptr{Cvoid}))

  GC.@preserve cell_is_new begin
    p4est_iterate(mesh.p4est,
                  C_NULL, # ghost layer
                  pointer(cell_is_new),
                  iter_volume_c, # iter_volume
                  C_NULL, # iter_face
                  C_NULL) # iter_corner
  end

  # Changed cells are all that haven't been set to zero above
  new_cells = findall(==(1), cell_is_new)

  return new_cells
end


end # @muladd
