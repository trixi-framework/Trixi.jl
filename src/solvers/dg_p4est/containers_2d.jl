# Initialize data structures in element container
function init_elements!(elements, mesh::P4estMesh{2}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, jacobian_matrix,
          contravariant_vectors, inverse_jacobian = elements

  calc_node_coordinates!(node_coordinates, mesh, basis.nodes)

  for element in 1:ncells(mesh)
    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

    calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
  end

  return nothing
end


# Interpolate tree_node_coordinates to each quadrant
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{2},
                                nodes)
  # Hanging nodes will cause holes in the mesh if its polydeg is higher
  # than the polydeg of the solver.
  @assert length(nodes) >= length(mesh.nodes) "The solver can't have a lower polydeg than the mesh"

  tmp1 = zeros(real(mesh), 2, length(nodes), length(mesh.nodes))
  baryweights_in = barycentric_weights(mesh.nodes)
  matrix1 = Matrix{real(mesh)}(undef, length(mesh.nodes), length(mesh.nodes))
  matrix2 = similar(matrix1)

  # Macros from p4est
  p4est_root_len = 1 << P4EST_MAXLEVEL
  p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

  trees = unsafe_wrap_sc(p4est_tree_t, mesh.p4est.trees)

  for tree in eachindex(trees)
    offset = trees[tree].quadrants_offset
    quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree].quadrants)

    for i in eachindex(quadrants)
      element = offset + i
      quad = quadrants[i]

      quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

      nodes_out_x = 2 * (quad_length * 1/2 * (nodes .+ 1) .+ quad.x / p4est_root_len) .- 1
      nodes_out_y = 2 * (quad_length * 1/2 * (nodes .+ 1) .+ quad.y / p4est_root_len) .- 1
      polynomial_interpolation_matrix!(matrix1, mesh.nodes, nodes_out_x, baryweights_in)
      polynomial_interpolation_matrix!(matrix2, mesh.nodes, nodes_out_y, baryweights_in)

      multiply_dimensionwise!(
        view(node_coordinates, :, :, :, element),
        matrix1, matrix2,
        view(mesh.tree_node_coordinates, :, :, :, tree),
        tmp1
      )
    end
  end

  return node_coordinates
end


# Iterate over all interfaces and extract inner interface data to interface container
# This function will be passed to p4est in init_interfaces! below
function init_interfaces_iter_face(info, user_data)
  if info.sides.elem_count != 2
    # Not an inner interface
    return nothing
  end

  sides = (unsafe_load_sc(p4est_iter_face_side_t, info.sides, 1),
           unsafe_load_sc(p4est_iter_face_side_t, info.sides, 2))

  if sides[1].is_hanging == true || sides[2].is_hanging == true
    # Mortar, no normal interface
    return nothing
  end

  # Unpack user_data = [interfaces, interface_id, mesh] and increment interface_id
  ptr = Ptr{Any}(user_data)
  data_array = unsafe_wrap(Array, ptr, 3)
  interfaces = data_array[1]
  interface_id = data_array[2]
  data_array[2] = interface_id + 1
  mesh = data_array[3]

  # Function barrier because the unpacked user_data above is type-unstable
  init_interfaces_iter_face_inner(info, sides, interfaces, interface_id, mesh)
end

# Function barrier for type stability
function init_interfaces_iter_face_inner(info, sides, interfaces, interface_id, mesh)
  # Load local trees from global trees array, one-based indexing
  trees = (unsafe_load_sc(p4est_tree_t, mesh.p4est.trees, sides[1].treeid + 1),
           unsafe_load_sc(p4est_tree_t, mesh.p4est.trees, sides[2].treeid + 1))
  # Quadrant numbering offsets of the quadrants at this interface
  offsets = SVector(trees[1].quadrants_offset,
                    trees[2].quadrants_offset)

  local_quad_ids = SVector(sides[1].is.full.quadid, sides[2].is.full.quadid)
  # Global IDs of the neighboring quads
  quad_ids = offsets + local_quad_ids

  # Write data to interfaces container
  # p4est uses zero-based indexing; convert to one-based indexing
  interfaces.element_ids[1, interface_id] = quad_ids[1] + 1
  interfaces.element_ids[2, interface_id] = quad_ids[2] + 1

  # Face at which the interface lies
  faces = (sides[1].face, sides[2].face)

  # Relative orientation of the two cell faces,
  # 0 for aligned coordinates, 1 for reversed coordinates.
  orientation = info.orientation

  # Iterate over primary and secondary element
  for side in 1:2
    # Align interface in positive coordinate direction of primary element.
    # For orientation == 1, the secondary element needs to be indexed backwards
    # relative to the interface.
    if side == 1 || orientation == 0
      # Forward indexing
      i = :i
    else
      # Backward indexing
      i = :i_backwards
    end

    if faces[side] == 0
      # Index face in negative x-direction
      interfaces.node_indices[side, interface_id] = (:one, i)
    elseif faces[side] == 1
      # Index face in positive x-direction
      interfaces.node_indices[side, interface_id] = (:end, i)
    elseif faces[side] == 2
      # Index face in negative y-direction
      interfaces.node_indices[side, interface_id] = (i, :one)
    else # faces[side] == 3
      # Index face in positive y-direction
      interfaces.node_indices[side, interface_id] = (i, :end)
    end
  end

  return nothing
end

function init_interfaces!(interfaces, mesh::P4estMesh{2})
  # Let p4est iterate over all interfaces and call init_interfaces_iter_face
  iter_face_c = @cfunction(init_interfaces_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
  user_data = [interfaces, 1, mesh]

  iterate_faces(mesh, iter_face_c, user_data)

  return interfaces
end


# Iterate over all interfaces and extract boundary data to boundary container
# This function will be passed to p4est in init_boundaries! below
function init_boundaries_iter_face(info, user_data)
  if info.sides.elem_count == 2
    # Not a boundary
    return nothing
  end

  # Unpack user_data = [boundaries, boundary_id, mesh] and increment boundary_id
  ptr = Ptr{Any}(user_data)
  data_array = unsafe_wrap(Array, ptr, 3)
  boundaries = data_array[1]
  boundary_id = data_array[2]
  data_array[2] += 1
  mesh = data_array[3]

  # Function barrier because the unpacked user_data above is type-unstable
  init_boundaries_iter_face_inner(info, boundaries, boundary_id, mesh)
end

# Function barrier for type stability
function init_boundaries_iter_face_inner(info, boundaries, boundary_id, mesh)
  # Extract boundary data
  side = unsafe_load_sc(p4est_iter_face_side_t, info.sides)
  # Load tree from global trees array, one-based indexing
  tree = unsafe_load_sc(p4est_tree_t, mesh.p4est.trees, side.treeid + 1)
  # Quadrant numbering offset of this quadrant
  offset = tree.quadrants_offset

  # Verify before accessing is.full, but this should never happen
  @assert side.is_hanging == false

  local_quad_id = side.is.full.quadid
  # Global ID of this quad
  quad_id = offset + local_quad_id

  # Write data to boundaries container
  # p4est uses zero-based indexing; convert to one-based indexing
  boundaries.element_ids[boundary_id] = quad_id + 1

  # Face at which the boundary lies
  face = side.face

  if face == 0
    # Index face in negative x-direction
    boundaries.node_indices[boundary_id] = (:one, :i)
  elseif face == 1
    # Index face in positive x-direction
    boundaries.node_indices[boundary_id] = (:end, :i)
  elseif face == 2
    # Index face in negative y-direction
    boundaries.node_indices[boundary_id] = (:i, :one)
  else # face == 3
    # Index face in positive y-direction
    boundaries.node_indices[boundary_id] = (:i, :end)
  end

  # One-based indexing
  boundaries.name[boundary_id] = mesh.boundary_names[face + 1, side.treeid + 1]

  return nothing
end

function init_boundaries!(boundaries, mesh::P4estMesh{2})
  # Let p4est iterate over all interfaces and call init_boundaries_iter_face
  iter_face_c = @cfunction(init_boundaries_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
  user_data = [boundaries, 1, mesh]

  iterate_faces(mesh, iter_face_c, user_data)

  return boundaries
end


# Iterate over all interfaces and extract mortar data to mortar container
# This function will be passed to p4est in init_mortars! below
function init_mortars_iter_face(info, user_data)
  if info.sides.elem_count != 2
    # Not an inner interface
    return nothing
  end

  sides = (unsafe_load_sc(p4est_iter_face_side_t, info.sides, 1),
           unsafe_load_sc(p4est_iter_face_side_t, info.sides, 2))

  if sides[1].is_hanging == false && sides[2].is_hanging == false
    # Normal interface, no mortar
    return nothing
  end

  # Unpack user_data = [mortars, mortar_id, mesh] and increment mortar_id
  ptr = Ptr{Any}(user_data)
  data_array = unsafe_wrap(Array, ptr, 3)
  mortars = data_array[1]
  mortar_id = data_array[2]
  data_array[2] += 1
  mesh = data_array[3]

  # Function barrier because the unpacked user_data above is type-unstable
  init_mortars_iter_face_inner(info, sides, mortars, mortar_id, mesh)
end

# Function barrier for type stability
function init_mortars_iter_face_inner(info, sides, mortars, mortar_id, mesh)
  # Load local trees from global trees array, one-based indexing
  trees = (unsafe_load_sc(p4est_tree_t, mesh.p4est.trees, sides[1].treeid + 1),
           unsafe_load_sc(p4est_tree_t, mesh.p4est.trees, sides[2].treeid + 1))
  # Quadrant numbering offsets of the quadrants at this interface
  offsets = SVector(trees[1].quadrants_offset,
                    trees[2].quadrants_offset)

  if sides[1].is_hanging == true
    # Left is small (1), right is large (2)
    small_large = [1, 2]

    local_small_quad_ids = sides[1].is.hanging.quadid
    # Global IDs of the two small quads
    small_quad_ids = offsets[1] .+ local_small_quad_ids

    # Just be sure before accessing is.full
    @assert sides[2].is_hanging == false
    large_quad_id = offsets[2] + sides[2].is.full.quadid
  else # sides[2].is_hanging == true
    # Left is large (2), right is small (1)
    small_large = [2, 1]

    local_small_quad_ids = sides[2].is.hanging.quadid
    # Global IDs of the two small quads
    small_quad_ids = offsets[2] .+ local_small_quad_ids

    # Just be sure before accessing is.full
    @assert sides[1].is_hanging == false
    large_quad_id = offsets[1] + sides[1].is.full.quadid
  end

  # Write data to mortar container, 1 and 2 are the small elements
  # p4est uses zero-based indexing; convert to one-based indexing
  mortars.element_ids[1, mortar_id] = small_quad_ids[1] + 1
  mortars.element_ids[2, mortar_id] = small_quad_ids[2] + 1
  # 3 is the large element
  mortars.element_ids[3, mortar_id] = large_quad_id + 1

  # Face at which the interface lies
  faces = [sides[1].face, sides[2].face]

  # Relative orientation of the two cell faces,
  # 0 for aligned coordinates, 1 for reversed coordinates.
  orientation = info.orientation

  for side in 1:2
    # Align mortar in positive coordinate direction of small side.
    # For orientation == 1, the large side needs to be indexed backwards
    # relative to the mortar.
    if small_large[side] == 1 || orientation == 0
      # Forward indexing for small side or orientation == 0
      i = :i
    else
      # Backward indexing for large side with reversed orientation
      i = :i_backwards
    end

    if faces[side] == 0
      # Index face in negative x-direction
      mortars.node_indices[small_large[side], mortar_id] = (:one, i)
    elseif faces[side] == 1
      # Index face in positive x-direction
      mortars.node_indices[small_large[side], mortar_id] = (:end, i)
    elseif faces[side] == 2
      # Index face in negative y-direction
      mortars.node_indices[small_large[side], mortar_id] = (i, :one)
    else # faces[side] == 3
      # Index face in positive y-direction
      mortars.node_indices[small_large[side], mortar_id] = (i, :end)
    end
  end

  return nothing
end

function init_mortars!(mortars, mesh::P4estMesh{2})
  iter_face_c = @cfunction(init_mortars_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
  user_data = [mortars, 1, mesh]

  iterate_faces(mesh, iter_face_c, user_data)

  return mortars
end
