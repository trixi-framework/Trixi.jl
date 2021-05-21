# Initialize data structures in element container
function init_elements!(elements, mesh::P4estMesh{2}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, jacobian_matrix,
          contravariant_vectors, inverse_jacobian = elements

  calc_node_coordinates!(node_coordinates, mesh, basis)

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
                                basis)
  tmp1 = zeros(real(mesh), 2, nnodes(basis), length(mesh.nodes))

  # Macros from p4est
  p4est_root_len = 1 << P4EST_MAXLEVEL
  p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

  trees = convert_sc_array(p4est_tree_t, mesh.p4est.trees)

  for tree in eachindex(trees)
    offset = trees[tree].quadrants_offset
    quadrants = convert_sc_array(p4est_quadrant_t, trees[tree].quadrants)

    for i in eachindex(quadrants)
      element = offset + i
      quad = quadrants[i]

      quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

      nodes_out_x = 2 * (quad_length * 1/2 * (basis.nodes .+ 1) .+ quad.x / p4est_root_len) .- 1
      nodes_out_y = 2 * (quad_length * 1/2 * (basis.nodes .+ 1) .+ quad.y / p4est_root_len) .- 1
      matrix1 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_x)
      matrix2 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_y)

      multiply_dimensionwise!(
        view(node_coordinates, :, :, :, element),
        matrix1, matrix2,
        view(mesh.tree_node_coordinates, :, :, :, tree),
        tmp1
      )
    end
  end
end


# Iterate over all interfaces and extract interface data to interface container
# This function will be passed to p4est in init_interfaces! below
function init_interfaces_iter_face(info, user_data)
  # Unpack user_data = [interfaces, interface_id, mesh] and increment interface_id
  ptr = Ptr{Any}(user_data)
  data_array = unsafe_wrap(Array, ptr, 3)
  interfaces = data_array[1]
  interface_id = data_array[2]
  data_array[2] += 1
  mesh = data_array[3]

  @assert info.sides.elem_count == 2 "Boundaries are not supported yet"

  # Extract interface data
  sides = convert_sc_array(p4est_iter_face_side_t, info.sides)
  # Global trees array
  trees = convert_sc_array(p4est_tree_t, mesh.p4est.trees)
  # Quadrant numbering offsets of the quadrants at this interface
  offsets = [trees[sides[1].treeid + 1].quadrants_offset,
             trees[sides[2].treeid + 1].quadrants_offset]

  if sides[1].is_hanging != 0 || sides[2].is_hanging != 0
    error("Hanging nodes are not supported yet")
  end

  local_quad_ids = [sides[1].is.full.quadid, sides[2].is.full.quadid]
  # Global IDs of the neighboring quads
  quad_ids = offsets + local_quad_ids
  # Face at which the interface lies
  faces = [sides[1].face, sides[2].face]

  quad_to_face = unsafe_wrap(Array, mesh.p4est_mesh.quad_to_face, 4 * ncells(mesh))
  # Relative orientation of the two cell faces,
  # 0 for aligned coordinates, 1 for reversed coordinates.
  # `quad_to_face` has one entry for each quadrant's face, +1 for one-based indexing.
  # The value in quad_to_face is `orientation * 4 + face`.
  orientation = (quad_to_face[quad_ids[1] * 4 + faces[1] + 1] - faces[2]) / 4

  # Write data to interfaces container
  interfaces.element_ids[1, interface_id] = quad_ids[1] + 1
  interfaces.element_ids[2, interface_id] = quad_ids[2] + 1

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
      i = :mi
    end

    if faces[side] == 0
      # Copy to left interface
      interfaces.node_indices[side, interface_id] = (:one, i)
    elseif faces[side] == 1
      # Copy to right interface
      interfaces.node_indices[side, interface_id] = (:end, i)
    elseif faces[side] == 2
      # Copy to bottom interface
      interfaces.node_indices[side, interface_id] = (i, :one)
    else # faces[side] == 3
      # Copy to top interface
      interfaces.node_indices[side, interface_id] = (i, :end)
    end
  end

  return nothing
end


function init_interfaces!(interfaces, mesh::P4estMesh{2})
  # Let p4est iterate over all interfaces and extract interface data to interface container
  iter_face_c = @cfunction(init_interfaces_iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
  user_data = [interfaces, 1, mesh]

  GC.@preserve user_data begin
    p4est_iterate(mesh.p4est,
                  C_NULL, # ghost layer
                  # user data [interfaces, interface_id, mesh]
                  pointer(user_data),
                  C_NULL, # iter_volume
                  iter_face_c, # iter_face
                  C_NULL) # iter_corner
  end

  return interfaces
end
