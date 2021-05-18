# Initialize data structures in element container
function init_elements!(elements, mesh::P4estMesh{2}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, jacobian_matrix,
          contravariant_vectors, inverse_jacobian = elements

  calc_node_coordinates!(node_coordinates, mesh, basis)

  # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
  for element in 1:mesh.p4est_mesh.local_num_quadrants
    # calc_node_coordinates!(node_coordinates, element, mesh, basis)

    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

    calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
  end

  return nothing
end


function convert_sc_array(type, sc_array)
  n_elements = sc_array.elem_count
  element_size = sc_array.elem_size

  @assert element_size == sizeof(type)

  return [unsafe_wrap(type, sc_array.array + element_size * i) for i in 0:n_elements-1]
end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates,
                                mesh::P4estMesh{2},
                                basis::LobattoLegendreBasis)
  # u_tmp1 = Array{real(mesh), 3}(undef, 2, nnodes(dg), nnodes(dg))
  # u_tmp2 = Array{real(mesh), 3}(undef, 2, nnodes(dg), nnodes(dg))

  P4EST_ROOT_LEN = 1 << P4EST_MAXLEVEL
  P4EST_QUADRANT_LEN(l) = 1 << (P4EST_MAXLEVEL - l)

  trees = convert_sc_array(p4est_tree_t, mesh.p4est.trees)

  for tree in eachindex(trees)
    offset = trees[tree].quadrants_offset
    quadrants = convert_sc_array(p4est_quadrant_t, trees[tree].quadrants)

    for i in eachindex(quadrants)
      element = offset + i
      quad = quadrants[i]

      quad_length = P4EST_QUADRANT_LEN(quad.level) / P4EST_ROOT_LEN

      nodes_out_x = 2 * quad_length * (1/2 * (basis.nodes .+ 1) .+ quad.x) .- 1
      nodes_out_y = 2 * quad_length * (1/2 * (basis.nodes .+ 1) .+ quad.y) .- 1

      matrix1 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_x)
      matrix2 = polynomial_interpolation_matrix(mesh.nodes, nodes_out_y)

      multiply_dimensionwise!(
        view(node_coordinates, :, :, :, element),
        matrix1, matrix2,
        view(mesh.tree_node_coordinates, :, :, :, tree)
      )
    end
  end
end


function iter_face(info, user_data)
  ptr = Ptr{Any}(user_data)
  data_array = unsafe_wrap(Array, ptr, 3)
  interfaces = data_array[1]
  interface_id = data_array[2]
  data_array[2] += 1
  mesh = data_array[3]

  @assert info.sides.elem_count == 2 "Boundaries are not supported yet"

  sides = convert_sc_array(p4est_iter_face_side_t, info.sides)
  trees = convert_sc_array(p4est_tree_t, mesh.p4est.trees)
  offsets = [trees[sides[1].treeid + 1].quadrants_offset, trees[sides[2].treeid + 1].quadrants_offset]

  if sides[1].is_hanging != 0 || sides[2].is_hanging != 0
    error("Hanging nodes are not supported yet")
  end

  local_quad_ids = [sides[1].is.full.quadid, sides[2].is.full.quadid]
  quad_ids = offsets + local_quad_ids
  faces = [sides[1].face, sides[2].face]
  local_num_quadrants = mesh.p4est_mesh.local_num_quadrants

  quad_to_face = unsafe_wrap(Array, mesh.p4est_mesh.quad_to_face, 4 * local_num_quadrants)
  orientations = [quad_to_face[quad_ids[2] * 4 + faces[2] + 1],
                  quad_to_face[quad_ids[1] * 4 + faces[1] + 1]] - faces

  interfaces.element_ids[1, interface_id] = quad_ids[1] + 1
  interfaces.element_ids[2, interface_id] = quad_ids[2] + 1

  for side in 1:2
    if faces[side] == 0
      # Copy to left interface
      interfaces.node_indices[side, interface_id] = (:one, :i)
    elseif faces[side] == 1
      # Copy to right interface
      interfaces.node_indices[side, interface_id] = (:end, :i)
    elseif faces[side] == 2
      # Copy to bottom interface
      interfaces.node_indices[side, interface_id] = (:i, :one)
    elseif faces[side] == 3
      # Copy to top interface
      interfaces.node_indices[side, interface_id] = (:i, :end)
    else
      error("Something went horribly wrong")
    end
  end

  return nothing
end


function init_interfaces!(interfaces, mesh::P4estMesh{2})
  iter_face_c = @cfunction(iter_face, Cvoid, (Ptr{p4est_iter_face_info_t}, Ptr{Cvoid}))
  p4est_iterate(mesh.p4est, C_NULL, Base.unsafe_convert(Ptr{Nothing}, [interfaces, 1, mesh]), C_NULL, iter_face_c, C_NULL)

  return interfaces
end
