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


function init_interfaces!(interfaces, mesh::P4estMesh{2})
  linear_indices = LinearIndices(size(mesh))
  interface_id = 1

  # Neighbors in x-direction
  for cell_y in 1:size(mesh, 2)
    # Inner elements
    for cell_x in 2:size(mesh, 1)
      right_element = linear_indices[cell_x, cell_y]
      left_element = linear_indices[cell_x - 1, cell_y]

      interfaces.element_ids[1, interface_id] = left_element
      interfaces.element_ids[2, interface_id] = right_element

      # Copy to right interface
      interfaces.node_indices[1, interface_id] = (:end, :i)
      # Copy to left interface
      interfaces.node_indices[2, interface_id] = (:one, :i)

      interface_id += 1
    end

    if isperiodic(mesh, 1)
      # Periodic boundary
      right_element = linear_indices[1, cell_y]
      left_element = linear_indices[end, cell_y]

      interfaces.element_ids[1, interface_id] = left_element
      interfaces.element_ids[2, interface_id] = right_element

      # Copy to right interface
      interfaces.node_indices[1, interface_id] = (:end, :i)
      # Copy to left interface
      interfaces.node_indices[2, interface_id] = (:one, :i)

      interface_id += 1
    end
  end

  # Neighbors in y-direction
  for cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      right_element = linear_indices[cell_x, cell_y]
      left_element = linear_indices[cell_x, cell_y - 1]

      interfaces.element_ids[1, interface_id] = left_element
      interfaces.element_ids[2, interface_id] = right_element

      # Copy to right interface
      interfaces.node_indices[1, interface_id] = (:i, :end)
      # Copy to left interface
      interfaces.node_indices[2, interface_id] = (:i, :one)

      interface_id += 1
    end

    if isperiodic(mesh, 2)
      # Periodic boundary
      right_element = linear_indices[cell_x, 1]
      left_element = linear_indices[cell_x, end]

      interfaces.element_ids[1, interface_id] = left_element
      interfaces.element_ids[2, interface_id] = right_element

      # Copy to right interface
      interfaces.node_indices[1, interface_id] = (:i, :end)
      # Copy to left interface
      interfaces.node_indices[2, interface_id] = (:i, :one)

      interface_id += 1
    end
  end

  return interfaces
end
