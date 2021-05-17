# Initialize data structures in element container
function init_elements!(elements, mesh::P4estMesh{2}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, jacobian_matrix,
          contravariant_vectors, inverse_jacobian = elements

  # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
  for element in 1:prod(size(mesh))
    calc_node_coordinates!(node_coordinates, element, mesh, basis)

    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

    calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
  end

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
function calc_node_coordinates!(node_coordinates, element,
                                mesh::P4estMesh{2},
                                basis::LobattoLegendreBasis)
  # TODO Interpolate for different bases and refined mesh
  for j in eachnode(basis), i in eachnode(basis), dim in 1:2
    node_coordinates[dim, i, j, element] = mesh.tree_node_coordinates[dim, i, j, element]
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
