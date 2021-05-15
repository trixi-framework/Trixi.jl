# Initialize data structures in element container
function init_elements!(elements, mesh::P4estMesh{2}, basis::LobattoLegendreBasis)
  @unpack node_coordinates, jacobian_matrix,
          contravariant_vectors, inverse_jacobian = elements

  linear_indices = LinearIndices(size(mesh))

  # Calculate node coordinates, Jacobian matrix, and inverse Jacobian determinant
  for cell_y in 1:size(mesh, 2), cell_x in 1:size(mesh, 1)
    element = linear_indices[cell_x, cell_y]

    calc_node_coordinates!(node_coordinates, element, cell_x, cell_y, mesh.mapping, mesh, basis)

    calc_jacobian_matrix!(jacobian_matrix, element, node_coordinates, basis)

    calc_contravariant_vectors!(contravariant_vectors, element, jacobian_matrix)

    calc_inverse_jacobian!(inverse_jacobian, element, jacobian_matrix)
  end

  return nothing
end


# Calculate physical coordinates to which every node of the reference element is mapped
# `mesh.mapping` is passed as an additional argument for type stability (function barrier)
function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, mapping,
                                mesh::P4estMesh{2},
                                basis::LobattoLegendreBasis)
  @unpack nodes = basis

  # Get cell length in reference mesh
  dx = 2 / size(mesh, 1)
  dy = 2 / size(mesh, 2)

  # Calculate node coordinates of reference mesh
  cell_x_offset = -1 + (cell_x-1) * dx + dx/2
  cell_y_offset = -1 + (cell_y-1) * dy + dy/2

  for j in eachnode(basis), i in eachnode(basis)
    # node_coordinates are the mapped reference node_coordinates
    node_coordinates[:, i, j, element] .= mapping(cell_x_offset + dx/2 * nodes[i],
                                                  cell_y_offset + dy/2 * nodes[j])
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
