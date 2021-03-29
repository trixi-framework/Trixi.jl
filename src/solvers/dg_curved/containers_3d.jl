function init_elements!(elements, mesh::CurvedMesh{3}, nodes)
  n_nodes = length(nodes)

  @unpack coordinates_min, coordinates_max = mesh

  linear_indices = LinearIndices(size(mesh))

  # Get cell length
  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)
  dy = (coordinates_max[2] - coordinates_min[2]) / size(mesh, 2)
  dz = (coordinates_max[3] - coordinates_min[3]) / size(mesh, 3)

  inverse_jacobian = 8/(dx * dy * dz)

  # Calculate inverse Jacobian and node coordinates
  for cell_x in 1:size(mesh, 1), cell_y in 1:size(mesh, 2), cell_z in 1:size(mesh, 3)
    element = linear_indices[cell_x, cell_y, cell_z]

    # Calculate node coordinates
    cell_x_offset = coordinates_min[1] + (cell_x-1) * dx + dx/2
    cell_y_offset = coordinates_min[2] + (cell_y-1) * dy + dy/2
    cell_z_offset = coordinates_min[3] + (cell_z-1) * dz + dz/2

    for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
      elements.node_coordinates[1, i, j, k, element] = cell_x_offset + dx/2 * nodes[i]
      elements.node_coordinates[2, i, j, k, element] = cell_y_offset + dy/2 * nodes[j]
      elements.node_coordinates[3, i, j, k, element] = cell_z_offset + dz/2 * nodes[k]
    end

    elements.inverse_jacobian[element] = inverse_jacobian
  end

  # Neighbors in x-direction
  for cell_y in 1:size(mesh, 2), cell_z in 1:size(mesh, 3)
    # Inner elements
    for cell_x in 2:size(mesh, 1)
      element = linear_indices[cell_x, cell_y, cell_z]
      elements.left_neighbors[1, element] = linear_indices[cell_x - 1, cell_y, cell_z]
    end

    # Periodic boundary
    elements.left_neighbors[1, linear_indices[1, cell_y, cell_z]] = linear_indices[end, cell_y, cell_z]
  end

  # Neighbors in y-direction
  for cell_x in 1:size(mesh, 1), cell_z in 1:size(mesh, 3)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y, cell_z]
      elements.left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1, cell_z]
    end

    # Periodic boundary
    elements.left_neighbors[2, linear_indices[cell_x, 1, cell_z]] = linear_indices[cell_x, end, cell_z]
  end

  # Neighbors in z-direction
  for cell_x in 1:size(mesh, 1), cell_y in 1:size(mesh, 2)
    # Inner elements
    for cell_z in 2:size(mesh, 3)
      element = linear_indices[cell_x, cell_y, cell_z]
      elements.left_neighbors[3, element] = linear_indices[cell_x, cell_y, cell_z - 1]
    end

    # Periodic boundary
    elements.left_neighbors[3, linear_indices[cell_x, cell_y, 1]] = linear_indices[cell_x, cell_y, end]
  end

  return nothing
end
