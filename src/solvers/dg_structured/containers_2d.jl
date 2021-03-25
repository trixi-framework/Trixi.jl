function init_elements!(elements, mesh::StructuredMesh{2, RealT}, nodes) where {RealT}
  n_nodes = length(nodes)

  @unpack coordinates_min, coordinates_max = mesh

  linear_indices = LinearIndices(size(mesh))

  # Get cell length
  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)
  dy = (coordinates_max[2] - coordinates_min[2]) / size(mesh, 2)

  inverse_jacobian = 4/(dx * dy)

  # Calculate inverse Jacobian and node coordinates
  for cell_x in 1:size(mesh, 1), cell_y in 1:size(mesh, 2)
    element = linear_indices[cell_x, cell_y]

    # Calculate node coordinates
    cell_x_offset = coordinates_min[1] + (cell_x-1) * dx + dx/2
    cell_y_offset = coordinates_min[2] + (cell_y-1) * dy + dy/2

    for j in 1:n_nodes, i in 1:n_nodes
      elements.node_coordinates[1, i, j, element] = cell_x_offset + dx/2 * nodes[i]
      elements.node_coordinates[2, i, j, element] = cell_y_offset + dy/2 * nodes[j]
    end

    elements.inverse_jacobian[element] = inverse_jacobian
  end

  # Neighbors in x-direction
  for cell_y in 1:size(mesh, 2)
    # Inner elements
    for cell_x in 2:size(mesh, 1)
      element = linear_indices[cell_x, cell_y]
      elements.left_neighbors[1, element] = linear_indices[cell_x - 1, cell_y]
    end

    # Periodic boundary
    elements.left_neighbors[1, linear_indices[1, cell_y]] = linear_indices[end, cell_y]
  end

  # Neighbors in y-direction
  for cell_x in 1:size(mesh, 1)
    # Inner elements
    for cell_y in 2:size(mesh, 2)
      element = linear_indices[cell_x, cell_y]
      elements.left_neighbors[2, element] = linear_indices[cell_x, cell_y - 1]
    end

    # Periodic boundary
    elements.left_neighbors[2, linear_indices[cell_x, 1]] = linear_indices[cell_x, end]
  end

  return nothing
end
