function init_elements!(elements, mesh::StructuredMesh{1, RealT}, nodes) where {RealT}
  n_nodes = length(nodes)

  @unpack coordinates_min, coordinates_max = mesh

  # Get cell length
  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)

  # Calculate inverse Jacobian as 1/(h/2)
  inverse_jacobian = 2/dx

  # Calculate inverse Jacobian and node coordinates
  for element_x in 1:size(mesh, 1)
    # Calculate node coordinates
    element_x_offset = coordinates_min[1] + (element_x-1) * dx + dx/2

    for i in 1:n_nodes
        elements.node_coordinates[1, i, element_x] = element_x_offset + dx/2 * nodes[i]
    end

    elements.inverse_jacobian[element_x] = inverse_jacobian

    # Boundary neighbors are overwritten below
    elements.left_neighbors[1, element_x] = element_x - 1
  end

  # Inner neighbors
  for element_x in 2:size(mesh, 1)
    elements.left_neighbors[1, element_x] = element_x - 1
  end

  # Periodic boundary
  elements.left_neighbors[1, 1] = size(mesh, 1)

  return nothing
end
