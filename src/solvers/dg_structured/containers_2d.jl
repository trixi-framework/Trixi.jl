function init_elements!(elements, mesh::StructuredMesh{2, RealT}, nodes) where {RealT}
  n_nodes = length(nodes)

  @unpack size, coordinates_min, coordinates_max = mesh

  # Get cell length
  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]
  dy = (coordinates_max[2] - coordinates_min[2]) / size[2]

  inverse_jacobian = 4/(dx * dy)

  # Calculate inverse Jacobian and node coordinates
  for element_x in 1:size[1], element_y in 1:size[2]
    # Calculate node coordinates
    element_x_offset = coordinates_min[1] + (element_x-1) * dx + dx/2
    element_y_offset = coordinates_min[2] + (element_y-1) * dy + dy/2

    node_coordinates = Array{RealT, 3}(undef, 2, n_nodes, n_nodes)

    for i in 1:n_nodes, j in 1:n_nodes
      node_coordinates[1, i, j] = element_x_offset + dx/2 * nodes[i]
      node_coordinates[2, i, j] = element_y_offset + dy/2 * nodes[j]
    end

    elements[element_x, element_y] = Element(node_coordinates, inverse_jacobian)
  end

  return nothing
end


# Initialize connectivity between elements and interfaces
function init_interfaces!(elements, mesh::StructuredMesh{2, RealT}, equations::AbstractEquations, dg::DG) where {RealT}
  nvars = nvariables(equations)

  @unpack size, linear_indices = mesh

  # Inner interfaces
  for element_x in 1:size[1], element_y in 1:size[2]
    # Interfaces in x-direction
    if element_x > 1
      left_element_id = linear_indices[element_x - 1, element_y]
      right_element_id = linear_indices[element_x, element_y]

      interface = Interface{2, RealT}(left_element_id, right_element_id, 1, nvars, nnodes(dg))

      elements[left_element_id].interfaces[2] = interface
      elements[right_element_id].interfaces[1] = interface
    end

    # Interfaces in y-direction
    if element_y > 1
      left_element_id = linear_indices[element_x, element_y - 1]
      right_element_id = linear_indices[element_x, element_y]

      interface = Interface{2, RealT}(left_element_id, right_element_id, 2, nvars, nnodes(dg))
      
      elements[left_element_id].interfaces[4] = interface
      elements[right_element_id].interfaces[3] = interface
    end
  end

  # Boundaries in x-direction
  for element_y in 1:size[2]
    left_element_id = linear_indices[end, element_y]
    right_element_id = linear_indices[begin, element_y]

    interface = Interface{2, RealT}(left_element_id, right_element_id, 1, nvars, nnodes(dg))

    elements[left_element_id].interfaces[2] = interface
    elements[right_element_id].interfaces[1] = interface
  end

  # Boundaries in y-direction
  for element_x in 1:size[1]
    left_element_id = linear_indices[element_x, end]
    right_element_id = linear_indices[element_x, begin]

    interface = Interface{2, RealT}(left_element_id, right_element_id, 2, nvars, nnodes(dg))

    elements[left_element_id].interfaces[4] = interface
    elements[right_element_id].interfaces[3] = interface
  end

  return nothing
end
