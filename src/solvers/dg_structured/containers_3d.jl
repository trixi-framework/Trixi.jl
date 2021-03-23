function init_elements!(elements, mesh::StructuredMesh{3, RealT}, nodes) where {RealT}
  n_nodes = length(nodes)

  @unpack size, coordinates_min, coordinates_max = mesh

  # Get cell length
  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]
  dy = (coordinates_max[2] - coordinates_min[2]) / size[2]
  dz = (coordinates_max[3] - coordinates_min[3]) / size[3]

  inverse_jacobian = 8/(dx * dy * dz)

  # Calculate inverse Jacobian and node coordinates
  for element_x in 1:size[1], element_y in 1:size[2], element_z in 1:size[3]
    # Calculate node coordinates
    element_x_offset = coordinates_min[1] + (element_x-1) * dx + dx/2
    element_y_offset = coordinates_min[2] + (element_y-1) * dy + dy/2
    element_z_offset = coordinates_min[3] + (element_z-1) * dz + dz/2

    node_coordinates = Array{RealT, 4}(undef, 3, n_nodes, n_nodes, n_nodes)

    for i in 1:n_nodes, j in 1:n_nodes, k in 1:n_nodes
      node_coordinates[1, i, j, k] = element_x_offset + dx/2 * nodes[i]
      node_coordinates[2, i, j, k] = element_y_offset + dy/2 * nodes[j]
      node_coordinates[3, i, j, k] = element_z_offset + dz/2 * nodes[k]
    end

    elements[element_x, element_y, element_z] = Element(node_coordinates, inverse_jacobian)
  end

  return nothing
end


# Initialize connectivity between elements and interfaces
function init_interfaces!(elements, mesh::StructuredMesh{3, RealT}, equations::AbstractEquations, dg::DG) where {RealT}
  nvars = nvariables(equations)

  @unpack size = mesh

  # Inner interfaces
  for element_x in 1:size[1], element_y in 1:size[2], element_z in 1:size[3]
    # Interfaces in x-direction
    interface = Interface{3, RealT}(nvars, nnodes(dg), 1)

    if element_x > 1
      elements[element_x, element_y, element_z].interfaces[1] = interface
      elements[element_x - 1, element_y, element_z].interfaces[2] = interface
    end

    # Interfaces in y-direction
    interface = Interface{3, RealT}(nvars, nnodes(dg), 2)

    if element_y > 1
      elements[element_x, element_y, element_z].interfaces[3] = interface
      elements[element_x, element_y - 1, element_z].interfaces[4] = interface
    end

    # Interfaces in z-direction
    interface = Interface{3, RealT}(nvars, nnodes(dg), 3)

    if element_z > 1
      elements[element_x, element_y, element_z].interfaces[5] = interface
      elements[element_x, element_y, element_z - 1].interfaces[6] = interface
    end
  end

  # Boundaries in x-direction
  for element_y in 1:size[2], element_z in 1:size[3]
    interface = Interface{3, RealT}(nvars, nnodes(dg), 1)

    elements[begin, element_y, element_z].interfaces[1] = interface
    elements[end, element_y, element_z].interfaces[2] = interface
  end

  # Boundaries in y-direction
  for element_x in 1:size[1], element_z in 1:size[3]
    interface = Interface{3, RealT}(nvars, nnodes(dg), 2)

    elements[element_x, begin, element_z].interfaces[3] = interface
    elements[element_x, end, element_z].interfaces[4] = interface
  end

  # Boundaries in z-direction
  for element_x in 1:size[1], element_y in 1:size[2]
    interface = Interface{3, RealT}(nvars, nnodes(dg), 3)

    elements[element_x, element_y, begin].interfaces[5] = interface
    elements[element_x, element_y, end].interfaces[6] = interface
  end

  return nothing
end
