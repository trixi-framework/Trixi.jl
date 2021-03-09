function init_elements!(elements, mesh::StructuredMesh{RealT, 2}, nodes) where {RealT}
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

    node_coordinates = Array{SVector{2, RealT}, 2}(undef, n_nodes, n_nodes)

    for i in 1:n_nodes, j in 1:n_nodes
      node_coordinates[i, j] = SVector(element_x_offset + dx/2 * nodes[i], 
                                       element_y_offset + dy/2 * nodes[j])
    end

    elements[element_x, element_y] = Element{RealT, 2}(node_coordinates, inverse_jacobian)
  end

  return nothing
end


# Initialize connectivity between elements and interfaces
function init_interfaces!(elements, mesh::StructuredMesh{RealT, 2}, equations::AbstractEquations, dg::DG) where {RealT}
  nvars = nvariables(equations)

  # Inner interfaces
  for element_x in 1:mesh.size[1], element_y in 1:mesh.size[2]
    # Vertical interfaces
    interface = Interface{RealT, 2}(nvars, nnodes(dg), 1)

    if element_x > 1
      elements[element_x, element_y].interfaces[1] = interface
      elements[element_x - 1, element_y].interfaces[2] = interface
    end

    # Horizontal interfaces
    interface = Interface{RealT, 2}(nvars, nnodes(dg), 2)

    if element_y > 1
      elements[element_x, element_y].interfaces[3] = interface
      elements[element_x, element_y - 1].interfaces[4] = interface
    end
  end

  # Vertical boundaries
  for element_y in 1:mesh.size[2]
    interface_left = Interface{RealT, 2}(nvars, nnodes(dg), 1)
    interface_right = Interface{RealT, 2}(nvars, nnodes(dg), 1)

    elements[1, element_y].interfaces[1] = interface_left
    elements[end, element_y].interfaces[2] = interface_right
  end

  # Horizontal boundaries
  for element_x in 1:mesh.size[1]
    interface_left = Interface{RealT, 2}(nvars, nnodes(dg), 2)
    interface_right = Interface{RealT, 2}(nvars, nnodes(dg), 2)

    elements[element_x, 1].interfaces[3] = interface_left
    elements[element_x, end].interfaces[4] = interface_right
  end

  return nothing
end