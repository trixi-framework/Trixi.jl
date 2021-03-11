function init_elements!(elements, mesh::StructuredMesh{RealT, 1}, nodes) where {RealT}
  n_nodes = length(nodes)

  @unpack size, coordinates_min, coordinates_max = mesh

  # Get cell length
  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]

  # Calculate inverse Jacobian as 1/(h/2)
  inverse_jacobian = 2/dx

  # Calculate inverse Jacobian and node coordinates
  for element_x in 1:size[1]
    # Calculate node coordinates
    element_x_offset = coordinates_min[1] + (element_x-1) * dx + dx/2

    node_coordinates = Vector{SVector{1, RealT}}(undef, n_nodes)
    # node_coordinates = Array{RealT, 2}(undef, 1, n_nodes)
    for i in 1:n_nodes
        node_coordinates[i] = SVector( element_x_offset + dx/2 * nodes[i] )
    end

    elements[element_x] = Element{RealT, 1}(node_coordinates, inverse_jacobian)
  end

  return nothing
end


# Initialize connectivity between elements and interfaces
function init_interfaces!(elements, mesh::StructuredMesh{RealT, 1}, equations::AbstractEquations, dg::DG) where {RealT}
  nvars = nvariables(equations)

  # Inner interfaces
  for element_x in 2:mesh.size[1]
    interface = Interface{RealT, 1}(nvars, nnodes(dg), 1)

    elements[element_x].interfaces[1] = interface
    elements[element_x - 1].interfaces[2] = interface
  end

  # Boundary interfaces
  interface_left = Interface{RealT, 1}(nvars, nnodes(dg), 1)
  interface_right = Interface{RealT, 1}(nvars, nnodes(dg), 1)

  elements[1].interfaces[1] = interface_left
  elements[end].interfaces[2] = interface_right

  return nothing
end
