struct Interface{RealT<:Real, NDIMS}
  u_left::Array{RealT, NDIMS} # [variables, i, j]
  u_right::Array{RealT, NDIMS} # [variables, i, j]
  orientation::Int64
  surface_flux_values::Array{RealT, NDIMS} # [variables, i, j]
end

function Interface{RealT, NDIMS}(nvars, nnodes) where {RealT<:Real, NDIMS}
  # TODO Is there a more elegant solution for this?
  u_left = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)
  u_right = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)

  orientation = 1 # TODO

  surface_flux_values = Array{RealT, NDIMS}(undef, nvars, fill(nnodes, NDIMS-1)...)

  return Interface{RealT, NDIMS}(u_left, u_right, orientation, surface_flux_values)
end


struct Element{RealT<:Real, NDIMS}
  node_coordinates::Array{SVector{NDIMS, RealT}, NDIMS}
  # node_coordinates::Array{RealT, 2}
  inverse_jacobian::RealT
  interfaces::Vector{Interface{RealT, NDIMS}} # [orientation]
end

function Element{RealT, NDIMS}(node_coordinates, inverse_jacobian) where {RealT<:Real, NDIMS}
  interfaces = Array{Interface{RealT, NDIMS}}(undef, NDIMS * 2)

  return Element{RealT, NDIMS}(node_coordinates, inverse_jacobian, interfaces)
end


# struct ElementContainer{RealT<:Real, NDIMS}
#   elements::Array{Element{RealT, NDIMS}, NDIMS}
# end

# function ElementContainer{RealT<:Real}(size)
#   NDIMS = length(size)

#   elements = Array{Element{RealT, NDIMS}, NDIMS}(undef, size...)
# end


# Create element container and initialize element data
function init_elements(mesh::StructuredMesh, equations::AbstractEquations{1, NVARS},
    basis::LobattoLegendreBasis{T, NNODES}, ::Type{RealT}) where {RealT<:Real, NVARS, T, NNODES}

  # elements = ElementContainer{RealT}(mesh.size)

  NDIMS = length(mesh.size)

  elements = StructArray{Element{RealT, NDIMS}, NDIMS}(undef, mesh.size...)

  init_elements!(elements, mesh, basis.nodes)
  return elements
end

# TODO
@inline nelements(elements::StructArray) = prod(size(elements))


# Only 1D
function init_elements!(elements, mesh::StructuredMesh{RealT, NDIMS}, nodes) where {RealT, NDIMS}
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

    elements[element_x] = Element{RealT, NDIMS}(node_coordinates, inverse_jacobian)
  end

  return nothing
end


# TODO only 1D
# Initialize connectivity between elements and interfaces
function init_interfaces!(elements, mesh::StructuredMesh{RealT, NDIMS}, equations::AbstractEquations, dg::DG) where {RealT, NDIMS}
  nvars = nvariables(equations)

  # Inner interfaces
  for element_x in 2:mesh.size[1]
    interface = Interface{RealT, NDIMS}(nvars, nnodes(dg))

    elements[element_x].interfaces[1] = interface
    elements[element_x - 1].interfaces[2] = interface
  end

  # Boundary interfaces
  interface_left = Interface{RealT, NDIMS}(nvars, nnodes(dg))
  interface_right = Interface{RealT, NDIMS}(nvars, nnodes(dg))

  elements[1].interfaces[1] = interface_left
  elements[end].interfaces[2] = interface_right

  return nothing
end
