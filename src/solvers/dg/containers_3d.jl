
# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh3D,
                       equations::AbstractEquations{3, NVARS},
                       basis::LobattoLegendreBasis{T, NNODES}, ::Type{RealT}) where {RealT<:Real, NVARS, T, NNODES}
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer3D{RealT, NVARS, NNODES-1}(n_elements)

  init_elements!(elements, cell_ids, mesh, basis.nodes)
  return elements
end


# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh3D,
                         elements::ElementContainer3D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer3D{RealT, NVARS, POLYDEG}(n_interfaces)

  # Connect elements with interfaces
  init_interfaces!(interfaces, elements, mesh)
  return interfaces
end


# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh3D,
                         elements::ElementContainer3D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer3D{RealT, NVARS, POLYDEG}(n_boundaries)

  # Connect elements with boundaries
  init_boundaries!(boundaries, elements, mesh)
  return boundaries
end


# Create mortar container and initialize mortar data in `elements`.
function init_mortars(cell_ids, mesh::TreeMesh3D,
                      elements::ElementContainer3D{RealT, NVARS, POLYDEG},
                      mortar::LobattoLegendreMortarL2) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize containers
  n_mortars = count_required_mortars(mesh, cell_ids)
  mortars = L2MortarContainer3D{RealT, NVARS, POLYDEG}(n_mortars)

  # Connect elements with mortars
  init_mortars!(mortars, elements, mesh)
  return mortars
end

