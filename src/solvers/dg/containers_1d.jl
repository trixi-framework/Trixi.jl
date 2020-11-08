
# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh1D,
                       equations::AbstractEquations{1, NVARS},
                       basis::LobattoLegendreBasis{T, NNODES}, ::Type{RealT}) where {RealT<:Real, NVARS, T, NNODES}
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer1D{RealT, NVARS, NNODES-1}(n_elements)

  init_elements!(elements, cell_ids, mesh, basis.nodes)
  return elements
end


# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh1D,
                         elements::ElementContainer1D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer1D{RealT, NVARS, POLYDEG}(n_interfaces)

  # Connect elements with interfaces
  init_interfaces!(interfaces, elements, mesh)
  return interfaces
end


# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh1D,
                         elements::ElementContainer1D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer1D{RealT, NVARS, POLYDEG}(n_boundaries)

  # Connect elements with boundaries
  init_boundaries!(boundaries, elements, mesh)
  return boundaries
end

