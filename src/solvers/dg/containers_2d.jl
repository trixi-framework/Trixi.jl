
# Create element container and initialize element data
function init_elements(cell_ids, mesh::TreeMesh2D,
                       equations::AbstractEquations{2, NVARS},
                       basis::LobattoLegendreBasis{T, NNODES}, ::Type{RealT}) where {RealT<:Real, NVARS, T, NNODES}
  # Initialize container
  n_elements = length(cell_ids)
  elements = ElementContainer2D{RealT, NVARS, NNODES-1}(n_elements)

  init_elements!(elements, cell_ids, mesh, basis.nodes)
  return elements
end


# Create interface container and initialize interface data in `elements`.
function init_interfaces(cell_ids, mesh::TreeMesh2D,
                         elements::ElementContainer2D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_interfaces = count_required_interfaces(mesh, cell_ids)
  interfaces = InterfaceContainer2D{RealT, NVARS, POLYDEG}(n_interfaces)

  # Connect elements with interfaces
  init_interfaces!(interfaces, elements, mesh)
  return interfaces
end


# Create boundaries container and initialize boundary data in `elements`.
function init_boundaries(cell_ids, mesh::TreeMesh2D,
                         elements::ElementContainer2D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_boundaries = count_required_boundaries(mesh, cell_ids)
  boundaries = BoundaryContainer2D{RealT, NVARS, POLYDEG}(n_boundaries)

  # Connect elements with boundaries
  init_boundaries!(boundaries, elements, mesh)
  return boundaries
end


# Create mortar container and initialize mortar data in `elements`.
function init_mortars(cell_ids, mesh::TreeMesh2D,
                      elements::ElementContainer2D{RealT, NVARS, POLYDEG},
                      mortar::LobattoLegendreMortarL2) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize containers
  n_mortars = count_required_mortars(mesh, cell_ids)
  mortars = L2MortarContainer2D{RealT, NVARS, POLYDEG}(n_mortars)

  # Connect elements with mortars
  init_mortars!(mortars, elements, mesh)
  return mortars
end


# Create MPI interface container and initialize MPI interface data in `elements`.
function init_mpi_interfaces(cell_ids, mesh::TreeMesh2D,
                             elements::ElementContainer2D{RealT, NVARS, POLYDEG}) where {RealT<:Real, NVARS, POLYDEG}
  # Initialize container
  n_mpi_interfaces = count_required_mpi_interfaces(mesh, cell_ids)
  mpi_interfaces = MPIInterfaceContainer2D{RealT, NVARS, POLYDEG}(n_mpi_interfaces)

  # Connect elements with interfaces
  init_mpi_interfaces!(mpi_interfaces, elements, mesh)
  return mpi_interfaces
end

