
# Container data structure (structure-of-arrays style) for DG elements on curved unstructured mesh
struct UnstructuredElementContainer2D{RealT<:Real, uEltype<:Real, NVARS, POLYDEG}
  node_coordinates     ::Array{RealT, 4}   # [ndims, nnodes, nnodes, nelement]
  jacobian_matrix      ::Array{RealT, 5}   # [ndims, ndims, nnodes, nnodes, nelement]
  inverse_jacobian     ::Array{RealT, 3}   # [nnodes, nnodes, nelement]
  contravariant_vectors::Array{RealT, 5}   # [ndims, ndims, nnodes, nnodes, nelement]
  normal_directions    ::Array{RealT, 4}   # [ndims, nnodes, local sides, nelement]
  surface_flux_values  ::Array{uEltype, 4} # [variables, nnodes, local sides, elements]
end


# construct an empty curved element container to be filled later with geometries in the
# unstructured mesh constructor
function UnstructuredElementContainer2D{RealT, uEltype, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, uEltype<:Real, NVARS, POLYDEG}

  nnodes = POLYDEG + 1
  nan_RealT = convert(RealT, NaN)
  nan_uEltype = convert(uEltype, NaN)

  node_coordinates      = fill(nan_RealT, (2, nnodes, nnodes, capacity))
  jacobian_matrix       = fill(nan_RealT, (2, 2, nnodes, nnodes, capacity))
  inverse_jacobian      = fill(nan_RealT, (nnodes, nnodes, capacity))
  contravariant_vectors = fill(nan_RealT, (2, 2, nnodes, nnodes, capacity))
  normal_directions     = fill(nan_RealT, (2, nnodes, 4, capacity))
  surface_flux_values   = fill(nan_uEltype, (NVARS, nnodes, 4, capacity))

  return UnstructuredElementContainer2D{RealT, uEltype, NVARS, POLYDEG}(node_coordinates,
                                                                        jacobian_matrix,
                                                                        inverse_jacobian,
                                                                        contravariant_vectors,
                                                                        normal_directions,
                                                                        surface_flux_values)
end


@inline nelements(elements::UnstructuredElementContainer2D) = size(elements.inverse_jacobian, 3)

@inline eachelement(elements::UnstructuredElementContainer2D) = Base.OneTo(nelements(elements))

Base.eltype(::UnstructuredElementContainer2D{RealT, uEltype}) where {RealT, uEltype} = uEltype


@inline function get_surface_normal(vec, indices...)
  # way to extract the normal vector at the surfaces without allocating
  surface_vector = SVector(ntuple(j -> vec[j, indices...], 2))
  return surface_vector
end


function init_elements(RealT, uEltype, mesh, dg_nodes, nvars, polydeg)
  elements = UnstructuredElementContainer2D{RealT, uEltype, nvars, polydeg}(mesh.n_elements)
  init_elements!(elements, mesh, dg_nodes)
  return elements
end


function init_elements!(elements::UnstructuredElementContainer2D, mesh, dg_nodes)
  four_corners = zeros(eltype(mesh.corners), 4, 2)

  # loop through elements and call the correct constructor based on whether the element is curved
  for element in eachelement(elements)
    if mesh.element_is_curved[element]
      init_element!(elements, element, dg_nodes, view(mesh.surface_curves, :, element))
    else # straight sided element
      for i in 1:4, j in 1:2
        # pull the (x,y) values of these corners out of the global corners array
        four_corners[i, j] = mesh.corners[j, mesh.element_node_ids[i, element]]
      end
      init_element!(elements, element, dg_nodes, four_corners)
    end
  end
end


# initialize all the values in the container of a general element (either straight sided or curved)
function init_element!(elements, element, nodes, corners_or_surface_curves)

  calc_node_coordinates!(elements.node_coordinates, element, nodes, corners_or_surface_curves)

  calc_metric_terms!(elements.jacobian_matrix, element, nodes, corners_or_surface_curves)

  calc_inverse_jacobian!(elements.inverse_jacobian, element, elements.jacobian_matrix)

  calc_contravariant_vectors!(elements.contravariant_vectors, element, elements.jacobian_matrix)

  calc_normal_directions!(elements.normal_directions, element, nodes, corners_or_surface_curves)

  return elements
end


# generic container for the interior interfaces of an unstructured mesh
struct UnstructuredInterfaceContainer2D{uEltype<:Real, NVARS, POLYDEG}
  u                ::Array{uEltype, 4} # [primary/secondary, variables, i, interfaces]
  start_index      ::Vector{Int}       # [interfaces]
  index_increment  ::Vector{Int}       # [interfaces]
  element_ids      ::Array{Int, 2}     # [primary/secondary, interfaces]
  element_side_ids ::Array{Int, 2}     # [primary/secondary, interfaces]
end


# construct an empty curved interface container to be filled later with neighbour information in the
# unstructured mesh constructor
function UnstructuredInterfaceContainer2D{uEltype, NVARS, POLYDEG}(capacity::Integer) where {uEltype<:Real, NVARS, POLYDEG}

  n_nodes = POLYDEG + 1
  nan_uEltype = convert(uEltype, NaN)

  u                = fill(nan_uEltype, (2, NVARS, n_nodes, capacity))
  start_index      = fill(typemin(Int), capacity)
  index_increment  = fill(typemin(Int), capacity)
  element_ids      = fill(typemin(Int), (2, capacity))
  element_side_ids = fill(typemin(Int), (2, capacity))

  return UnstructuredInterfaceContainer2D{uEltype, NVARS, POLYDEG}(u, start_index, index_increment,
                                                                   element_ids, element_side_ids)
end


@inline ninterfaces(interfaces::UnstructuredInterfaceContainer2D) = length(interfaces.start_index)


Base.eltype(::UnstructuredInterfaceContainer2D{uEltype}) where {uEltype} = uEltype


function init_interfaces(uEltype, mesh, nvars, polydeg)

  interfaces = UnstructuredInterfaceContainer2D{uEltype, nvars, polydeg}(mesh.n_interfaces)

  # extract and save the appropriate neighbour information from the mesh skeleton
  init_interfaces!(interfaces, mesh.neighbour_information, mesh.boundary_names, polydeg,
                   mesh.n_elements, Val(isperiodic(mesh)))

  return interfaces
end


function init_interfaces!(interfaces, edge_information, boundary_names, polydeg, n_elements,
                          periodic::Val{false})

  n_surfaces = size(edge_information,2)
  intr_count = 1
  for j in 1:n_surfaces
    if edge_information[4,j] > 0
      # get the primary/secondary element information and coupling for an interior interface
      interfaces.element_ids[1,intr_count]      = edge_information[3,j]      # primary element id
      interfaces.element_ids[2,intr_count]      = edge_information[4,j]      # secondary element id
      interfaces.element_side_ids[1,intr_count] = edge_information[5,j]      # primary side id
      interfaces.element_side_ids[2,intr_count] = abs(edge_information[6,j]) # secondary side id
      # default the start and increment indexing
      interfaces.start_index[intr_count] = 1
      interfaces.index_increment[intr_count] = 1
      if edge_information[6,j] < 0
      # coordinate system in the secondary element is "flipped" compared to the primary element.
      # Adjust the start and increment indexes such that the secondary element coordinate system
      # can match the primary neighbour when surface coupling is computed
        interfaces.start_index[intr_count] = polydeg + 1
        interfaces.index_increment[intr_count] = -1
      end
      intr_count += 1
    end
  end

  return nothing
end


function init_interfaces!(interfaces, edge_information, boundary_names, polydeg, n_elements,
                          periodic::Val{true})

  n_surfaces = size(edge_information,2)
  # for now this set a fully periodic domain
  #   TODO: possibly adjust to be able to set periodic in only the x or y direction
  for j in 1:n_surfaces
    if edge_information[4,j] > 0
      # get the primary/secondary element information and coupling for an interior interface
      interfaces.element_ids[1,j]      = edge_information[3,j]      # primary element id
      interfaces.element_ids[2,j]      = edge_information[4,j]      # secondary element id
      interfaces.element_side_ids[1,j] = edge_information[5,j]      # primary side id
      interfaces.element_side_ids[2,j] = abs(edge_information[6,j]) # secondary side id
      # default the start and increment indexing
      interfaces.start_index[j] = 1
      interfaces.index_increment[j] = 1
      if edge_information[6,j] < 0
        # coordinate system in the secondary element is "flipped" compared to the primary element.
        # Adjust the start and increment indexes such that the secondary element coordinate system
        # can match the primary neighbour when surface coupling is computed
        interfaces.start_index[j] = polydeg + 1
        interfaces.index_increment[j] = -1
      end
    else
      # way to set periodic BCs where we are assuming to have a structured mesh with internal curves
      primary_side = edge_information[5,j]
      primary_element = edge_information[3,j]
      # Note: This is a way to get the neighbour element number and local side from a square
      #       structured mesh where the element local surface numbering is right-handed
      if boundary_names[primary_side, primary_element] === :Bottom
        secondary_element = primary_element + (n_elements - convert(Int, sqrt(n_elements)))
        secondary_side    = 3
      elseif boundary_names[primary_side, primary_element] === :Top
        secondary_element = primary_element - (n_elements - convert(Int, sqrt(n_elements)))
        secondary_side    = 1
      elseif boundary_names[primary_side, primary_element] === :Left
        secondary_element = primary_element + (convert(Int, sqrt(n_elements)) - 1)
        secondary_side    = 2
      elseif boundary_names[primary_side, primary_element] === :Right
        secondary_element = primary_element - (convert(Int, sqrt(n_elements)) - 1)
        secondary_side    = 4
      end
      interfaces.element_ids[1,j]      = primary_element
      interfaces.element_ids[2,j]      = secondary_element
      interfaces.element_side_ids[1,j] = primary_side
      interfaces.element_side_ids[2,j] = secondary_side
      # set the start and increment indexing
      #  Note! We assume that the periodic mesh has no flipped element coordinate systems
      interfaces.start_index[j] = 1
      interfaces.index_increment[j] = 1
    end
  end

  return nothing
end


# generic container for the boundary interfaces of an unstructured mesh
struct UnstructuredBoundaryContainer2D{RealT<:Real, uEltype<:Real, NVARS, POLYDEG}
  u               ::Array{uEltype, 3} # [variables, i, boundaries]
  element_id      ::Vector{Int}       # [boundaries]
  element_side_id ::Vector{Int}       # [boundaries]
  node_coordinates::Array{RealT, 3}   # [ndims, nnodes, boundaries]
  name            ::Vector{Symbol}    # [boundaries]
end


# construct an empty curved boundary container to be filled later with neighbour information in the
# unstructured mesh constructor
function UnstructuredBoundaryContainer2D{RealT, uEltype, NVARS, POLYDEG}(capacity::Integer) where {RealT<:Real, uEltype<:Real, NVARS, POLYDEG}

  n_nodes = POLYDEG + 1
  nan_RealT = convert(RealT, NaN)
  nan_uEltype = convert(uEltype, NaN)

  u                = fill(nan_uEltype, (NVARS, n_nodes, capacity))
  element_id       = fill(typemin(Int), capacity)
  element_side_id  = fill(typemin(Int), capacity)
  node_coordinates = fill(nan_RealT, (2, n_nodes, capacity))
  name             = fill(:empty, capacity)

  return UnstructuredBoundaryContainer2D{RealT, uEltype, NVARS, POLYDEG}(u, element_id, element_side_id,
                                                                         node_coordinates, name)
end


@inline nboundaries(boundaries::UnstructuredBoundaryContainer2D) = length(boundaries.name)


function init_boundaries(RealT, uEltype, mesh, elements, nvars, polydeg)

  boundaries = UnstructuredBoundaryContainer2D{RealT, uEltype, nvars, polydeg}(mesh.n_boundaries)

  # extract and save the appropriate boundary information provided any physical boundaries exist
  if mesh.n_boundaries > 0
    init_boundaries!(boundaries, mesh.neighbour_information, mesh.boundary_names, elements)
  end
  return boundaries
end


function init_boundaries!(boundaries, edge_information, boundary_names, elements)

  n_surfaces = size(edge_information,2)
  bndy_count = 1
  for j in 1:n_surfaces
    if edge_information[4,j] == 0
      # get the primary element information at a boundary interface
      primary_element = edge_information[3,j]
      primary_side    = edge_information[5,j]
      boundaries.element_id[bndy_count]      = primary_element
      boundaries.element_side_id[bndy_count] = primary_side

      # extract the physical boundary's name from the global list
      boundaries.name[bndy_count] = boundary_names[primary_side, primary_element]

      # Store copy of the (x,y) node coordinates on the physical boundary
      enc = elements.node_coordinates
      if primary_side == 1
        boundaries.node_coordinates[:, :, bndy_count] .= enc[:, :,    1, primary_element]
      elseif primary_side == 2
        boundaries.node_coordinates[:, :, bndy_count] .= enc[:, end,  :, primary_element]
      elseif primary_side == 3
        boundaries.node_coordinates[:, :, bndy_count] .= enc[:, :,  end, primary_element]
      else # primary_side == 4
        boundaries.node_coordinates[:, :, bndy_count] .= enc[:, 1,    :, primary_element]
      end
      bndy_count += 1
    end
  end

  return nothing
end
