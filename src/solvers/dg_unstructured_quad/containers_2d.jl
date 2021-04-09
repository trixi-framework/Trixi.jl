
# Container data structure (structure-of-arrays style) for DG elements on curved unstructured mesh
struct UnstructuredElementContainer2D{RealT<:Real, NVARS, POLYDEG}
  node_coordinates   ::Array{RealT, 4} # [ndims, nnodes, nnodes, nelement]
  X_xi               ::Array{RealT, 3} # [nnodes, nnodes, nelement]
  X_eta              ::Array{RealT, 3} # [nnodes, nnodes, nelement]
  Y_xi               ::Array{RealT, 3} # [nnodes, nnodes, nelement]
  Y_eta              ::Array{RealT, 3} # [nnodes, nnodes, nelement]
  inverse_jacobian   ::Array{RealT, 3} # [nnodes, nnodes, nelement]
  normals            ::Array{RealT, 4} # [ndims, nnodes, local sides, nelement]
  tangents           ::Array{RealT, 4} # [ndims, nnodes, local sides, nelement]
  scaling            ::Array{RealT, 3} # [nnodes, local sides, nelement]
  surface_flux_values::Array{RealT, 4} # [variables, nnodes, local sides, elements]
end


# construct an empty curved element container to be filled later with geometries in the
# unstructured mesh constructor
function UnstructuredElementContainer2D{RealT, NVARS, POLYDEG}(capacity::Int64) where {RealT<:Real, NVARS, POLYDEG}

  nnodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  node_coordinates    = fill(nan, (2, nnodes, nnodes, capacity))
  X_xi                = fill(nan, (nnodes, nnodes, capacity))
  X_eta               = fill(nan, (nnodes, nnodes, capacity))
  Y_xi                = fill(nan, (nnodes, nnodes, capacity))
  Y_eta               = fill(nan, (nnodes, nnodes, capacity))
  inverse_jacobian    = fill(nan, (nnodes, nnodes, capacity))
  normals             = fill(nan, (2, nnodes, 4, capacity))
  tangents            = fill(nan, (2, nnodes, 4, capacity))
  scaling             = fill(nan, (nnodes, 4, capacity))
  surface_flux_values = fill(nan, (NVARS, nnodes, 4, capacity))

  return UnstructuredElementContainer2D{RealT, NVARS, POLYDEG}(node_coordinates,
                                                               X_xi, X_eta, Y_xi, Y_eta,
                                                               inverse_jacobian,
                                                               normals, tangents, scaling,
                                                               surface_flux_values)
end


# TODO: adjust this to use the same nelement and each element as the other parts of Trixi
@inline nelements(elements::UnstructuredElementContainer2D) = size(elements.inverse_jacobian, 3)

@inline eachelement(elements::UnstructuredElementContainer2D) = Base.OneTo(nelements(elements))


function init_elements(RealT, mesh, dg_nodes, nvars, poly_deg)

  elements = UnstructuredElementContainer2D{RealT, nvars, poly_deg}(mesh.n_elements)

  four_corners = zeros(4, 2)
  #loop through elements and call the correct constructor based on the is curved
  for element_id in eachelement(elements)
    if mesh.element_is_curved[element_id]
      init_element!(elements, element_id, poly_deg, dg_nodes, mesh.elements_curves[:, element_id])
    else # straight sided element
      for i in 1:4
        # pull the (x,y) values of these corners out of the global corners array
        four_corners[i, :] .= mesh.corners[:, mesh.element_node_ids[i, element_id]]
      end
      init_element!(elements, element_id, poly_deg, dg_nodes, four_corners)
    end
  end

  return elements
end


# initialize all the values in the container on a curved sided element
function init_element!(elements, element_id, polydeg, nodes, GammaCurves::Array{GammaCurve,1})

  calc_node_coordinates!(elements.node_coordinates, element_id, nodes, polydeg, GammaCurves)

  calc_metric_terms!(elements.X_xi, elements.X_eta, elements.Y_xi, elements.Y_eta, element_id,
                     nodes, polydeg, GammaCurves)

  calc_inverse_jacobian!(elements.inverse_jacobian, element_id, elements.X_xi, elements.X_eta,
                         elements.Y_xi, elements.Y_eta)

  calc_normals_scaling_and_tangents!(elements.normals, elements.scaling, elements.tangents,
                                     element_id, nodes, polydeg, GammaCurves)

  return elements
end


# initialize all the values in the container on a straight sided element
function init_element!(elements, element_id, polydeg, nodes, corners)

  calc_node_coordinates!(elements.node_coordinates, element_id, nodes, polydeg, corners)

  calc_metric_terms!(elements.X_xi, elements.X_eta, elements.Y_xi, elements.Y_eta, element_id,
                     nodes, polydeg, corners)

  calc_inverse_jacobian!(elements.inverse_jacobian, element_id, elements.X_xi, elements.X_eta,
                         elements.Y_xi, elements.Y_eta)

  calc_normals_scaling_and_tangents!(elements.normals, elements.scaling, elements.tangents,
                                     element_id, nodes, polydeg, corners)

  return elements
end


# generic container for the interior interfaces of an unstructured mesh
struct UnstructuredInterfaceContainer2D{RealT<:Real, NVARS, POLYDEG}
  u                ::Array{RealT, 4} # [primary/secondary, variables, i, interfaces]
  start_index      ::Vector{Int64}   # [interfaces]
  inc_index        ::Vector{Int64}   # [interfaces]
  element_ids      ::Array{Int64, 2} # [primary/secondary, interfaces]
  element_side_ids ::Array{Int64, 2} # [primary/secondary, interfaces]
end


# construct an empty curved interface container to be filled later with neighbour information in the
# unstructured mesh constructor
function UnstructuredInterfaceContainer2D{RealT, NVARS, POLYDEG}(capacity::Int64) where {RealT<:Real, NVARS, POLYDEG}

  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  u                = fill(nan, (2, NVARS, n_nodes, capacity))
  start_index      = fill(typemin(Int64), capacity)
  inc_index        = fill(typemin(Int64), capacity)
  element_ids      = fill(typemin(Int64), (2, capacity))
  element_side_ids = fill(typemin(Int64), (2, capacity))

  return UnstructuredInterfaceContainer2D{RealT, NVARS, POLYDEG}(u, start_index, inc_index,
                                                                 element_ids, element_side_ids)
end


@inline ninterfaces(interfaces::UnstructuredInterfaceContainer2D) = length(interfaces.start_index)

@inline eachinterface(interfaces::UnstructuredInterfaceContainer2D) = Base.OneTo(ninterfaces(interfaces))


function init_interfaces(RealT, mesh, nvars, poly_deg)

  # get the number of internal interfaces in the mesh
  if isperiodic(mesh)
    interior_interfaces = mesh.n_interfaces
  else
    interior_interfaces = mesh.n_interfaces - mesh.n_boundary
  end

  interfaces = UnstructuredInterfaceContainer2D{RealT, nvars, poly_deg}(interior_interfaces)

  # extract and save the appropriate neighbour information from the mesh skeleton
  init_interfaces!(interfaces, mesh.neighbour_information, mesh.boundary_names, poly_deg,
                   mesh.n_elements, Val(isperiodic(mesh)))

  return interfaces
end


function init_interfaces!(interfaces, edge_information, boundary_names, polydeg, n_elements,
                          periodic::Val{false})

  total_n_interfaces = size(edge_information,2)
  intr_count = 1
  for j in 1:total_n_interfaces
    if edge_information[4,j] > 0
      # get the primary/secondary element information and coupling for an interior interface
      interfaces.element_ids[1,intr_count]      = edge_information[3,j]      # primary element id
      interfaces.element_ids[2,intr_count]      = edge_information[4,j]      # secondary element id
      interfaces.element_side_ids[1,intr_count] = edge_information[5,j]      # primary side id
      interfaces.element_side_ids[2,intr_count] = abs(edge_information[6,j]) # secondary side id
      # default the start and increment indexing
      interfaces.start_index[intr_count] = 1
      interfaces.inc_index[intr_count]   = 1
      if edge_information[6,j] < 0
      # coordinate system in the secondary element is "flipped" compared to the primary element.
      # Adjust the start and increment indexes such that the secondary element coordinate system
      # can match the primary neighbour when surface coupling is computed
        interfaces.start_index[intr_count] = polydeg + 1
        interfaces.inc_index[intr_count]   = -1
      end
      intr_count += 1
    end
  end

  return nothing
end


function init_interfaces!(interfaces, edge_information, boundary_names, polydeg, n_elements,
                          periodic::Val{true})

  total_n_interfaces = size(edge_information,2)
  # for now this set a fully periodic domain
  #   TODO: possibly adjust to be able to set periodic in only the x or y direction
  for j in 1:total_n_interfaces
    if edge_information[4,j] > 0
      # get the primary/secondary element information and coupling for an interior interface
      interfaces.element_ids[1,j]      = edge_information[3,j]      # primary element id
      interfaces.element_ids[2,j]      = edge_information[4,j]      # secondary element id
      interfaces.element_side_ids[1,j] = edge_information[5,j]      # primary side id
      interfaces.element_side_ids[2,j] = abs(edge_information[6,j]) # secondary side id
      # default the start and increment indexing
      interfaces.start_index[j] = 1
      interfaces.inc_index[j]   = 1
      if edge_information[6,j] < 0
        # coordinate system in the secondary element is "flipped" compared to the primary element.
        # Adjust the start and increment indexes such that the secondary element coordinate system
        # can match the primary neighbour when surface coupling is computed
        interfaces.start_index[j] = polydeg + 1
        interfaces.inc_index[j]   = -1
      end
    else
      # way to set periodic BCs where we are assuming to have a structured mesh with internal curves
      primary_side = edge_information[5,j]
      primary_element = edge_information[3,j]
      if boundary_names[primary_side, primary_element] == "Bottom"
        secondary_element = primary_element + (n_elements - convert(Int64, sqrt(n_elements)))
        secondary_side    = 3
      elseif boundary_names[primary_side, primary_element] == "Top"
        secondary_element = primary_element - (n_elements - convert(Int64, sqrt(n_elements)))
        secondary_side    = 1
      elseif boundary_names[primary_side, primary_element] == "Left"
        secondary_element = primary_element + (convert(Int64, sqrt(n_elements)) - 1)
        secondary_side    = 2
      elseif boundary_names[primary_side, primary_element] == "Right"
        secondary_element = primary_element - (convert(Int64, sqrt(n_elements)) - 1)
        secondary_side    = 4
      end
      interfaces.element_ids[1,j]      = primary_element
      interfaces.element_ids[2,j]      = secondary_element
      interfaces.element_side_ids[1,j] = primary_side
      interfaces.element_side_ids[2,j] = secondary_side
      # set the start and increment indexing
      #  Note! We assume that the periodic mesh has no flipped element coordinate systems
      interfaces.start_index[j] = 1
      interfaces.inc_index[j]   = 1
    end
  end

  return nothing
end


# generic container for the boundary interfaces of an unstructured mesh
struct UnstructuredBoundaryContainer2D{RealT<:Real, NVARS, POLYDEG}
  u               ::Array{RealT, 4} # [primary, variables, i, boundaries]
  element_id      ::Vector{Int64}   # [boundaries]
  element_side_id ::Vector{Int64}   # [boundaries]
  node_coordinates::Array{RealT, 3} # [ndims, nnodes, boundaries]
  name            ::Vector{String}  # [boundaries]
end


# construct an empty curved boundary container to be filled later with neighbour information in the
# unstructured mesh constructor
function UnstructuredBoundaryContainer2D{RealT, NVARS, POLYDEG}(capacity::Int64) where {RealT<:Real, NVARS, POLYDEG}

  n_nodes = POLYDEG + 1
  nan = convert(RealT, NaN)

  u                = fill(nan, (1, NVARS, n_nodes, capacity))
  element_id       = fill(typemin(Int64), capacity)
  element_side_id  = fill(typemin(Int64), capacity)
  node_coordinates = fill(nan, (2, n_nodes, capacity))
  name             = fill("empty", capacity)

  return UnstructuredBoundaryContainer2D{RealT, NVARS, POLYDEG}(u, element_id, element_side_id,
                                                                node_coordinates, name)
end


@inline nboundaries(boundaries::UnstructuredBoundaryContainer2D) = length(boundaries.name)

@inline eachboundary(boundaries::UnstructuredBoundaryContainer2D) = Base.OneTo(nboundaries(boundaries))


function init_boundaries(RealT, mesh, elements, nvars, poly_deg)

  boundaries = UnstructuredBoundaryContainer2D{RealT, nvars, poly_deg}(mesh.n_boundary)

  # extract and save the appropriate neighbour information
  init_boundaries!(boundaries, mesh.neighbour_information, mesh.boundary_names, elements, poly_deg)
  return boundaries
end


function init_boundaries!(boundaries, edge_information, boundary_names, elements, polydeg)

  total_n_interfaces = size(edge_information,2)
  bndy_count = 1
  for j in 1:total_n_interfaces
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
