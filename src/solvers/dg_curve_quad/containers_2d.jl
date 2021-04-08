
include("element_geometry.jl")

# Container data structure (structure-of-arrays style) for DG elements on curved unstructured mesh
struct UnstructuredElementContainer2D{RealT<:Real, NNODES, NVARS, NELEMENTS}
  geometry           ::Vector{ElementGeometry} # contains Jacobian, metric terms, normals, etc.
  surface_flux_values::Array{RealT , 4}        # [variables, i, local sides, elements]
end


# construct an empty curved element container to be filled later with geometries in the
# unstructured mesh constructor
function UnstructuredElementContainer2D(RealT, nvars, polydeg, nelements)

  geometry            = Vector{ElementGeometry}(undef, nelements)
  surface_flux_values = zeros( nvars , polydeg + 1 , 4 , nelements )

  return UnstructuredElementContainer2D{RealT, polydeg+1, nvars, nelements}(geometry,
                                                                            surface_flux_values)
end


# generic container for the interior interfaces of an unstructured mesh
struct UnstructuredInterfaceContainer2D{RealT<:Real, NVARS, POLYDEG}
  u                ::Array{RealT, 4}   # [primary/secondary, variables, i, interfaces]
  start_index      ::Vector{Int64}     # [interfaces]
  inc_index        ::Vector{Int64}     # [interfaces]
  element_ids      ::Array{Int64, 2}   # [primary/secondary, interfaces]
  element_side_ids ::Array{Int64, 2}   # [primary/secondary, interfaces]
end


# construct an empty curved interface container to be filled later with neighbour information in the
# unstructured mesh constructor
function UnstructuredInterfaceContainer2D{RealT, NVARS, POLYDEG}(capacity::Int64) where {RealT<:Real, NVARS, POLYDEG}

  n_nodes = POLYDEG + 1

  u                = zeros(2, NVARS, n_nodes, capacity)
  start_index      = Vector{Int64}(undef, capacity)
  inc_index        = Vector{Int64}(undef, capacity)
  element_ids      = Array{Int64}(undef, (2 , capacity))
  element_side_ids = Array{Int64}(undef, (2 , capacity))

  return UnstructuredInterfaceContainer2D{RealT, NVARS, POLYDEG}( u,
                                                                  start_index,
                                                                  inc_index,
                                                                  element_ids,
                                                                  element_side_ids )
end


@inline ninterfaces(interfaces::UnstructuredInterfaceContainer2D) = length(interfaces.start_index)

@inline eachinterface(interfaces::UnstructuredInterfaceContainer2D) = Base.OneTo(ninterfaces(interfaces))


function init_interfaces(RealT, edge_information, boundary_names, nvars, poly_deg,
                         n_interfaces, n_elements)

  interfaces = UnstructuredInterfaceContainer2D{RealT, nvars, poly_deg}(n_interfaces)

  # extract and save the appropriate neighbour information
  init_interfaces!(interfaces, edge_information, boundary_names, poly_deg, n_elements)
  return interfaces
end


function init_interfaces!(interfaces, edge_information, boundary_names, polydeg, n_elements)

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
      interfaces.element_ids[:,j]      .= ( primary_element, secondary_element )
      interfaces.element_side_ids[:,j] .= ( primary_side   , secondary_side    )
      # set the start and increment indexing
      #  Note! We assume that the periodic mesh has no flipped element coordinate systems
      interfaces.start_index[j] = 1
      interfaces.inc_index[j]   = 1
    end
  end

  return nothing
end


function init_interfaces(RealT, edge_information, nvars, poly_deg, n_interfaces)

  interfaces = UnstructuredInterfaceContainer2D{RealT, nvars, poly_deg}(n_interfaces)

  # extract and save the appropriate neighbour information
  init_interfaces!(interfaces, edge_information, poly_deg)
  return interfaces
end


function init_interfaces!(interfaces, edge_information, polydeg)

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


# generic container for the boundary interfaces of an unstructured mesh
struct UnstructuredBoundaryContainer2D{RealT<:Real, NVARS, POLYDEG}
  u               ::Array{RealT, 4} # [primary, variables, i, boundaries]
  element_id      ::Vector{Int64}   # [boundaries]
  element_side_id ::Vector{Int64}   # [boundaries]
  name            ::Vector{String}  # [boundaries]
end


# construct an empty curved boundary container to be filled later with neighbour information in the
# unstructured mesh constructor
function UnstructuredBoundaryContainer2D{RealT, NVARS, POLYDEG}(capacity::Int64) where {RealT<:Real, NVARS, POLYDEG}

  n_nodes = POLYDEG + 1

  u               = zeros(1, NVARS, n_nodes, capacity)
  element_id      = Vector{Int64}(undef, capacity)
  element_side_id = Vector{Int64}(undef, capacity)
  name            = fill("empty", capacity)

  return UnstructuredBoundaryContainer2D{RealT, NVARS, POLYDEG}( u,
                                                                 element_id,
                                                                 element_side_id,
                                                                 name )
end


@inline nboundaries(boundaries::UnstructuredBoundaryContainer2D) = length(boundaries.name)

@inline eachboundary(boundaries::UnstructuredBoundaryContainer2D) = Base.OneTo(nboundaries(boundaries))


function init_boundaries(RealT, edge_information, boundary_names, nvars, poly_deg, n_boundaries)

  boundaries = UnstructuredBoundaryContainer2D{RealT, nvars, poly_deg}(n_boundaries)

  # extract and save the appropriate neighbour information
  init_boundaries!(boundaries, edge_information, boundary_names, poly_deg)
  return boundaries
end


function init_boundaries!(boundaries, edge_information, boundary_names, polydeg)

  total_n_interfaces = size(edge_information,2)

  bndy_count = 1
  for j in 1:total_n_interfaces
    if edge_information[4,j] == 0
      # get the primary element information at a boundary interface
      boundaries.element_id[bndy_count]      = edge_information[3,j] # primary element id
      boundaries.element_side_id[bndy_count] = edge_information[5,j] # primary side id
      boundaries.name[bndy_count]            = boundary_names[ edge_information[5,j], # local side id
                                                               edge_information[3,j]  # global element id
                                                             ]
      bndy_count += 1
    end
  end

  return nothing
end
