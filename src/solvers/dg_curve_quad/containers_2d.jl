
include("element_geometry.jl")

# Container data structure (structure-of-arrays style) for DG elements on curved mesh
#  ARW: this could get updated because this contains the information of a single element so we use
#       an array of containers later. Could change to be a single container with an array of geometries
struct GeneralElementContainer2D{RealT<:Real, NNODES, NVARS}
  geometry           ::ElementGeometry{RealT, NNODES}         # contains Jacobian, metric terms, normals, etc.
  bndy_names         ::SVector{4, String}                     # names of the four sides of the element needed for boundary conditions
  surface_u_values   ::MArray{Tuple{NVARS, NNODES, 4}, RealT} # [variables, i, local sides]
  surface_flux_values::MArray{Tuple{NVARS, NNODES, 4}, RealT} # [variables, i, local sides]
end

nvariables(::GeneralElementContainer2D{RealT, NNODES, NVARS}) where {RealT, NNODES, NVARS} = NVARS

# construct a single straight-sided element and its geometry for use in the DG approximation
function GeneralElementContainer2D(RealT, nvars, polydeg, nodes, corners, boundaries)

  bndy_names          = boundaries
  geometry            = ElementGeometry(RealT, polydeg, nodes, corners)
  surface_u_values    = zeros( nvars , polydeg + 1 , 4 )
  surface_flux_values = zeros( nvars , polydeg + 1 , 4 )

  return GeneralElementContainer2D{RealT, polydeg+1, nvars}( geometry,
                                                             bndy_names,
                                                             surface_u_values,
                                                             surface_flux_values )
end


# construct a single curved element and its geometry for use in the DG approximation
function GeneralElementContainer2D(RealT, nvars, polydeg, nodes, GammaCurves::Array{GammaCurve, 1}, boundaries)

  bndy_names          = boundaries
  geometry            = ElementGeometry(RealT, polydeg, nodes, GammaCurves)
  surface_u_values    = zeros( nvars , polydeg + 1 , 4 )
  surface_flux_values = zeros(nvars, polydeg+1 , 4)

  return GeneralElementContainer2D{RealT, polydeg+1, nvars}( geometry,
                                                             bndy_names,
                                                             surface_u_values,
                                                             surface_flux_values )
end


# generic container for the edges in an unstructured mesh
struct GeneralInterfaceContainer2D
  interface_type   ::String
  start_index      ::Int64
  inc_index        ::Int64
  element_ids      ::SVector{2, Int64}
  element_side_ids ::SVector{2, Int64}
end


# constructor for the generic edge in an unstructured mesh with edge_information from a file which
# contains the following information:
#    edge_information[1] = start node ID
#    edge_information[2] = end node ID
#    edge_information[3] = element ID on left
#    edge_information[4] = element ID on right
#    edge_information[5] = side of left element
#    edge_information[6] = side of right element
function GeneralInterfaceContainer2D(RealT, edge_information, poly_deg)

# default that all edges are interior to the mesh
  interface_type = "Interior"
# default for the start and increment indexing, reset below if necessary
  start_idx = 1
  inc_idx   = 1
# Check the boundaries/orientation along each edge
  if edge_information[4] == 0
  # this edge is on a physical boundary as it has no neighbour
    interface_type = "Boundary"
  elseif edge_information[6] < 0
  # coordinate system in the secondary element is "flipped" compared to the primary element. So, the start
  # and increment indexes are adjusted such that the secondary neighbour coordinate system can match
  # the primary neighbour when surface coupling is computed
    start_idx = poly_deg + 1
    inc_idx   = -1
  end

  return GeneralInterfaceContainer2D( interface_type, start_idx, inc_idx,
                                    ( edge_information[3] , edge_information[4] ),
                                    ( edge_information[5] , abs(edge_information[6]) ) )
end
