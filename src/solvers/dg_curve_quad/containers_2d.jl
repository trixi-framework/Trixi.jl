
include("element_geometry.jl")

# Container data structure (structure-of-arrays style) for DG elements on curved mesh
struct GeneralElementContainer2D{RealT<:Real, NNODES}
  geometry  ::ElementGeometry{RealT, NNODES} # contains Jacobian, metric terms, normals, etc.
  bndy_names::SVector{4, String}             # names of the four boundaries of the element needed for boundary conditions
end


# construct a single straight-sided element and its geometry for use in the DG approximation
function GeneralElementContainer2D(RealT, polydeg, nodes, corners, boundaries)

  bndy_names = boundaries
  geometry   = ElementGeometry(RealT, polydeg, nodes, corners)

  return GeneralElementContainer2D{RealT, polydeg+1}(geometry, bndy_names)
end


# construct a single curved element and its geometry for use in the DG approximation
function GeneralElementContainer2D(RealT, polydeg, nodes, GammaCurves::Array{GammaCurve, 1}, boundaries)

  bndy_names = boundaries
  geometry   = ElementGeometry(RealT, polydeg, nodes, GammaCurves)

  return GeneralElementContainer2D{RealT, polydeg+1}(geometry, bndy_names)
end


# generic container for the edges in an unstructured mesh
struct GeneralInterfaceContainer2D
  edge_type        ::String
  start_idx        ::Int64
  inc_idx          ::Int64
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
function GeneralInterfaceContainer2D(edge_information, poly_deg)

# default that all edges are interior to the mesh
  edge_type = "Interior"
# default for the start and increment indexing, reset later if necessary
  start_idx = 1
  inc_idx   = 1
# Check the boundaries/orientation along each edge (the defaults are set in the edge constructor)
  if edge_information[4] == 0
  # this edge is on a physical boundary as it has no neighbour
    edge_type = "Boundary"
  elseif edge_information[6] < 0
  # coordinate system in the right element is "flipped" compared to the left element. So, the start
  # and increment indexes are adjusted such that the right neighbour coordinate system can match
  # the left neighbour when coupling is computed
    start_idx = poly_deg + 1
    inc_idx   = -1
  end

  return GeneralInterfaceContainer2D( edge_type, start_idx, inc_idx,
                                     ( edge_information[3] , edge_information[4] ),
                                     ( edge_information[5] , edge_information[6] ) )
end
