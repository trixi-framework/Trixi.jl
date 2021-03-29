
include("element_geometry.jl")

# Container data structure (structure-of-arrays style) for DG elements on curved mesh
struct GeneralElementContainer2D{RealT<:Real, NNODES}
#  corner_ids::Vector{Int, 4}             # indicies of the corners nodes TODO: could remove as un-necessary
  geometry  ::ElementGeometry{RealT, NNODES}      # contains Jacobian, metric terms, normals, etc.
  bndy_names::SVector{4, String}          # names of the four boundaries of the element needed for boundary conditions
end


# construct a single straight-sided element and its geometry for use in the DG approximation
function GeneralElementContainer2D(RealT, polydeg, nodes, corners, boundaries) #is_curved::Bool)

  bndy_names = boundaries
  geometry   = ElementGeometry(RealT, polydeg, nodes, corners)

  return GeneralElementContainer2D{RealT, polydeg+1}(geometry, bndy_names)
end


# construct a single curved element and its geometry for use in the DG approximation
function GeneralElementContainer2D(RealT, polydeg, nodes, GammaCurves::Array{GammaCurve, 1}, boundaries) #is_curved::Bool)

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


# constructor for the generic edge in an unstructured mesh where edge_information was read in from a file
function GeneralInterfaceContainer2D(edge_information, poly_deg)

# default that all edges are interior to the mesh
  edge_type = "Interior"
# default for the start and increment indexing, reset later if necessary
  start_idx = 1
  inc_idx   = 1
# Check the boundaries/orientation along each edge (the defaults are set in the edge constructor)
  if edge_information[4] == 0
    edge_type = "Boundary"
  elseif edge_information[6] < 0
    start_idx = poly_deg + 1
    inc_idx   = -1
  end
# get the left and right element neighbour IDs
#  element_ids[1] = edge_information[3]
#  element_ids[2] = edge_information[4]
# get the local side number for the left and right neighbour elements
#  element_side_ids[1] = edge_information[5]
#  element_side_ids[2] = edge_information[6]

  return GeneralInterfaceContainer2D( edge_type, start_idx, inc_idx,
                                     ( edge_information[3] , edge_information[4] ),
                                     ( edge_information[5] , edge_information[6] ) )
end
