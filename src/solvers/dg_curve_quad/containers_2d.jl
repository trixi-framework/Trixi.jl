
include("element_geometry.jl")

# Container data structure (structure-of-arrays style) for DG elements on curved unstructured mesh
struct CurvedElementContainer2D{RealT<:Real, NNODES, NVARS, NELEMENTS}
  geometry           ::Vector{ElementGeometry} # contains Jacobian, metric terms, normals, etc.
  bndy_names         ::Array{String, 2}        # names of the four sides of the element needed for boundary conditions
  surface_u_values   ::Array{RealT , 4}        # [variables, i, local sides, elements]
  surface_flux_values::Array{RealT , 4}        # [variables, i, local sides, elements]
end


# construct an empty curved element container to be filled later with its geometry
function CurvedElementContainer2D(RealT, nvars, polydeg, nelements)

  bndy_names          = fill("empty", (4, nelements) )
  geometry            = Vector{ElementGeometry}(undef, nelements)
  surface_u_values    = zeros( nvars , polydeg + 1 , 4 , nelements )
  surface_flux_values = zeros( nvars , polydeg + 1 , 4 , nelements )

  return CurvedElementContainer2D{RealT, polydeg+1, nvars, nelements}( geometry,
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
