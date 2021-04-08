
include("curve_interpolant.jl")
include("containers_2d.jl")

struct UnstructuredMesh{NDIMS, RealT<:Real} <: Trixi.AbstractMesh{NDIMS}
  mesh_filename::String
  interfaces   ::UnstructuredInterfaceContainer2D
  boundaries   ::UnstructuredBoundaryContainer2D
  elements     ::UnstructuredElementContainer2D
end


# constructor for an unstructured mesh read in from a file
function UnstructuredMesh(RealT, filename, nvars, polydeg, dg_nodes, boundary_conditions)

  NDIMS = 2

  # readin all the information from the mesh file into a string array
  file_lines = readlines(open(filename))

  # readin the number of nodes, number of interfaces, number of elements and local polynomial degree
  current_line  = split(file_lines[1])
  n_corners     = parse(Int64,current_line[1])
  n_interfaces  = parse(Int64,current_line[2])
  n_elements    = parse(Int64,current_line[3])
  mesh_poly_deg = parse(Int64,current_line[4])

  # create the Chebyshev-Gauss-Lobatto nodes used to represent any curved boundaries
  cheby_nodes, _ = chebyshev_gauss_lobatto_nodes_weights(mesh_poly_deg + 1)

  # declare memory and work arrays for the pieces of the mesh
  corner_nodes = Array{RealT}(undef, n_corners, 2)

  # temporary container to for the interface neighbour information and connectivity
  edge_info  = Array{Int64}(undef, (6, n_interfaces))

  element_ids = Array{Int64}(undef, 4)
  elements    = UnstructuredElementContainer2D(RealT, nvars, polydeg, n_elements)

  GammaCurves = Array{GammaCurve, 1}(undef, 4)

  # readin an store the nodes that dictate the corners of the elements needed to construct the
  # element geometry terms
  file_idx = 2
  for j in 1:n_corners
    current_line      = split(file_lines[file_idx])
    corner_nodes[j,1] = parse(RealT, current_line[1])
    corner_nodes[j,2] = parse(RealT, current_line[2])
    file_idx         += 1
  end

  # readin an store the nodes that dictate the interfaces, neighbour data, and orientations contains
  # the following:
  #    edge_info[1] = start node ID
  #    edge_info[2] = end node ID
  #    edge_info[3] = ID of the primary element
  #    edge_info[4] = ID of the secondary element (if 0 then it is a physical boundary)
  #    edge_info[5] = local side ID on the primary element
  #    edge_info[6] = local side ID on the secondary element
  # Interface containers constructed below after the elements as we need the boundary names
  n_boundary = 0
  for j in 1:n_interfaces
    current_line   = split(file_lines[file_idx])
    edge_info[1,j] = parse(Int64,current_line[1])
    edge_info[2,j] = parse(Int64,current_line[2])
    edge_info[3,j] = parse(Int64,current_line[3])
    edge_info[4,j] = parse(Int64,current_line[4])
    edge_info[5,j] = parse(Int64,current_line[5])
    edge_info[6,j] = parse(Int64,current_line[6])
    if edge_info[4,j] == 0
      n_boundary += 1
    end
    file_idx += 1
  end

  # readin an store the nodes that dictate the corners of the elements
  curved_check   = Array{Int64}(undef, 4)
  cornerNodeVals = Array{RealT}(undef, 4, 2)
  tempNodes      = Array{RealT}(undef, 4, 2)
  CurveVals      = Array{RealT}(undef, mesh_poly_deg + 1, 2)
  bndy_names     = Array{String}(undef, 4, n_elements)
  for j in 1:n_elements
    # pull the corner node IDs
    current_line   = split(file_lines[file_idx])
    element_ids[1] = parse(Int64,current_line[1])
    element_ids[2] = parse(Int64,current_line[2])
    element_ids[3] = parse(Int64,current_line[3])
    element_ids[4] = parse(Int64,current_line[4])
    for i in 1:4
      # pull the (x,y) values of these corners out of the nodes array
      cornerNodeVals[i,:] = corner_nodes[element_ids[i],:]
    end
    # pull the information to check if boundary is curved in order to readin additional data
    file_idx += 1
    current_line    = split(file_lines[file_idx])
    curved_check[1] = parse(Int64,current_line[1])
    curved_check[2] = parse(Int64,current_line[2])
    curved_check[3] = parse(Int64,current_line[3])
    curved_check[4] = parse(Int64,current_line[4])
    if sum(curved_check) == 0
      # quadrilateral element is straight sided
      file_idx  += 1
      bndy_names[:,j] = split(file_lines[file_idx])
      # construct quadrilateral geometry using only corner information
      elements.geometry[j] = ElementGeometry(RealT, polydeg, dg_nodes, cornerNodeVals)
    else
      # quadrilateral element has at least one curved side
      m1 = 1
      m2 = 2
      tempNodes[1,:] = cornerNodeVals[4,:]
      tempNodes[2,:] = cornerNodeVals[2,:]
      tempNodes[3,:] = cornerNodeVals[3,:]
      tempNodes[4,:] = cornerNodeVals[1,:]
      for i in 1:4
        if curved_check[i] == 0
          # when curved_check[i] is 0 then the "curve" from cornerNode(i) to cornerNode(i+1) is a
          # straight line. So we must construct the interpolant for this line
          for k in 1:mesh_poly_deg + 1
            CurveVals[k,1] = tempNodes[m1,1] + 0.5 * (cheby_nodes[k] + 1.0) * (  tempNodes[m2,1]
                                                                               - tempNodes[m1,1] )
            CurveVals[k,2] = tempNodes[m1,2] + 0.5 * (cheby_nodes[k] + 1.0) * (  tempNodes[m2,2]
                                                                               - tempNodes[m1,2] )
          end
        else
          # when curved_check[i] is 1 this curved boundary information is supplied by the mesh
          # generator. So we just read it into a work array
          for k in 1:mesh_poly_deg + 1
            file_idx      += 1
            current_line   = split(file_lines[file_idx])
            CurveVals[k,1] = parse(RealT,current_line[1])
            CurveVals[k,2] = parse(RealT,current_line[2])
          end
        end
        # construct the curve interpolant for the current side
        GammaCurves[i] = GammaCurve(RealT, mesh_poly_deg, CurveVals[:,1], CurveVals[:,2])
        # indexing update that contains a "flip" to ensure correct element orientation
        # when we construct the straight line "curves" when curved_check[i] == 0
        m1 += 1
        if i == 3
          m2 = 1
        else
          m2 += 1
        end
      end
      # finally read in the boundary names where "---" means an internal connection
      file_idx  += 1
      bndy_names[:,j] = split(file_lines[file_idx])
      # construct quadrilateral geometry using all the curve information
      elements.geometry[j] = ElementGeometry(RealT, polydeg, dg_nodes, GammaCurves)
    end
    # one last increment to the global index to read the next piece of element information
    file_idx += 1
  end

  # construct the interior interface and boundary interface neighbour information for either a
  # pully periodic mesh or a general mesh with physical boundaries
  if typeof(boundary_conditions) == Trixi.BoundaryConditionPeriodic
    interfaces = init_interfaces(RealT, edge_info, bndy_names, nvars, polydeg,
                                 n_interfaces, n_elements)
    boundaries = UnstructuredBoundaryContainer2D{RealT, nvars, poly_deg}(0)
  else
    interfaces = init_interfaces(RealT, edge_info, nvars, polydeg, n_interfaces - n_boundary)
    boundaries = init_boundaries(RealT, edge_info, bndy_names, nvars, polydeg, n_boundary)
  end

  return UnstructuredMesh{NDIMS, RealT}(filename, interfaces, boundaries, elements)
end

@inline Base.ndims(::UnstructuredMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::UnstructuredMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT
