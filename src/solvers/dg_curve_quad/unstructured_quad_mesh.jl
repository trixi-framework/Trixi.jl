
struct UnstructuredQuadMesh{RealT<:Real, NNODES, NCORNERS, NEDGES, NELEMENTS}
  mesh_filename::String
  corners ::SArray{Tuple{NCORNERS, 2}, RealT}
  edges   ::SVector{NEDGES, GeneralInterfaceContainer2D}
  elements::SVector{NELEMENTS, GeneralElementContainer2D}
end


# constructor for an unstructured mesh read in from a file
function UnstructuredQuadMesh(RealT, filename, polydeg, dg_nodes)

  # readin all the information from the mesh file into a string array
  file_lines = readlines(open(filename))

  # readin the number of nodes, number of edges, number of elements and local polynomial degree
  current_line  = split(file_lines[1])
  n_corners     = parse(Int64,current_line[1])
  n_edges       = parse(Int64,current_line[2])
  n_elements    = parse(Int64,current_line[3])
  mesh_poly_deg = parse(Int64,current_line[4])

  # create the Chebyshev-Gauss-Lobatto nodes used to represent any curved boundaries
  cheby_nodes, _ = chebyshev_gauss_lobatto_nodes_weights(mesh_poly_deg + 1)

  # declare memory and work arrays for the pieces of the mesh
  corner_nodes = Array{RealT}(undef, n_corners, 2)

  edge_info = Array{Int64}(undef, 6) # temporary container to readin edge information
  edges     = Array{GeneralInterfaceContainer2D, 1}(undef, n_edges)

  element_ids = Array{Int64}(undef, 4)
  elements    = Array{GeneralElementContainer2D, 1}(undef, n_elements)

  GammaCurves = Array{GammaCurve, 1}(undef, 4)

  # readin an store the nodes that dictate the corners of the elements
  file_idx = 2
  for j in 1:n_corners
    current_line      = split(file_lines[file_idx])
    corner_nodes[j,1] = parse(RealT, current_line[1])
    corner_nodes[j,2] = parse(RealT, current_line[2])
    file_idx         += 1
  end

  # readin an store the nodes that dictate the edges and their neighbours
  for j in 1:n_edges
    current_line = split(file_lines[file_idx])
    edge_info[1] = parse(Int64,current_line[1])
    edge_info[2] = parse(Int64,current_line[2])
    edge_info[3] = parse(Int64,current_line[3])
    edge_info[4] = parse(Int64,current_line[4])
    edge_info[5] = parse(Int64,current_line[5])
    edge_info[6] = parse(Int64,current_line[6])
    edges[j] = GeneralInterfaceContainer2D(edge_info, polydeg)
    file_idx += 1
  end

  # readin an store the nodes that dictate the corners of the elements
  curved_check   = Array{Int64}(undef, 4)
  cornerNodeVals = Array{RealT}(undef, 4, 2)
  tempNodes      = Array{RealT}(undef, 4, 2)
  CurveVals      = Array{RealT}(undef, mesh_poly_deg + 1, 2)
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
      bndy_names = split(file_lines[file_idx])
      # construct quadrilateral geometry using only corner information
      elements[j] = GeneralElementContainer2D(RealT, polydeg, dg_nodes, cornerNodeVals, bndy_names)
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
            CurveVals[k,1] = tempNodes[m1,1] + 0.5 * (cheby_nodes[k] + 1.0) * (tempNodes[m2,1] - tempNodes[m1,1])
            CurveVals[k,2] = tempNodes[m1,2] + 0.5 * (cheby_nodes[k] + 1.0) * (tempNodes[m2,2] - tempNodes[m1,2])
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
        # when we construct the straight line "curves"
        m1 += 1
        if i == 3
          m2 = 1
        else
          m2 += 1
        end
      end
      # finally read in the boundary names where "---" means an internal connection
      file_idx  += 1
      bndy_names = split(file_lines[file_idx])
      # construct quadrilateral geometry using all the curve information
      elements[j] = GeneralElementContainer2D(RealT, polydeg, dg_nodes, GammaCurves, bndy_names)
    end
    # one last increment to the global index to read in the next element information
    file_idx += 1
  end

  return UnstructuredQuadMesh{RealT, polydeg+1, n_corners, n_edges, n_elements}(
         filename, corner_nodes, edges, elements)
end
