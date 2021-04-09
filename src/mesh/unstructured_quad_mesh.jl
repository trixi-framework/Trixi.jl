"""
    UnstructuredQuadMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

An unstructured (possibly curved) quadrilateral mesh.

All mesh information, neighbour coupling, and boundary curve information is read in from a mesh file

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
struct UnstructuredQuadMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}
  filename             ::String
  n_corners            ::Int64
  n_interfaces         ::Int64
  n_boundary           ::Int64
  n_elements           ::Int64
  poly_deg             ::Int64
  corners              ::Array{RealT, 2}  # [ndims, n_corners]
  neighbour_information::Array{Int64, 2}  # [neighbour node/element/edge ids, n_interfaces]
  boundary_names       ::Array{String, 2} # [local sides, n_elements]
  periodicity          ::Bool
  element_node_ids     ::Array{Int64, 2} # [node ids, n_elements]
  element_is_curved    ::Vector{Bool}
  elements_curves      ::Array{GammaCurve, 2} # [local sides, n_elements]
end


# constructor for an unstructured mesh read in from a file
# TODO: currently only full periodcity is supported
function UnstructuredQuadMesh(RealT, filename, periodic)

  NDIMS = 2

  # readin all the information from the mesh file into a string array
  file_lines = readlines(open(filename))

  # readin the number of nodes, number of interfaces, number of elements and local polynomial degree
  current_line  = split(file_lines[1])
  n_corners     = parse(Int64,current_line[1])
  n_interfaces  = parse(Int64,current_line[2])
  n_elements    = parse(Int64,current_line[3])
  mesh_poly_deg = parse(Int64,current_line[4])

  mesh_nnodes = mesh_poly_deg + 1

  # counter to step through the mesh file line by line
  file_idx = 2

  # readin an store the nodes that dictate the corners of the elements needed to construct the
  # element geometry terms
  corner_nodes = Array{RealT}(undef, (2, n_corners))
  for j in 1:n_corners
    current_line       = split(file_lines[file_idx])
    corner_nodes[1, j] = parse(RealT, current_line[1])
    corner_nodes[2, j] = parse(RealT, current_line[2])
    file_idx += 1
  end

  # readin an store the nodes that dictate the interfaces, neighbour data, and orientations contains
  # the following:
  #    interface_info[1] = start node ID
  #    interface_info[2] = end node ID
  #    interface_info[3] = ID of the primary element
  #    interface_info[4] = ID of the secondary element (if 0 then it is a physical boundary)
  #    interface_info[5] = local side ID on the primary element
  #    interface_info[6] = local side ID on the secondary element
  # container to for the interface neighbour information and connectivity
  interface_info  = Array{Int64}(undef, (6, n_interfaces))
  n_boundary = 0
  for j in 1:n_interfaces
    current_line   = split(file_lines[file_idx])
    interface_info[1, j] = parse(Int64,current_line[1])
    interface_info[2, j] = parse(Int64,current_line[2])
    interface_info[3, j] = parse(Int64,current_line[3])
    interface_info[4, j] = parse(Int64,current_line[4])
    interface_info[5, j] = parse(Int64,current_line[5])
    interface_info[6, j] = parse(Int64,current_line[6])

    # count the number of physical boundaries
    if interface_info[4,j] == 0
      n_boundary += 1
    end
    file_idx += 1
  end

  # work arrays to pull to correct corners of a given element (agnostic to curvature) and local
  # copies of the curved boundary information
  element_node_ids = Array{Int64}(undef, (4, n_elements))
  curved_check     = Vector{Int64}(undef, 4)
  cornerNodeVals   = Array{RealT}(undef, (4, 2))
  tempNodes        = Array{RealT}(undef, (4, 2))
  CurveVals        = Array{RealT}(undef, (mesh_nnodes, 2))

  # readin an store the curved boundary information of the elements
  element_is_curved = Array{Bool}(undef, n_elements)
  element_curves    = Array{GammaCurve}(undef, (4, n_elements))
  bndy_names        = Array{String}(undef, (4, n_elements))

  # create the Chebyshev-Gauss-Lobatto nodes used to represent any curved boundaries that are
  # required to construct the sides
  cheby_nodes, _ = chebyshev_gauss_lobatto_nodes_weights(mesh_nnodes)

  for j in 1:n_elements
    # pull the corner node IDs
    current_line           = split(file_lines[file_idx])
    element_node_ids[1, j] = parse(Int64,current_line[1])
    element_node_ids[2, j] = parse(Int64,current_line[2])
    element_node_ids[3, j] = parse(Int64,current_line[3])
    element_node_ids[4, j] = parse(Int64,current_line[4])
    for i in 1:4
      # pull the (x,y) values of these corners out of the nodes array
      cornerNodeVals[i, :] .= corner_nodes[:, element_node_ids[i, j]]
    end
    # pull the information to check if boundary is curved in order to read in additional data
    file_idx += 1
    current_line    = split(file_lines[file_idx])
    curved_check[1] = parse(Int64,current_line[1])
    curved_check[2] = parse(Int64,current_line[2])
    curved_check[3] = parse(Int64,current_line[3])
    curved_check[4] = parse(Int64,current_line[4])
    if sum(curved_check) == 0
      # quadrilateral element is straight sided
      element_is_curved[j] = false
      file_idx  += 1
      # read all the boundary names
      bndy_names[:, j] = split(file_lines[file_idx])
    else
      # quadrilateral element has at least one curved side
      element_is_curved[j] = true

      # flip node ordering to make sure the element is right-handed for the interpolations
      m1 = 1
      m2 = 2
      tempNodes[1, :] = cornerNodeVals[4, :]
      tempNodes[2, :] = cornerNodeVals[2, :]
      tempNodes[3, :] = cornerNodeVals[3, :]
      tempNodes[4, :] = cornerNodeVals[1, :]
      for i in 1:4
        if curved_check[i] == 0
          # when curved_check[i] is 0 then the "curve" from cornerNode(i) to cornerNode(i+1) is a
          # straight line. So we must construct the interpolant for this line
          for k in 1:mesh_nnodes
            CurveVals[k, 1] = tempNodes[m1, 1] + 0.5 * (cheby_nodes[k] + 1.0) * (  tempNodes[m2, 1]
                                                                                 - tempNodes[m1, 1])
            CurveVals[k, 2] = tempNodes[m1, 2] + 0.5 * (cheby_nodes[k] + 1.0) * (  tempNodes[m2, 2]
                                                                                 - tempNodes[m1, 2])
          end
        else
          # when curved_check[i] is 1 this curved boundary information is supplied by the mesh
          # generator. So we just read it into a work array
          for k in 1:mesh_nnodes
            file_idx      += 1
            current_line   = split(file_lines[file_idx])
            CurveVals[k, 1] = parse(RealT,current_line[1])
            CurveVals[k, 2] = parse(RealT,current_line[2])
          end
        end
        # construct the curve interpolant for the current side
        element_curves[i, j] = GammaCurve(RealT, mesh_poly_deg, CurveVals[:,1], CurveVals[:,2])
        # indexing update that contains a "flip" to ensure correct element orientation
        # if we need to construct the straight line "curves" when curved_check[i] == 0
        m1 += 1
        if i == 3
          m2 = 1
        else
          m2 += 1
        end
      end
      # finally read in the boundary names where "---" means an internal connection
      file_idx  += 1
      bndy_names[:, j] = split(file_lines[file_idx])
    end
    # one last increment to the global index to read the next piece of element information
    file_idx += 1
  end

  return UnstructuredQuadMesh{NDIMS, RealT}(filename, n_corners, n_interfaces, n_boundary,
                                            n_elements, mesh_poly_deg, corner_nodes,
                                            interface_info, bndy_names, periodic,
                                            element_node_ids, element_is_curved, element_curves)
end

@inline Base.ndims(::UnstructuredQuadMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::UnstructuredQuadMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT

# Check if mesh is periodic
isperiodic(mesh::UnstructuredQuadMesh) = mesh.periodicity

Base.size(mesh::UnstructuredQuadMesh) = mesh.n_elements


function Base.show(io::IO, ::UnstructuredQuadMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "UnstructuredQuadMesh{", NDIMS, ", ", RealT, "}")
end


function Base.show(io::IO, ::MIME"text/plain", mesh::UnstructuredQuadMesh{NDIMS, RealT}) where {NDIMS, RealT}
  if get(io, :compact, false)
    show(io, mesh)
  else
    summary_header(io, "UnstructuredQuadMesh{" * string(NDIMS) * ", " * string(RealT) * "}")
    summary_line(io, "mesh file", mesh.filename)
    summary_line(io, "size", size(mesh))
    summary_line(io, "faces", mesh.n_interfaces)
    summary_line(io, "mesh polynomial degree", mesh.poly_deg)
    summary_footer(io)
  end
end
