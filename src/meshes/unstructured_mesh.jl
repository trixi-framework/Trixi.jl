# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    UnstructuredMesh2D <: AbstractMesh{2}

An unstructured (possibly curved) quadrilateral mesh.

    UnstructuredMesh2D(filename; RealT=Float64, periodicity=false)

All mesh information, neighbour coupling, and boundary curve information is read in
from a mesh file `filename`.
"""
mutable struct UnstructuredMesh2D{RealT <: Real,
                                  CurvedSurfaceT <: CurvedSurface{RealT}} <:
               AbstractMesh{2}
    filename              :: String
    n_corners             :: Int
    n_surfaces            :: Int # total number of surfaces
    n_interfaces          :: Int # number of interior surfaces
    n_boundaries          :: Int # number of surfaces on the physical boundary
    n_elements            :: Int
    polydeg               :: Int
    corners               :: Array{RealT, 2}  # [ndims, n_corners]
    neighbour_information :: Array{Int, 2}  # [neighbour node/element/edge ids, n_surfaces]
    boundary_names        :: Array{Symbol, 2} # [local sides, n_elements]
    periodicity           :: Bool
    element_node_ids      :: Array{Int, 2} # [node ids, n_elements]
    element_is_curved     :: Vector{Bool}
    surface_curves        :: Array{CurvedSurfaceT, 2} # [local sides, n_elements]
    current_filename      :: String
    unsaved_changes       :: Bool # if true, the mesh will be saved for plotting
end

# constructor for an unstructured mesh read in from a file
# TODO: this mesh file parsing and construction of the mesh skeleton can likely be improved in terms
#       of performance
function UnstructuredMesh2D(filename; RealT = Float64, periodicity = false,
                            unsaved_changes = true)

    # readin all the information from the mesh file into a string array
    file_lines = readlines(open(filename))

    # readin the number of nodes, number of interfaces, number of elements and local polynomial degree
    current_line = split(file_lines[2])
    n_corners = parse(Int, current_line[1])
    n_surfaces = parse(Int, current_line[2])
    n_elements = parse(Int, current_line[3])
    mesh_polydeg = parse(Int, current_line[4])

    mesh_nnodes = mesh_polydeg + 1

    # The types of structs used in the following depend on information read from
    # the mesh file. Thus, this cannot be type stable at all. Hence, we allocate
    # the memory now and introduce a function barrier before continuing to read
    # data from the file.
    corner_nodes = Array{RealT}(undef, (2, n_corners))
    interface_info = Array{Int}(undef, (6, n_surfaces))
    element_node_ids = Array{Int}(undef, (4, n_elements))
    curved_check = Vector{Int}(undef, 4)
    quad_corners = Array{RealT}(undef, (4, 2))
    quad_corners_flipped = Array{RealT}(undef, (4, 2))
    curve_values = Array{RealT}(undef, (mesh_nnodes, 2))
    element_is_curved = Array{Bool}(undef, n_elements)
    CurvedSurfaceT = CurvedSurface{RealT}
    surface_curves = Array{CurvedSurfaceT}(undef, (4, n_elements))
    boundary_names = Array{Symbol}(undef, (4, n_elements))

    # create the Chebyshev-Gauss-Lobatto nodes used to represent any curved boundaries that are
    # required to construct the sides
    cheby_nodes_, _ = chebyshev_gauss_lobatto_nodes_weights(mesh_nnodes)
    bary_weights_ = barycentric_weights(cheby_nodes_)
    cheby_nodes = SVector{mesh_nnodes}(cheby_nodes_)
    bary_weights = SVector{mesh_nnodes}(bary_weights_)

    arrays = (; corner_nodes, interface_info, element_node_ids, curved_check,
              quad_corners, quad_corners_flipped, curve_values,
              element_is_curved, surface_curves, boundary_names)
    counters = (; n_corners, n_surfaces, n_elements)

    n_boundaries = parse_mesh_file!(arrays, RealT, CurvedSurfaceT, file_lines, counters,
                                    cheby_nodes, bary_weights)

    # get the number of internal interfaces in the mesh
    if periodicity
        n_interfaces = n_surfaces
        n_boundaries = 0
    else
        n_interfaces = n_surfaces - n_boundaries
    end

    return UnstructuredMesh2D{RealT, CurvedSurfaceT}(filename, n_corners, n_surfaces,
                                                     n_interfaces, n_boundaries,
                                                     n_elements, mesh_polydeg,
                                                     corner_nodes,
                                                     interface_info, boundary_names,
                                                     periodicity,
                                                     element_node_ids,
                                                     element_is_curved, surface_curves,
                                                     "", unsaved_changes)
end

function parse_mesh_file!(arrays, RealT, CurvedSurfaceT, file_lines, counters,
                          cheby_nodes, bary_weights)
    @unpack (corner_nodes, interface_info, element_node_ids, curved_check,
    quad_corners, quad_corners_flipped, curve_values,
    element_is_curved, surface_curves, boundary_names) = arrays
    @unpack n_corners, n_surfaces, n_elements = counters
    mesh_nnodes = length(cheby_nodes)

    # counter to step through the mesh file line by line
    file_idx = 3

    # readin an store the nodes that dictate the corners of the elements needed to construct the
    # element geometry terms
    for j in 1:n_corners
        current_line = split(file_lines[file_idx])
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
    n_boundaries = 0
    for j in 1:n_surfaces
        current_line = split(file_lines[file_idx])
        interface_info[1, j] = parse(Int, current_line[1])
        interface_info[2, j] = parse(Int, current_line[2])
        interface_info[3, j] = parse(Int, current_line[3])
        interface_info[4, j] = parse(Int, current_line[4])
        interface_info[5, j] = parse(Int, current_line[5])
        interface_info[6, j] = parse(Int, current_line[6])

        # count the number of physical boundaries
        if interface_info[4, j] == 0
            n_boundaries += 1
        end
        file_idx += 1
    end

    # work arrays to pull to correct corners of a given element (agnostic to curvature) and local
    # copies of the curved boundary information

    # readin an store the curved boundary information of the elements

    for j in 1:n_elements
        # pull the corner node IDs
        current_line = split(file_lines[file_idx])
        element_node_ids[1, j] = parse(Int, current_line[1])
        element_node_ids[2, j] = parse(Int, current_line[2])
        element_node_ids[3, j] = parse(Int, current_line[3])
        element_node_ids[4, j] = parse(Int, current_line[4])
        for i in 1:4
            # pull the (x,y) values of these corners out of the nodes array
            quad_corners[i, :] .= corner_nodes[:, element_node_ids[i, j]]
        end
        # pull the information to check if boundary is curved in order to read in additional data
        file_idx += 1
        current_line = split(file_lines[file_idx])
        curved_check[1] = parse(Int, current_line[1])
        curved_check[2] = parse(Int, current_line[2])
        curved_check[3] = parse(Int, current_line[3])
        curved_check[4] = parse(Int, current_line[4])
        if sum(curved_check) == 0
            # quadrilateral element is straight sided
            element_is_curved[j] = false
            file_idx += 1
            # read all the boundary names
            boundary_names[:, j] = map(Symbol, split(file_lines[file_idx]))
        else
            # quadrilateral element has at least one curved side
            element_is_curved[j] = true

            # flip node ordering to make sure the element is right-handed for the interpolations
            m1 = 1
            m2 = 2
            @views quad_corners_flipped[1, :] .= quad_corners[4, :]
            @views quad_corners_flipped[2, :] .= quad_corners[2, :]
            @views quad_corners_flipped[3, :] .= quad_corners[3, :]
            @views quad_corners_flipped[4, :] .= quad_corners[1, :]
            for i in 1:4
                if curved_check[i] == 0
                    # when curved_check[i] is 0 then the "curve" from corner `i` to corner `i+1` is a
                    # straight line. So we must construct the interpolant for this line
                    for k in 1:mesh_nnodes
                        curve_values[k, 1] = linear_interpolate(cheby_nodes[k],
                                                                quad_corners_flipped[m1,
                                                                                     1],
                                                                quad_corners_flipped[m2,
                                                                                     1])
                        curve_values[k, 2] = linear_interpolate(cheby_nodes[k],
                                                                quad_corners_flipped[m1,
                                                                                     2],
                                                                quad_corners_flipped[m2,
                                                                                     2])
                    end
                else
                    # when curved_check[i] is 1 this curved boundary information is supplied by the mesh
                    # generator. So we just read it into a work array
                    for k in 1:mesh_nnodes
                        file_idx += 1
                        current_line = split(file_lines[file_idx])
                        curve_values[k, 1] = parse(RealT, current_line[1])
                        curve_values[k, 2] = parse(RealT, current_line[2])
                    end
                end
                # construct the curve interpolant for the current side
                surface_curves[i, j] = CurvedSurfaceT(cheby_nodes, bary_weights,
                                                      copy(curve_values))
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
            file_idx += 1
            boundary_names[:, j] = map(Symbol, split(file_lines[file_idx]))
        end
        # one last increment to the global index to read the next piece of element information
        file_idx += 1
    end

    return n_boundaries
end

@inline Base.ndims(::UnstructuredMesh2D) = 2
@inline Base.real(::UnstructuredMesh2D{RealT}) where {RealT} = RealT

# Check if mesh is periodic
isperiodic(mesh::UnstructuredMesh2D) = mesh.periodicity

Base.length(mesh::UnstructuredMesh2D) = mesh.n_elements

function Base.show(io::IO,
                   ::UnstructuredMesh2D{RealT, CurvedSurfaceT}) where {RealT,
                                                                       CurvedSurfaceT}
    print(io, "UnstructuredMesh2D{2, ", RealT, ", ", CurvedSurfaceT, "}")
end

function Base.show(io::IO, ::MIME"text/plain",
                   mesh::UnstructuredMesh2D{RealT, CurvedSurfaceT}) where {RealT,
                                                                           CurvedSurfaceT
                                                                           }
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "UnstructuredMesh2D{" * string(2) * ", " * string(RealT) * ", " *
                       string(CurvedSurfaceT) * "}")
        summary_line(io, "mesh file", mesh.filename)
        summary_line(io, "number of elements", length(mesh))
        summary_line(io, "faces", mesh.n_surfaces)
        summary_line(io, "mesh polynomial degree", mesh.polydeg)
        summary_footer(io)
    end
end
end # @muladd
