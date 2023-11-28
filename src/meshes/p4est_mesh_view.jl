@muladd begin
#! format: noindent

mutable struct P4estMeshView{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    parent::P4estMesh{NDIMS, RealT}
    mapping::Any # Not relevant for performance
    index_min::NTuple{NDIMS, Int}
    index_max::NTuple{NDIMS, Int}
end

function P4estMeshView(parent::P4estMesh{NDIMS, RealT};
                        index_min = ntuple(_ -> 1, Val(NDIMS)),
                        index_max = size(parent)) where {NDIMS, RealT}
    @assert index_min <= index_max
    @assert all(index_min .> 0)
    @assert index_max <= size(parent)

    return P4estMeshView{NDIMS, RealT}(parent, parent.mapping, index_min,
                                       index_max)
end

# Check if mesh is periodic
function isperiodic(mesh::P4estMeshView)
    @unpack parent = mesh
    return isperiodic(parent) && size(parent) == size(mesh)
end

function isperiodic(mesh::P4estMeshView, dimension)
    @unpack parent, index_min, index_max = mesh
    return (isperiodic(parent, dimension) &&
            index_min[dimension] == 1 &&
            index_max[dimension] == size(parent, dimension))
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
function Base.size(mesh::P4estMeshView)
    @unpack index_min, index_max = mesh
    return index_max .- index_min .+ 1
end
function Base.size(mesh::P4estMeshView, i)
    @unpack index_min, index_max = mesh
    return index_max[i] - index_min[i] + 1
end
Base.axes(mesh::P4estMeshView) = map(Base.OneTo, size(mesh))
Base.axes(mesh::P4estMeshView, i) = Base.OneTo(size(mesh, i))

# function calc_node_coordinates!(node_coordinates, element,
#                                 cell_x, cell_y, mapping,
#                                 mesh::P4estMeshView{2},
#                                 # basis::LobattoLegendreBasis)
#                                 basis)
#     @unpack nodes = basis
#     @unpack parent, index_min, index_max = mesh

#     # Get cell length in reference mesh
#     dx = 2 / size(parent, 1)
#     dy = 2 / size(parent, 2)

#     # Calculate index offsets with respect to parent
#     parent_offset_x = index_min[1] - 1
#     parent_offset_y = index_min[2] - 1

#     # Calculate node coordinates of reference mesh
#     cell_x_offset = -1 + (cell_x - 1 + parent_offset_x) * dx + dx / 2
#     cell_y_offset = -1 + (cell_y - 1 + parent_offset_y) * dy + dy / 2
#     p4est_tree_t
#     for j in eachnode(basis), i in eachnode(basis)
#         # node_coordinates are the mapped reference node_coordinates
#         node_coordinates[:, i, j, element] .= mapping(cell_x_offset + dx / 2 * nodes[i],
#                                                       cell_y_offset + dy / 2 * nodes[j])
#     end
# end

# Interpolate tree_node_coordinates to each quadrant at the specified nodes
function calc_node_coordinates!(node_coordinates,
        mesh::P4estMeshView{2},
        nodes::AbstractVector)
    @unpack parent, index_min, index_max = mesh

    # We use `StrideArray`s here since these buffers are used in performance-critical
    # places and the additional information passed to the compiler makes them faster
    # than native `Array`s.
    tmp1 = StrideArray(undef, real(parent),
    StaticInt(2), static_length(nodes), static_length(parent.nodes))
    matrix1 = StrideArray(undef, real(parent),
    static_length(nodes), static_length(parent.nodes))
    matrix2 = similar(matrix1)
    baryweights_in = barycentric_weights(parent.nodes)

    # Macros from `p4est`
    p4est_root_len = 1 << P4EST_MAXLEVEL
    p4est_quadrant_len(l) = 1 << (P4EST_MAXLEVEL - l)

    trees = unsafe_wrap_sc(p4est_tree_t, parent.p4est.trees)
    @autoinfiltrate

    for tree in eachindex(trees)
        offset = trees[tree].quadrants_offset
        quadrants = unsafe_wrap_sc(p4est_quadrant_t, trees[tree].quadrants)

        for i in eachindex(quadrants)
            element = offset + i
            quad = quadrants[i]

            quad_length = p4est_quadrant_len(quad.level) / p4est_root_len

            nodes_out_x = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
            quad.x / p4est_root_len) .- 1
            nodes_out_y = 2 * (quad_length * 1 / 2 * (nodes .+ 1) .+
            quad.y / p4est_root_len) .- 1
            polynomial_interpolation_matrix!(matrix1, parent.nodes, nodes_out_x,
                            baryweights_in)
            polynomial_interpolation_matrix!(matrix2, parent.nodes, nodes_out_y,
                            baryweights_in)

            multiply_dimensionwise!(view(node_coordinates, :, :, :, element),
                    matrix1, matrix2,
                    view(parent.tree_node_coordinates, :, :, :, tree),
                    tmp1)
        end
    end

    return node_coordinates
end
end # @muladd
