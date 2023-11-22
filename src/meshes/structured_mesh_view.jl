@muladd begin
#! format: noindent

mutable struct StructuredMeshView{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    parent::StructuredMesh{NDIMS, RealT}
    mapping::Any # Not relevant for performance
    index_min::NTuple{NDIMS, Int}
    index_max::NTuple{NDIMS, Int}
end

function StructuredMeshView(parent::StructuredMesh{NDIMS, RealT};
                            index_min, index_max) where {NDIMS, RealT}
    @assert index_min <= index_max
    @assert all(index_min .> 0)
    @assert index_max <= size(parent)

    return StructuredMeshView{NDIMS, RealT}(parent, parent.mapping, index_min,
                                            index_max)
end

function StructuredMeshView(parent::StructuredMesh{NDIMS, RealT}) where {NDIMS, RealT}
    return StructuredMeshView(parent, ntuple(_ -> 1, Val(NDIMS)), size(parent))
end

# Check if mesh is periodic
function isperiodic(mesh::StructuredMeshView)
    @unpack parent = mesh
    return isperiodic(parent) && size(parent) == size(mesh)
end

function isperiodic(mesh::StructuredMeshView, dimension)
    @unpack parent, index_min, index_max = mesh
    return (isperiodic(parent, dimension) &&
            index_min[dimension] == 1 &&
            index_max[dimension] == size(parent, dimension))
end

@inline Base.ndims(::StructuredMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::StructuredMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
function Base.size(mesh::StructuredMeshView)
    @unpack index_min, index_max = mesh
    return index_max .- index_min .+ 1
end
function Base.size(mesh::StructuredMeshView, i)
    @unpack index_min, index_max = mesh
    return index_max[i] - index_min[i] + 1
end
Base.axes(mesh::StructuredMeshView) = map(Base.OneTo, size(mesh))
Base.axes(mesh::StructuredMeshView, i) = Base.OneTo(size(mesh, i))

function calc_node_coordinates!(node_coordinates, element,
                                cell_x, cell_y, mapping,
                                mesh::StructuredMeshView{2},
                                #                                basis::LobattoLegendreBasis)
                                basis)
    @unpack nodes = basis
    @unpack parent, index_min, index_max = mesh

    # Get cell length in reference mesh
    dx = 2 / size(parent, 1)
    dy = 2 / size(parent, 2)

    # Calculate index offsets with respect to parent
    parent_offset_x = index_min[1] - 1
    parent_offset_y = index_min[2] - 1

    # Calculate node coordinates of reference mesh
    cell_x_offset = -1 + (cell_x - 1 + parent_offset_x) * dx + dx / 2
    cell_y_offset = -1 + (cell_y - 1 + parent_offset_y) * dy + dy / 2

    for j in eachnode(basis), i in eachnode(basis)
        # node_coordinates are the mapped reference node_coordinates
        node_coordinates[:, i, j, element] .= mapping(cell_x_offset + dx / 2 * nodes[i],
                                                      cell_y_offset + dy / 2 * nodes[j])
    end
end
end # @muladd
