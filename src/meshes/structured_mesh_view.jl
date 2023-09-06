mutable struct StructuredMeshView{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    parent::StructuredMesh{NDIMS, RealT}
    mapping::Any # Not relevant for performance
end

function StructuredMeshView(parent::StructuredMesh{NDIMS, RealT}) where {NDIMS, RealT}
    return StructuredMeshView{NDIMS, RealT}(parent, parent.mapping)
end

# Check if mesh is periodic
isperiodic(mesh::StructuredMeshView) = all(mesh.parent.periodicity)
isperiodic(mesh::StructuredMeshView, dimension) = mesh.parent.periodicity[dimension]

@inline Base.ndims(::StructuredMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::StructuredMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
Base.size(mesh::StructuredMeshView) = mesh.parent.cells_per_dimension
Base.size(mesh::StructuredMeshView, i) = mesh.parent.cells_per_dimension[i]
Base.axes(mesh::StructuredMeshView) = map(Base.OneTo, mesh.parent.cells_per_dimension)
Base.axes(mesh::StructuredMeshView, i) = Base.OneTo(mesh.parent.cells_per_dimension[i])
