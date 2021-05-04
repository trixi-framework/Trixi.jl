# mesh data structure
struct UnstructuredMesh{NDIMS,Tv,Ti}
    VXYZ::NTuple{NDIMS,Tv}
    EToV::Matrix{Ti}
end

function Base.show(io::IO, mesh::UnstructuredMesh{NDIMS}) where {NDIMS}
    @nospecialize mesh
    println("Unstructured mesh in $NDIMS dimensions.")
end

