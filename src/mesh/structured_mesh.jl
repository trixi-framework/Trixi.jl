struct StructuredMesh{RealT<:Real, NDIMS}
    size::Tuple{Vararg{Int64, NDIMS}}
    coordinates_min::Tuple{Vararg{RealT, NDIMS}}
    coordinates_max::Tuple{Vararg{RealT, NDIMS}}
end

function StructuredMesh{RealT}(size, coordinates_min, coordinates_max) where {RealT<:Real}
    NDIMS = length(size)

    return StructuredMesh{RealT, NDIMS}(size, coordinates_min, coordinates_max)
end

@inline Base.ndims(mesh::StructuredMesh{RealT, NDIMS}) where {RealT, NDIMS} = NDIMS