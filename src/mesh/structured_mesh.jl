struct StructuredMesh{RealT<:Real, NDIMS}
    size::Vector{Int64}
    coordinates_min::Vector{RealT}
    coordinates_max::Vector{RealT}
end

function StructuredMesh{RealT}(size, coordinates_min, coordinates_max) where {RealT<:Real}
    NDIMS = length(size)

    return StructuredMesh{RealT, NDIMS}(size, coordinates_min, coordinates_max)
end

@inline Base.ndims(mesh::StructuredMesh{RealT, NDIMS}) where {RealT, NDIMS} = NDIMS