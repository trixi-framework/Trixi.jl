struct StructuredMesh{RealT<:Real, NDIMS}
  size::Tuple{Vararg{Int64, NDIMS}}
  coordinates_min::Tuple{Vararg{RealT, NDIMS}}
  coordinates_max::Tuple{Vararg{RealT, NDIMS}}
  linear_indices::LinearIndices{NDIMS, Tuple{Vararg{UnitRange{Int64}, NDIMS}}}
end

function StructuredMesh{RealT}(size, coordinates_min, coordinates_max) where {RealT<:Real}
  NDIMS = length(size)

  return StructuredMesh{RealT, NDIMS}(size, coordinates_min, coordinates_max, LinearIndices(size))
end


@inline Base.ndims(mesh::StructuredMesh{RealT, NDIMS}) where {RealT, NDIMS} = NDIMS