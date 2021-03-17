struct StructuredMesh{NDIMS, RealT<:Real}
  size::Tuple{Vararg{Int64, NDIMS}}
  coordinates_min::Tuple{Vararg{RealT, NDIMS}}
  coordinates_max::Tuple{Vararg{RealT, NDIMS}}
  linear_indices::LinearIndices{NDIMS, Tuple{Vararg{UnitRange{Int64}, NDIMS}}}
end

function StructuredMesh(size, coordinates_min, coordinates_max)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  NDIMS = length(size)

  return StructuredMesh{NDIMS, RealT}(size, coordinates_min, coordinates_max, LinearIndices(size))
end


@inline Base.ndims(mesh::StructuredMesh{NDIMS}) where {NDIMS} = NDIMS
