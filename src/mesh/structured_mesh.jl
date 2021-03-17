struct StructuredMesh{NDIMS, RealT<:Real}
  size::NTuple{NDIMS, Int}
  coordinates_min::NTuple{NDIMS, Int}
  coordinates_max::NTuple{NDIMS, Int}
  linear_indices::LinearIndices{NDIMS, NTuple{NDIMS, UnitRange{Int}}}
end

function StructuredMesh(size, coordinates_min, coordinates_max)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  NDIMS = length(size)

  return StructuredMesh{NDIMS, RealT}(size, coordinates_min, coordinates_max, LinearIndices(size))
end


@inline Base.ndims(mesh::StructuredMesh{NDIMS}) where {NDIMS} = NDIMS
