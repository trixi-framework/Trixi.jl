struct StructuredMesh{RealT<:Real, NDIMS}
  size::NTuple{NDIMS, Int}
  coordinates_min::NTuple{NDIMS, Int}
  coordinates_max::NTuple{NDIMS, Int}
  linear_indices::LinearIndices{NDIMS, NTuple{NDIMS, UnitRange{Int}}}
end

function StructuredMesh{RealT}(size, coordinates_min, coordinates_max) where {RealT<:Real}
  NDIMS = length(size)

  return StructuredMesh{RealT, NDIMS}(size, coordinates_min, coordinates_max, LinearIndices(size))
end


@inline Base.ndims(mesh::StructuredMesh{RealT, NDIMS}) where {RealT, NDIMS} = NDIMS
