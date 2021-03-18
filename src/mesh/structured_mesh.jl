mutable struct StructuredMesh{NDIMS, RealT<:Real}
  size::NTuple{NDIMS, Int}
  coordinates_min::NTuple{NDIMS, RealT}
  coordinates_max::NTuple{NDIMS, RealT}
  linear_indices::LinearIndices{NDIMS, NTuple{NDIMS, UnitRange{Int}}}
  current_filename::String
  unsaved_changes::Bool
end

function StructuredMesh(size, coordinates_min, coordinates_max)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  NDIMS = length(size)

  return StructuredMesh{NDIMS, RealT}(size, coordinates_min, coordinates_max, LinearIndices(size), "", true)
end


@inline Base.ndims(::StructuredMesh{NDIMS}) where {NDIMS} = NDIMS


function Base.show(io::IO, ::StructuredMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "StructuredMesh{", NDIMS, ", ", RealT, "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::StructuredMesh{NDIMS, RealT}) where {NDIMS, RealT}
  if get(io, :compact, false)
    show(io, mesh)
  else
    setup = [
            "coordinates_min" => mesh.coordinates_min,
            "coordinates_max" => mesh.coordinates_max,
            "size" => mesh.size
            ]
    summary_box(io, "StructuredMesh{" * string(NDIMS) * ", " * string(RealT) * "}", setup)
  end
end
