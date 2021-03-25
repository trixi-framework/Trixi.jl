# Note: This is an experimental feature and may be changed in future releases without notice.
mutable struct StructuredMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}
  cells_per_dimension::NTuple{NDIMS, Int}
  coordinates_min::NTuple{NDIMS, RealT}
  coordinates_max::NTuple{NDIMS, RealT}
  current_filename::String
  unsaved_changes::Bool
end

function StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  NDIMS = length(cells_per_dimension)

  return StructuredMesh{NDIMS, RealT}(cells_per_dimension, coordinates_min, coordinates_max, "", true)
end


@inline Base.ndims(::StructuredMesh{NDIMS}) where {NDIMS} = NDIMS
Base.size(mesh::StructuredMesh) = mesh.cells_per_dimension
Base.size(mesh::StructuredMesh, i) = mesh.cells_per_dimension[i]

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
            "size" => size(mesh)
            ]
    summary_box(io, "StructuredMesh{" * string(NDIMS) * ", " * string(RealT) * "}", setup)
  end
end


function total_volume(mesh::StructuredMesh)
  return prod(mesh.coordinates_max .- mesh.coordinates_min)
end
