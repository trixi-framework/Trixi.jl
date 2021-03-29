"""
    CurvedMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

A structured curved mesh.

Different numbers of cells per dimension are possible and arbitrary functions 
can be used as domain faces.

Note: This is an experimental feature and may be changed in future releases without notice.
"""
mutable struct CurvedMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}
  cells_per_dimension::NTuple{NDIMS, Int}
  faces::Vector{Function}
  current_filename::String
  unsaved_changes::Bool
end


"""
    CurvedMesh(cells_per_dimension, faces, RealT)

Create a CurvedMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTUPLE{NDIMS, Int}`: the number of cells in each dimension.
- `faces::Vector{Function}`: a vector of `2 * NDIMS` functions that describe the faces of the domain.
                             Each function must take `NDIMS-1` arguments.
                             `faces[1]` describes the face onto which the face in negative x-direction 
                             of the unit hypercube is mapped. The face in positive x-direction of
                             the unit hypercube will be mapped onto the face described by `faces[2]`.
                             `faces[3:4]` describe the faces in positive and negative y-direction respectively 
                             (in 2D and 3D).
                             `faces[5:6]` describe the faces in positive and negative z-direction respectively
                             (in 3D).
- `RealT::Type`: The type that should be used for coordinates.
"""
function CurvedMesh(cells_per_dimension, faces, RealT; unsaved_changes=true)
  NDIMS = length(cells_per_dimension)

  return CurvedMesh{NDIMS, RealT}(cells_per_dimension, faces, "", unsaved_changes)
end


function CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max, RealT)
  NDIMS = length(cells_per_dimension)
  faces = coordinates2faces(coordinates_min, coordinates_max)

  return CurvedMesh{NDIMS, RealT}(cells_per_dimension, faces, "", true)
end

function coordinates2faces(coordinates_min::NTuple{1}, coordinates_max::NTuple{1})
  f1() = [ coordinates_min[1] ]
  f2() = [ coordinates_max[1] ]

  return [f1, f2]
end


# In 1D
function bilinear_mapping(x, mesh)
  return 0.5 * ((1 - x) * mesh.faces[1]() +
                (1 + x) * mesh.faces[2]())
end


# In 2D
function bilinear_mapping(x, y, mesh)
  @unpack faces = mesh

  x1 = faces[1](-1) # Bottom left
  @assert x1 ≈ faces[3](-1) "faces[1](-1) needs to match faces[3](-1) (bottom left corner)"
  x2 = faces[2](-1) # Bottom right
  @assert x2 ≈ faces[3](1) "faces[2](-1) needs to match faces[3](1) (bottom right corner)"
  x3 = faces[1](1) # Top left
  @assert x3 ≈ faces[4](-1) "faces[1](1) needs to match faces[4](-1) (top left corner)"
  x4 = faces[2](1) # Top right
  @assert x4 ≈ faces[4](1) "faces[2](1) needs to match faces[4](1) (top right corner)"

  return 0.25 * (x1 * (1 - x) * (1 - y) +
                 x2 * (1 + x) * (1 - y) +
                 x3 * (1 - x) * (1 + y) +
                 x4 * (1 + x) * (1 + y))
end


function transfinite_mapping(x, y, mesh)
  @unpack faces = mesh

  linear_interpolation_x(x, y) = 0.5 * (faces[1](y) * (1 - x) + faces[2](y) * (1 + x))
  linear_interpolation_y(x, y) = 0.5 * (faces[3](x) * (1 - y) + faces[4](x) * (1 + y))

  return linear_interpolation_x(x, y) + linear_interpolation_y(x, y) - bilinear_mapping(x, y, mesh)
end


@inline Base.ndims(::CurvedMesh{NDIMS}) where {NDIMS} = NDIMS
Base.size(mesh::CurvedMesh) = mesh.cells_per_dimension
Base.size(mesh::CurvedMesh, i) = mesh.cells_per_dimension[i]


function Base.show(io::IO, ::CurvedMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "CurvedMesh{", NDIMS, ", ", RealT, "}")
end


function Base.show(io::IO, ::MIME"text/plain", mesh::CurvedMesh{NDIMS, RealT}) where {NDIMS, RealT}
  if get(io, :compact, false)
    show(io, mesh)
  else
    setup = [
            "size" => size(mesh),
            "faces" => mesh.faces
            ]
    summary_box(io, "CurvedMesh{" * string(NDIMS) * ", " * string(RealT) * "}", setup)
  end
end
