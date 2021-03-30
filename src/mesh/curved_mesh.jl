"""
    CurvedMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

A structured curved mesh.

Different numbers of cells per dimension are possible and arbitrary functions 
can be used as domain faces.

Note: This is an experimental feature and may be changed in future releases without notice.
"""
mutable struct CurvedMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}
  cells_per_dimension::NTuple{NDIMS, Int}
  faces::Tuple
  faces_as_string::Vector{String}
  current_filename::String
  unsaved_changes::Bool
end


"""
    CurvedMesh(cells_per_dimension, faces, RealT)

Create a CurvedMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTUPLE{NDIMS, Int}`: the number of cells in each dimension.
- `faces::NTuple{2*NDIMS,Function}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                                     Each function must take `NDIMS-1` arguments.
                                     `faces[1]` describes the face onto which the face in negative x-direction 
                                     of the unit hypercube is mapped. The face in positive x-direction of
                                     the unit hypercube will be mapped onto the face described by `faces[2]`.
                                     `faces[3:4]` describe the faces in positive and negative y-direction respectively 
                                     (in 2D and 3D).
                                     `faces[5:6]` describe the faces in positive and negative z-direction respectively
                                     (in 3D).
- `RealT::Type`: The type that should be used for coordinates.
- `faces_as_string::Vector{String}`: a vector which contains the string of the function definition of each face.
                                     If `CodeTracking` can't find the function definition, it can be passed directly here.
"""
function CurvedMesh(cells_per_dimension, faces, RealT::Type; unsaved_changes=true, faces_as_string=faces2string(faces))
  NDIMS = length(cells_per_dimension)

  return CurvedMesh{NDIMS, RealT}(cells_per_dimension, faces, faces_as_string, "", unsaved_changes)
end


function CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
  NDIMS = length(cells_per_dimension)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  faces, faces_as_string = coordinates2faces(coordinates_min, coordinates_max)

  return CurvedMesh(cells_per_dimension, faces, RealT; faces_as_string=faces_as_string)
end


function faces2string(faces)
  NDIMS = div(length(faces), 2)
  face2substring(face) = code_string(face, ntuple(_ -> Float64, NDIMS-1))

  return faces .|> face2substring .|> string |> collect
end


function coordinates2faces(coordinates_min::NTuple{1}, coordinates_max::NTuple{1})
  f1() = SVector(coordinates_min[1])
  f2() = SVector(coordinates_max[1])
  
  # CodeTracking can't find the definition here due to the dispatching by dimensions
  f1_as_string = "f1() = SVector($(coordinates_min[1]))"
  f2_as_string = "f2() = SVector($(coordinates_max[1]))"

  return (f1, f2), [f1_as_string, f2_as_string]
end

# This needs to be accessible outside of the function below when loading the mesh from a file
linear_interpolate(s, left_value, right_value) = 0.5 * ((1 - s) * left_value + (1 + s) * right_value)

function coordinates2faces(coordinates_min::NTuple{2}, coordinates_max::NTuple{2})
  f1(s) = SVector(coordinates_min[1], linear_interpolate(s, coordinates_min[2], coordinates_max[2]))
  f2(s) = SVector(coordinates_max[1], linear_interpolate(s, coordinates_min[2], coordinates_max[2]))
  f3(s) = SVector(linear_interpolate(s, coordinates_min[1], coordinates_max[1]), coordinates_min[2])
  f4(s) = SVector(linear_interpolate(s, coordinates_min[1], coordinates_max[1]), coordinates_max[2])
  
  # CodeTracking can't find the definition here due to the dispatching by dimensions
  f1_as_string = "f1(s) = SVector($(coordinates_min[1]), linear_interpolate(s, $(coordinates_min[2]), $(coordinates_max[2])))"
  f2_as_string = "f2(s) = SVector($(coordinates_max[1]), linear_interpolate(s, $(coordinates_min[2]), $(coordinates_max[2])))"
  f3_as_string = "f3(s) = SVector(linear_interpolate(s, $(coordinates_min[1]), $(coordinates_max[1])), $(coordinates_min[2]))"
  f4_as_string = "f4(s) = SVector(linear_interpolate(s, $(coordinates_min[1]), $(coordinates_max[1])), $(coordinates_max[2]))"

  return (f1, f2, f3, f4), [f1_as_string, f2_as_string, f3_as_string, f4_as_string]
end


# In 1D
function linear_mapping(x, mesh)
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
