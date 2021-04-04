"""
    CurvedMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

A structured curved mesh.

Different numbers of cells per dimension are possible and arbitrary functions 
can be used as domain faces.

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
mutable struct CurvedMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}
  cells_per_dimension::NTuple{NDIMS, Int}
  faces::Any # Not relevant for performance
  faces_as_string::Vector{String}
  current_filename::String
  unsaved_changes::Bool
end


"""
    CurvedMesh(cells_per_dimension, faces, RealT; unsaved_changes=true, faces_as_string=faces2string(faces))

Create a CurvedMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTupleE{NDIMS, Int}`: the number of cells in each dimension.
- `faces::NTuple{2*NDIMS, Function}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                                      Each function must take `NDIMS-1` arguments.
                                      `faces[1]` describes the face onto which the face in negative x-direction 
                                      of the unit hypercube is mapped. The face in positive x-direction of
                                      the unit hypercube will be mapped onto the face described by `faces[2]`.
                                      `faces[3:4]` describe the faces in positive and negative y-direction respectively 
                                      (in 2D and 3D).
                                      `faces[5:6]` describe the faces in positive and negative z-direction respectively
                                      (in 3D).
- `RealT::Type`: the type that should be used for coordinates.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
- `faces_as_string::Vector{String}`: a vector which contains the string of the function definition of each face.
                                     If `CodeTracking` can't find the function definition, it can be passed directly here.
"""
function CurvedMesh(cells_per_dimension, faces; RealT=Float64, unsaved_changes=true, faces_as_string=faces2string(faces))
  NDIMS = length(cells_per_dimension)

  return CurvedMesh{NDIMS, RealT}(Tuple(cells_per_dimension), faces, faces_as_string, "", unsaved_changes)
end


"""
    CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

Create a CurvedMesh that represents a uncurved structured mesh with a rectangular domain.

# Arguments
- `cells_per_dimension::NTuple{NDIMS, Int}`: the number of cells in each dimension.
- `coordinates_min::NTuple{NDIMS, RealT}`: coordinate of the corner in the negative direction of each dimension.
- `coordinates_max::NTuple{NDIMS, RealT}`: coordinate of the corner in the positive direction of each dimension.
"""
function CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)
  NDIMS = length(cells_per_dimension)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))
  faces, faces_as_string = coordinates2faces(Tuple(coordinates_min), Tuple(coordinates_max))

  return CurvedMesh(cells_per_dimension, faces; RealT=RealT, faces_as_string=faces_as_string)
end


# Extract a string of the code that defines the face functions
function faces2string(faces)
  NDIMS = div(length(faces), 2)
  face2substring(face) = code_string(face, ntuple(_ -> Float64, NDIMS-1))

  return faces .|> face2substring .|> string |> collect
end


# Convert min and max coordinates of a rectangle to the face functions of the rectangle
function coordinates2faces(coordinates_min::NTuple{1}, coordinates_max::NTuple{1})
  f1() = SVector(coordinates_min[1])
  f2() = SVector(coordinates_max[1])
  
  # CodeTracking can't find the definition here due to the dispatching by dimensions
  f1_as_string = "f1() = SVector($(coordinates_min[1]))"
  f2_as_string = "f2() = SVector($(coordinates_max[1]))"

  return (f1, f2), [f1_as_string, f2_as_string]
end


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


function coordinates2faces(coordinates_min::NTuple{3}, coordinates_max::NTuple{3})
  f1(s, t) = SVector(coordinates_min[1],
                     linear_interpolate(s, coordinates_min[2], coordinates_max[2]),
                     linear_interpolate(t, coordinates_min[3], coordinates_max[3]))

  f2(s, t) = SVector(coordinates_max[1], 
                     linear_interpolate(s, coordinates_min[2], coordinates_max[2]),
                     linear_interpolate(t, coordinates_min[3], coordinates_max[3]))

  f3(s, t) = SVector(linear_interpolate(s, coordinates_min[1], coordinates_max[1]), 
                     coordinates_min[2],
                     linear_interpolate(t, coordinates_min[3], coordinates_max[3]))

  f4(s, t) = SVector(linear_interpolate(s, coordinates_min[1], coordinates_max[1]),
                     coordinates_max[2],
                     linear_interpolate(t, coordinates_min[3], coordinates_max[3]))
  
  f5(s, t) = SVector(linear_interpolate(s, coordinates_min[1], coordinates_max[1]), 
                     linear_interpolate(t, coordinates_min[2], coordinates_max[2]),
                     coordinates_min[3])

  f6(s, t) = SVector(linear_interpolate(s, coordinates_min[1], coordinates_max[1]), 
                     linear_interpolate(t, coordinates_min[2], coordinates_max[2]),
                     coordinates_max[3])               
  
  # CodeTracking can't find the definition here due to the dispatching by dimensions
  f1_as_string = "f1(s, t) = SVector($(coordinates_min[1]),
                                     linear_interpolate(s, $(coordinates_min[2]), $(coordinates_max[2])),
                                     linear_interpolate(t, $(coordinates_min[3]), $(coordinates_max[3])))"

  f2_as_string = "f2(s, t) = SVector($(coordinates_max[1]),
                                     linear_interpolate(s, $(coordinates_min[2]), $(coordinates_max[2])),
                                     linear_interpolate(t, $(coordinates_min[3]), $(coordinates_max[3])))"

  f3_as_string = "f3(s, t) = SVector(linear_interpolate(s, $(coordinates_min[1]), $(coordinates_max[1])),
                                     $(coordinates_min[2]),
                                     linear_interpolate(t, $(coordinates_min[3]), $(coordinates_max[3])))"

  f4_as_string = "f4(s, t) = SVector(linear_interpolate(s, $(coordinates_min[1]), $(coordinates_max[1])),
                                     $(coordinates_max[2]),
                                     linear_interpolate(t, $(coordinates_min[3]), $(coordinates_max[3])))"

  f5_as_string = "f5(s, t) = SVector(linear_interpolate(s, $(coordinates_min[1]), $(coordinates_max[1])),
                                     linear_interpolate(t, $(coordinates_min[2]), $(coordinates_max[2])),
                                     $(coordinates_min[3]))"  

  f6_as_string = "f6(s, t) = SVector(linear_interpolate(s, $(coordinates_min[1]), $(coordinates_max[1])),
                                     linear_interpolate(t, $(coordinates_min[2]), $(coordinates_max[2])),
                                     $(coordinates_max[3]))"

  return (f1, f2, f3, f4, f5, f6), 
         [f1_as_string, f2_as_string, f3_as_string, f4_as_string, f5_as_string, f6_as_string]
end


# Interpolate linearly between left and right value where s should be between -1 and 1
linear_interpolate(s, left_value, right_value) = 0.5 * ((1 - s) * left_value + (1 + s) * right_value)


# In 1D
# Linear mapping from the reference element to the domain described by the faces
function linear_mapping(x, mesh)
  return linear_interpolate(x, mesh.faces[1](), mesh.faces[2]())
end


# In 2D
# Bilinear mapping from the reference element to the domain described by the faces
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


# In 3D
# Trilinear mapping from the reference element to the domain described by the faces
function trilinear_mapping(x, y, z, mesh)
  @unpack faces = mesh

  x1 = faces[1](-1, -1) # -x,-y,-z
  @assert x1 ≈ faces[3](-1, -1) ≈ faces[5](-1, -1) "faces[1](-1, -1), faces[3](-1, -1) and faces[5](-1, -1) need to match (-x, -y, -z corner)"

  x2 = faces[2](-1, -1) #  x,-y,-z
  @assert x2 ≈ faces[3]( 1, -1) ≈ faces[5]( 1, -1) "faces[2](-1, -1), faces[3]( 1, -1) and faces[5]( 1, -1) need to match ( x, -y, -z corner)"
  
  x3 = faces[1]( 1, -1) # -x, y,-z
  @assert x3 ≈ faces[4](-1, -1) ≈ faces[5](-1,  1) "faces[1]( 1, -1), faces[4](-1, -1) and faces[5](-1,  1) need to match (-x,  y, -z corner)"
  
  x4 = faces[2]( 1, -1) #  x, y,-z
  @assert x4 ≈ faces[4]( 1, -1) ≈ faces[5]( 1,  1) "faces[2]( 1, -1), faces[4]( 1, -1) and faces[5]( 1,  1) need to match ( x,  y, -z corner)"
  
  x5 = faces[1](-1,  1) # -x,-y, z
  @assert x5 ≈ faces[3](-1,  1) ≈ faces[6](-1, -1) "faces[1](-1,  1), faces[3](-1,  1) and faces[6](-1, -1) need to match (-x, -y,  z corner)"

  x6 = faces[2](-1,  1) #  x,-y, z
  @assert x6 ≈ faces[3]( 1,  1) ≈ faces[6]( 1, -1) "faces[2](-1,  1), faces[3]( 1,  1) and faces[6]( 1, -1) need to match ( x, -y,  z corner)"
  
  x7 = faces[1]( 1,  1) # -x, y, z
  @assert x7 ≈ faces[4](-1,  1) ≈ faces[6](-1,  1) "faces[1]( 1,  1), faces[4](-1,  1) and faces[6](-1,  1) need to match (-x,  y,  z corner)"
  
  x8 = faces[2]( 1,  1) #  x, y, z
  @assert x8 ≈ faces[4]( 1,  1) ≈ faces[6]( 1,  1) "faces[2]( 1,  1), faces[4]( 1,  1) and faces[6]( 1,  1) need to match ( x,  y,  z corner)"

  return 0.125 * (x1 * (1 - x) * (1 - y) * (1 - z) +
                  x2 * (1 + x) * (1 - y) * (1 - z) +
                  x3 * (1 - x) * (1 + y) * (1 - z) +
                  x4 * (1 + x) * (1 + y) * (1 - z) +
                  x5 * (1 - x) * (1 - y) * (1 + z) +
                  x6 * (1 + x) * (1 - y) * (1 + z) +
                  x7 * (1 - x) * (1 + y) * (1 + z) +
                  x8 * (1 + x) * (1 + y) * (1 + z) )
end


# In 2D
# Transfinite mapping from the reference element to the domain described by the faces
function transfinite_mapping(x, y, mesh)
  @unpack faces = mesh

  return linear_interpolate(x, faces[1](y), faces[2](y)) + 
         linear_interpolate(y, faces[3](x), faces[4](x)) - 
         bilinear_mapping(x, y, mesh)
end


@inline Base.ndims(::CurvedMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::CurvedMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT
Base.size(mesh::CurvedMesh) = mesh.cells_per_dimension
Base.size(mesh::CurvedMesh, i) = mesh.cells_per_dimension[i]


function Base.show(io::IO, ::CurvedMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "CurvedMesh{", NDIMS, ", ", RealT, "}")
end


function Base.show(io::IO, ::MIME"text/plain", mesh::CurvedMesh{NDIMS, RealT}) where {NDIMS, RealT}
  if get(io, :compact, false)
    show(io, mesh)
  else
    summary_header(io, "CurvedMesh{" * string(NDIMS) * ", " * string(RealT) * "}")
    summary_line(io, "size", size(mesh))
    summary_line(io, "faces", 2*NDIMS)
    summary_line(increment_indent(io), "negative x", mesh.faces_as_string[1])
    summary_line(increment_indent(io), "positive x", mesh.faces_as_string[2])
    if NDIMS > 1
      summary_line(increment_indent(io), "negative y", mesh.faces_as_string[3])
      summary_line(increment_indent(io), "positive y", mesh.faces_as_string[4])
    end
    if NDIMS > 2
      summary_line(increment_indent(io), "negative z", mesh.faces_as_string[5])
      summary_line(increment_indent(io), "positive z", mesh.faces_as_string[6])
    end
    summary_footer(io)
  end
end
