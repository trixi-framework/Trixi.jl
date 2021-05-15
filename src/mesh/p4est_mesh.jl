"""
    P4estMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}

A structured curved mesh.

Different numbers of cells per dimension are possible and arbitrary functions
can be used as domain faces.

!!! warning "Experimental code"
    This mesh type is experimental and can change any time.
"""
mutable struct P4estMesh{NDIMS, RealT<:Real} <: AbstractMesh{NDIMS}
  cells_per_dimension::NTuple{NDIMS, Int}
  mapping::Any # Not relevant for performance
  mapping_as_string::String
  periodicity::NTuple{NDIMS, Bool}
  current_filename::String
  unsaved_changes::Bool
end


"""
    P4estMesh(cells_per_dimension, mapping, RealT; unsaved_changes=true, mapping_as_string=mapping2string(mapping, length(cells_per_dimension)))

Create a P4estMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTupleE{NDIMS, Int}`: the number of cells in each dimension.
- `mapping`: a function of `NDIMS` variables to describe the mapping, which transforms
             the reference mesh to the physical domain.
             If no `mapping_as_string` is defined, this function must be defined with the name `mapping`
             to allow for restarts.
             This will be changed in the future, see https://github.com/trixi-framework/Trixi.jl/issues/541.
- `RealT::Type`: the type that should be used for coordinates.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
- `mapping_as_string::String`: the code that defines the `mapping`.
                               If `CodeTracking` can't find the function definition, it can be passed directly here.
                               The code string must define the mapping function with the name `mapping`.
                               This will be changed in the future, see https://github.com/trixi-framework/Trixi.jl/issues/541.
"""
function P4estMesh(cells_per_dimension, mapping; RealT=Float64, periodicity=true, unsaved_changes=true,
                    mapping_as_string=mapping2string(mapping, length(cells_per_dimension)))
  NDIMS = length(cells_per_dimension)

  # Convert periodicity to a Tuple of a Bool for every dimension
  if all(periodicity)
    # Also catches case where periodicity = true
    periodicity = ntuple(_->true, NDIMS)
  elseif !any(periodicity)
    # Also catches case where periodicity = false
    periodicity = ntuple(_->false, NDIMS)
  else
    # Default case if periodicity is an iterable
    periodicity = Tuple(periodicity)
  end

  return P4estMesh{NDIMS, RealT}(Tuple(cells_per_dimension), mapping, mapping_as_string, periodicity, "", unsaved_changes)
end


"""
    P4estMesh(cells_per_dimension, faces, RealT; unsaved_changes=true, faces_as_string=faces2string(faces))

Create a P4estMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTupleE{NDIMS, Int}`: the number of cells in each dimension.
- `faces::NTuple{2*NDIMS}`: a tuple of `2 * NDIMS` functions that describe the faces of the domain.
                            Each function must take `NDIMS-1` arguments.
                            `faces[1]` describes the face onto which the face in negative x-direction
                            of the unit hypercube is mapped. The face in positive x-direction of
                            the unit hypercube will be mapped onto the face described by `faces[2]`.
                            `faces[3:4]` describe the faces in positive and negative y-direction respectively
                            (in 2D and 3D).
                            `faces[5:6]` describe the faces in positive and negative z-direction respectively (in 3D).
- `RealT::Type`: the type that should be used for coordinates.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function P4estMesh(cells_per_dimension, faces::Tuple; RealT=Float64, periodicity=true)
  NDIMS = length(cells_per_dimension)

  validate_faces(faces)

  # Use the transfinite mapping with the correct number of arguments
  mapping = transfinite_mapping(faces)

  # Collect definitions of face functions in one string (separated by semicolons)
  face2substring(face) = code_string(face, ntuple(_ -> Float64, NDIMS-1))
  join_semicolon(strings) = join(strings, "; ")

  faces_definition = faces .|> face2substring .|> string |> join_semicolon

  # Include faces definition in `mapping_as_string` to allow for evaluation
  # without knowing the face functions
  mapping_as_string = "$faces_definition; faces = $(string(faces)); mapping = transfinite_mapping(faces)"

  return P4estMesh(cells_per_dimension, mapping; RealT=RealT, periodicity=periodicity, mapping_as_string=mapping_as_string)
end


"""
    P4estMesh(cells_per_dimension, coordinates_min, coordinates_max)

Create a P4estMesh that represents a uncurved structured mesh with a rectangular domain.

# Arguments
- `cells_per_dimension::NTuple{NDIMS, Int}`: the number of cells in each dimension.
- `coordinates_min::NTuple{NDIMS, RealT}`: coordinate of the corner in the negative direction of each dimension.
- `coordinates_max::NTuple{NDIMS, RealT}`: coordinate of the corner in the positive direction of each dimension.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function P4estMesh(cells_per_dimension, coordinates_min, coordinates_max; periodicity=true)
  NDIMS = length(cells_per_dimension)
  RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))

  mapping = coordinates2mapping(coordinates_min, coordinates_max)
  mapping_as_string = "coordinates_min = $coordinates_min; " *
                      "coordinates_max = $coordinates_max; " *
                      "mapping = coordinates2mapping(coordinates_min, coordinates_max)"
  return P4estMesh(cells_per_dimension, mapping; RealT=RealT, periodicity=periodicity, mapping_as_string=mapping_as_string)
end


# Check if mesh is periodic
isperiodic(mesh::P4estMesh) = all(mesh.periodicity)
isperiodic(mesh::P4estMesh, dimension) = mesh.periodicity[dimension]

@inline Base.ndims(::P4estMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT
Base.size(mesh::P4estMesh) = mesh.cells_per_dimension
Base.size(mesh::P4estMesh, i) = mesh.cells_per_dimension[i]
Base.axes(mesh::P4estMesh) = map(Base.OneTo, mesh.cells_per_dimension)
Base.axes(mesh::P4estMesh, i) = Base.OneTo(mesh.cells_per_dimension[i])


function Base.show(io::IO, ::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
  print(io, "P4estMesh{", NDIMS, ", ", RealT, "}")
end


function Base.show(io::IO, ::MIME"text/plain", mesh::P4estMesh{NDIMS, RealT}) where {NDIMS, RealT}
  if get(io, :compact, false)
    show(io, mesh)
  else
    summary_header(io, "P4estMesh{" * string(NDIMS) * ", " * string(RealT) * "}")
    summary_line(io, "size", size(mesh))

    summary_line(io, "mapping", "")
    # Print code lines of mapping_as_string
    mapping_lines = split(mesh.mapping_as_string, ";")
    for i in eachindex(mapping_lines)
      summary_line(increment_indent(io), "line $i", strip(mapping_lines[i]))
    end
    summary_footer(io)
  end
end
