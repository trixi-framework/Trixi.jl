# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StructuredMesh{NDIMS} <: AbstractMesh{NDIMS}

A structured curved mesh.

Different numbers of cells per dimension are possible and arbitrary functions
can be used as domain faces.
"""
mutable struct StructuredMesh{NDIMS, RealT <: Real} <: AbstractMesh{NDIMS}
    cells_per_dimension::NTuple{NDIMS, Int}
    mapping::Any # Not relevant for performance
    mapping_as_string::String
    periodicity::NTuple{NDIMS, Bool}
    current_filename::String
    unsaved_changes::Bool
end

"""
    StructuredMesh(cells_per_dimension, mapping; RealT=Float64, unsaved_changes=true, mapping_as_string=mapping2string(mapping, length(cells_per_dimension)))

Create a StructuredMesh of the given size and shape that uses `RealT` as coordinate type.

# Arguments
- `cells_per_dimension::NTupleE{NDIMS, Int}`: the number of cells in each dimension.
- `mapping`: a function of `NDIMS` variables to describe the mapping, which transforms
             the reference mesh to the physical domain.
             If no `mapping_as_string` is defined, this function must be defined with the name `mapping`
             to allow for restarts.
             This will be changed in the future, see [https://github.com/trixi-framework/Trixi.jl/issues/541](https://github.com/trixi-framework/Trixi.jl/issues/541).
- `RealT::Type`: the type that should be used for coordinates.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}`
                 deciding for each dimension if the boundaries in this dimension are periodic.
- `unsaved_changes::Bool`: if set to `true`, the mesh will be saved to a mesh file.
- `mapping_as_string::String`: the code that defines the `mapping`.
                               If `CodeTracking` can't find the function definition, it can be passed directly here.
                               The code string must define the mapping function with the name `mapping`.
                               This will be changed in the future, see [https://github.com/trixi-framework/Trixi.jl/issues/541](https://github.com/trixi-framework/Trixi.jl/issues/541).
"""
function StructuredMesh(cells_per_dimension, mapping; RealT = Float64,
                        periodicity = true, unsaved_changes = true,
                        mapping_as_string = mapping2string(mapping,
                                                           length(cells_per_dimension)))
    NDIMS = length(cells_per_dimension)

    # Convert periodicity to a Tuple of a Bool for every dimension
    if all(periodicity)
        # Also catches case where periodicity = true
        periodicity = ntuple(_ -> true, NDIMS)
    elseif !any(periodicity)
        # Also catches case where periodicity = false
        periodicity = ntuple(_ -> false, NDIMS)
    else
        # Default case if periodicity is an iterable
        periodicity = Tuple(periodicity)
    end

    return StructuredMesh{NDIMS, RealT}(Tuple(cells_per_dimension), mapping,
                                        mapping_as_string, periodicity, "",
                                        unsaved_changes)
end

"""
    StructuredMesh(cells_per_dimension, faces; RealT=Float64, unsaved_changes=true, faces_as_string=faces2string(faces))

Create a StructuredMesh of the given size and shape that uses `RealT` as coordinate type.

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
function StructuredMesh(cells_per_dimension, faces::Tuple; RealT = Float64,
                        periodicity = true)
    NDIMS = length(cells_per_dimension)

    validate_faces(faces)

    # Use the transfinite mapping with the correct number of arguments
    mapping = transfinite_mapping(faces)

    # Collect definitions of face functions in one string (separated by semicolons)
    face2substring(face) = code_string(face, ntuple(_ -> Float64, NDIMS - 1))
    join_newline(strings) = join(strings, "\n")

    faces_definition = faces .|> face2substring .|> string |> join_newline

    # Include faces definition in `mapping_as_string` to allow for evaluation
    # without knowing the face functions
    mapping_as_string = """
        $faces_definition
        faces = $(string(faces))
        mapping = transfinite_mapping(faces)
        """

    return StructuredMesh(cells_per_dimension, mapping; RealT = RealT,
                          periodicity = periodicity,
                          mapping_as_string = mapping_as_string)
end

"""
    StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max; periodicity=true)

Create a StructuredMesh that represents a uncurved structured mesh with a rectangular domain.

# Arguments
- `cells_per_dimension::NTuple{NDIMS, Int}`: the number of cells in each dimension.
- `coordinates_min::NTuple{NDIMS, RealT}`: coordinate of the corner in the negative direction of each dimension.
- `coordinates_max::NTuple{NDIMS, RealT}`: coordinate of the corner in the positive direction of each dimension.
- `periodicity`: either a `Bool` deciding if all of the boundaries are periodic or an `NTuple{NDIMS, Bool}` deciding for
                 each dimension if the boundaries in this dimension are periodic.
"""
function StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max;
                        periodicity = true)
    RealT = promote_type(eltype(coordinates_min), eltype(coordinates_max))

    mapping = coordinates2mapping(coordinates_min, coordinates_max)
    mapping_as_string = """
        coordinates_min = $coordinates_min
        coordinates_max = $coordinates_max
        mapping = coordinates2mapping(coordinates_min, coordinates_max)
        """
    return StructuredMesh(cells_per_dimension, mapping; RealT = RealT,
                          periodicity = periodicity,
                          mapping_as_string = mapping_as_string)
end

# Extract a string of the code that defines the mapping function
function mapping2string(mapping, ndims)
    string(code_string(mapping, ntuple(_ -> Float64, ndims)))
end

# An internal function wrapping `CodeTracking.code_string` with additional
# error checking to avoid some problems when calling this function in
# Jupyter notebooks or Documenter.jl environments. See
# - https://github.com/trixi-framework/Trixi.jl/issues/931
# - https://github.com/trixi-framework/Trixi.jl/pull/1084
function code_string(f, t)
    try
        return CodeTracking.code_string(f, t)
    catch e
        return ""
    end
end

# Interpolate linearly between left and right value where s should be between -1 and 1
function linear_interpolate(s, left_value, right_value)
    0.5f0 * ((1 - s) * left_value + (1 + s) * right_value)
end

# Convert min and max coordinates of a rectangle to the corresponding transformation mapping
function coordinates2mapping(coordinates_min::NTuple{1}, coordinates_max::NTuple{1})
    mapping(xi) = linear_interpolate(xi, coordinates_min[1], coordinates_max[1])
end

function coordinates2mapping(coordinates_min::NTuple{2}, coordinates_max::NTuple{2})
    function mapping(xi, eta)
        SVector(linear_interpolate(xi, coordinates_min[1], coordinates_max[1]),
                linear_interpolate(eta, coordinates_min[2], coordinates_max[2]))
    end
end

function coordinates2mapping(coordinates_min::NTuple{3}, coordinates_max::NTuple{3})
    function mapping(xi, eta, zeta)
        SVector(linear_interpolate(xi, coordinates_min[1], coordinates_max[1]),
                linear_interpolate(eta, coordinates_min[2], coordinates_max[2]),
                linear_interpolate(zeta, coordinates_min[3], coordinates_max[3]))
    end
end

# In 1D
# Linear mapping from the reference element to the domain described by the faces
function linear_mapping(x, faces)
    return linear_interpolate(x, faces[1](), faces[2]())
end

# In 2D
# Bilinear mapping from the reference element to the domain described by the faces
function bilinear_mapping(x, y, faces)
    x1 = faces[1](-1) # Bottom left
    x2 = faces[2](-1) # Bottom right
    x3 = faces[1](1) # Top left
    x4 = faces[2](1) # Top right

    return 0.25f0 * (x1 * (1 - x) * (1 - y) +
            x2 * (1 + x) * (1 - y) +
            x3 * (1 - x) * (1 + y) +
            x4 * (1 + x) * (1 + y))
end

# In 3D
# Trilinear mapping from the reference element to the domain described by the faces
function trilinear_mapping(x, y, z, faces)
    x1 = faces[1](-1, -1) # mapped from (-1,-1,-1)
    x2 = faces[2](-1, -1) # mapped from ( 1,-1,-1)
    x3 = faces[1](1, -1) # mapped from (-1, 1,-1)
    x4 = faces[2](1, -1) # mapped from ( 1, 1,-1)
    x5 = faces[1](-1, 1) # mapped from (-1,-1, 1)
    x6 = faces[2](-1, 1) # mapped from ( 1,-1, 1)
    x7 = faces[1](1, 1) # mapped from (-1, 1, 1)
    x8 = faces[2](1, 1) # mapped from ( 1, 1, 1)

    return 0.125f0 * (x1 * (1 - x) * (1 - y) * (1 - z) +
            x2 * (1 + x) * (1 - y) * (1 - z) +
            x3 * (1 - x) * (1 + y) * (1 - z) +
            x4 * (1 + x) * (1 + y) * (1 - z) +
            x5 * (1 - x) * (1 - y) * (1 + z) +
            x6 * (1 + x) * (1 - y) * (1 + z) +
            x7 * (1 - x) * (1 + y) * (1 + z) +
            x8 * (1 + x) * (1 + y) * (1 + z))
end

# Use linear mapping in 1D
transfinite_mapping(faces::NTuple{2, Any}) = x -> linear_mapping(x, faces)

# In 2D
# Transfinite mapping from the reference element to the domain described by the faces
function transfinite_mapping(faces::NTuple{4, Any})
    function mapping(x, y)
        (linear_interpolate(x, faces[1](y), faces[2](y)) +
         linear_interpolate(y, faces[3](x), faces[4](x)) -
         bilinear_mapping(x, y, faces))
    end
end

# In 3D
# Correction term for the Transfinite mapping
function correction_term_3d(x, y, z, faces)
    # Correction for x-terms
    c_x = linear_interpolate(x,
                             linear_interpolate(y, faces[3](-1, z), faces[4](-1, z)) +
                             linear_interpolate(z, faces[5](-1, y), faces[6](-1, y)),
                             linear_interpolate(y, faces[3](1, z), faces[4](1, z)) +
                             linear_interpolate(z, faces[5](1, y), faces[6](1, y)))

    # Correction for y-terms
    c_y = linear_interpolate(y,
                             linear_interpolate(x, faces[1](-1, z), faces[2](-1, z)) +
                             linear_interpolate(z, faces[5](x, -1), faces[6](x, -1)),
                             linear_interpolate(x, faces[1](1, z), faces[2](1, z)) +
                             linear_interpolate(z, faces[5](x, 1), faces[6](x, 1)))

    # Correction for z-terms
    c_z = linear_interpolate(z,
                             linear_interpolate(x, faces[1](y, -1), faces[2](y, -1)) +
                             linear_interpolate(y, faces[3](x, -1), faces[4](x, -1)),
                             linear_interpolate(x, faces[1](y, 1), faces[2](y, 1)) +
                             linear_interpolate(y, faces[3](x, 1), faces[4](x, 1)))

    # Each of the 12 edges are counted twice above
    # so we divide the correction term by two
    return 0.5f0 * (c_x + c_y + c_z)
end

# In 3D
# Transfinite mapping from the reference element to the domain described by the faces
function transfinite_mapping(faces::NTuple{6, Any})
    function mapping(x, y, z)
        (linear_interpolate(x, faces[1](y, z), faces[2](y, z)) +
         linear_interpolate(y, faces[3](x, z), faces[4](x, z)) +
         linear_interpolate(z, faces[5](x, y), faces[6](x, y)) -
         correction_term_3d(x, y, z, faces) +
         trilinear_mapping(x, y, z, faces))
    end
end

function validate_faces(faces::NTuple{2, Any}) end

function validate_faces(faces::NTuple{4, Any})
    @assert faces[1](-1)≈faces[3](-1) "faces[1](-1) needs to match faces[3](-1) (bottom left corner)"
    @assert faces[2](-1)≈faces[3](1) "faces[2](-1) needs to match faces[3](1) (bottom right corner)"
    @assert faces[1](1)≈faces[4](-1) "faces[1](1) needs to match faces[4](-1) (top left corner)"
    @assert faces[2](1)≈faces[4](1) "faces[2](1) needs to match faces[4](1) (top right corner)"
end

function validate_faces(faces::NTuple{6, Any})
    @assert (faces[1](-1, -1)≈
             faces[3](-1, -1)≈
             faces[5](-1, -1)) "faces[1](-1, -1), faces[3](-1, -1) and faces[5](-1, -1) need to match at (-1, -1, -1) corner"

    @assert (faces[2](-1, -1)≈
             faces[3](1, -1)≈
             faces[5](1, -1)) "faces[2](-1, -1), faces[3](1, -1) and faces[5](1, -1) need to match at (1, -1, -1) corner"

    @assert (faces[1](1, -1)≈
             faces[4](-1, -1)≈
             faces[5](-1, 1)) "faces[1](1, -1), faces[4](-1, -1) and faces[5](-1, 1) need to match at (-1, 1, -1) corner"

    @assert (faces[2](1, -1)≈
             faces[4](1, -1)≈
             faces[5](1, 1)) "faces[2](1, -1), faces[4](1, -1) and faces[5](1, 1) need to match at (1, 1, -1) corner"

    @assert (faces[1](-1, 1)≈
             faces[3](-1, 1)≈
             faces[6](-1, -1)) "faces[1](-1, 1), faces[3](-1, 1) and faces[6](-1, -1) need to match at (-1, -1, 1) corner"

    @assert (faces[2](-1, 1)≈
             faces[3](1, 1)≈
             faces[6](1, -1)) "faces[2](-1, 1), faces[3](1, 1) and faces[6](1, -1) need to match at (1, -1, 1) corner"

    @assert (faces[1](1, 1)≈
             faces[4](-1, 1)≈
             faces[6](-1, 1)) "faces[1](1, 1), faces[4](-1, 1) and faces[6](-1, 1) need to match at (-1, 1, 1) corner"

    @assert (faces[2](1, 1)≈
             faces[4](1, 1)≈
             faces[6](1, 1)) "faces[2](1, 1), faces[4](1, 1) and faces[6](1, 1) need to match at (1, 1, 1) corner"
end

# Check if mesh is periodic
isperiodic(mesh::StructuredMesh) = all(mesh.periodicity)
isperiodic(mesh::StructuredMesh, dimension) = mesh.periodicity[dimension]

@inline Base.ndims(::StructuredMesh{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::StructuredMesh{NDIMS, RealT}) where {NDIMS, RealT} = RealT
Base.size(mesh::StructuredMesh) = mesh.cells_per_dimension
Base.size(mesh::StructuredMesh, i) = mesh.cells_per_dimension[i]
Base.axes(mesh::StructuredMesh) = map(Base.OneTo, mesh.cells_per_dimension)
Base.axes(mesh::StructuredMesh, i) = Base.OneTo(mesh.cells_per_dimension[i])

function Base.show(io::IO, mesh::StructuredMesh)
    print(io, "StructuredMesh{", ndims(mesh), ", ", real(mesh), "}")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::StructuredMesh)
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io,
                       "StructuredMesh{" * string(ndims(mesh)) * ", " *
                       string(real(mesh)) * "}")
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
end # @muladd
