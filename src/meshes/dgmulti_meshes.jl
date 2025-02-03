# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    DGMultiMesh{NDIMS, ...}

`DGMultiMesh` describes a mesh type which wraps `StartUpDG.MeshData` and `boundary_faces` in a
dispatchable type. This is intended to store geometric data and connectivities for any type of
mesh (Cartesian, affine, curved, structured/unstructured).
"""
mutable struct DGMultiMesh{NDIMS, MeshType, MeshDataT <: MeshData{NDIMS}, RefElemDataT <: RefElemData, BoundaryFaceT} <: AbstractMesh{NDIMS}
    md::MeshDataT
    rd::RefElemDataT

    boundary_faces::BoundaryFaceT

    current_filename  :: String
    unsaved_changes   :: Bool

    function DGMultiMesh{NDIMS, MeshType, MeshDataT, RefElemDataT, BoundaryFaceT}(md, rd, bd) where {NDIMS, MeshType, MeshDataT, RefElemDataT, BoundaryFaceT}
      return new{NDIMS, MeshType, MeshDataT, RefElemDataT, BoundaryFaceT}(md, rd, bd, "", true)
    end
end

@inline Base.ndims(::DGMultiMesh{NDIMS}) where {NDIMS} = NDIMS
@inline ncells(mesh::DGMultiMesh) = Int(mesh.md.num_elements)

function get_element_type_from_string(input::String)
  str = lowercase(input)
  if startswith(str, "line")
    return Line
  elseif startswith(str, "tri")
    return Tri
  elseif startswith(str, "tet")
    return Tet
  elseif startswith(str, "quad")
    return Quad
  elseif startswith(str, "hex")
    return Hex
  elseif startswith(str, "wedge")
    return Wedge
  elseif startswith(str, "pyr")
    return Pyr
  else
    @error "Unknown element type: $input"
  end
end

const SerialDGMultiMesh{NDIMS} = DGMultiMesh{NDIMS}
@inline mpi_parallel(mesh::SerialDGMultiMesh) = False()

# enable use of @set and setproperties(...) for DGMultiMesh
function ConstructionBase.constructorof(::Type{DGMultiMesh{T1, T2, T3, T4, T5}}) where {T1,
                                                                                    T2,
                                                                                    T3,
                                                                                    T4,
                                                                                    T5}
    DGMultiMesh{T1, T2, T3, T4, T5}
end

function Base.show(io::IO, mesh::DGMultiMesh{NDIMS, MeshType}) where {NDIMS, MeshType}
    @nospecialize mesh # reduce precompilation time
    print(io, "$MeshType DGMultiMesh with NDIMS = $NDIMS.")
end

function Base.show(io::IO, ::MIME"text/plain",
                   mesh::DGMultiMesh{NDIMS, MeshType}) where {NDIMS, MeshType}
    @nospecialize mesh # reduce precompilation time
    if get(io, :compact, false)
        show(io, mesh)
    else
        summary_header(io, "DGMultiMesh{$NDIMS, $MeshType}, ")
        summary_line(io, "number of elements", mesh.md.num_elements)
        summary_line(io, "number of boundaries", length(mesh.boundary_faces))
        for (boundary_name, faces) in mesh.boundary_faces
            summary_line(increment_indent(io), "nfaces on $boundary_name",
                         length(faces))
        end
        summary_footer(io)
    end
end

function DGMultiMesh(md::MeshData{NDIMS}, rd::RefElemData, boundary_names=[]) where {NDIMS}
    return DGMultiMesh{NDIMS, rd.element_type, typeof(md), typeof(rd), typeof(boundary_names)}(md, rd, boundary_names)
end

end # @muladd
