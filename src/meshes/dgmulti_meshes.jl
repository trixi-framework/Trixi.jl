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
struct DGMultiMesh{NDIMS, MeshType, MeshDataT <: MeshData{NDIMS}, BoundaryFaceT}
    md::MeshDataT
    boundary_faces::BoundaryFaceT
end

# enable use of @set and setproperties(...) for DGMultiMesh
function ConstructionBase.constructorof(::Type{DGMultiMesh{T1, T2, T3, T4}}) where {T1,
                                                                                    T2,
                                                                                    T3,
                                                                                    T4}
    DGMultiMesh{T1, T2, T3, T4}
end

Base.ndims(::DGMultiMesh{NDIMS}) where {NDIMS} = NDIMS

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
end # @muladd
