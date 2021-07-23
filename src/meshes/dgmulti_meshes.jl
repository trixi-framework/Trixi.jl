# !!! warning "Experimental features"

abstract type AbstractMeshData{NDIMS, ElemType} end

"""
    VertexMappedMesh{NDIMS, ElemType, Nboundaries, Tv, Ti} <: AbstractMeshData{NDIMS, ElemType}

`VertexMappedMesh` describes a mesh which is constructed by an reference-to-physical
mapping which can be constructed using only the vertex positions.

Wraps `MeshData` and `boundary_faces` in a dispatchable mesh type.
"""
struct VertexMappedMesh{NDIMS, ElemType, MeshDataT <: MeshData{NDIMS}, Nboundaries} <: AbstractMeshData{NDIMS, ElemType}
  md::MeshDataT
  boundary_faces::Dict{Symbol, Vector{Int}}
end

Base.ndims(::VertexMappedMesh{NDIMS}) where {NDIMS} = NDIMS

function Base.show(io::IO, mesh::VertexMappedMesh{NDIMS, ElemType}) where {NDIMS, ElemType}
  @nospecialize mesh # reduce precompilation time
  print(io, "$ElemType VertexMappedMesh with NDIMS = $NDIMS.")
end

function Base.show(io::IO, ::MIME"text/plain", mesh::VertexMappedMesh{NDIMS, ElemType, MeshDataT, Nb}) where {NDIMS, ElemType, MeshDataT, Nb}
  @nospecialize mesh # reduce precompilation time
  if get(io, :compact, false)
    show(io, mesh)
  else
    summary_header(io, "VertexMappedMesh{$NDIMS, $ElemType, $MeshDataT, $Nb}, ")
    summary_line(io, "number of elements", mesh.md.num_elements)
    summary_line(io, "number of boundaries", length(mesh.boundary_faces))
    for (boundary_name, faces) in mesh.boundary_faces
      summary_line(increment_indent(io), "nfaces on $boundary_name", length(faces))
    end
    summary_footer(io)
  end
end

"""
    VertexMappedMesh(vertex_coordinates::NTuple{NDIMS, Vector{Tv}}, EToV, rd::RefElemData;
                     is_on_boundary = nothing,
                     is_periodic::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv}

- `vertex_coordinates` is a tuple of vectors containing x,y,... components of the vertex coordinates
- `EToV` is a 2D array containing element-to-vertex connectivities for each element
- `rd` is a `RefElemData` from `StartUpDG.jl`, and contains information associated with to the
  reference element (e.g., quadrature, basis evaluation, differentiation, etc).
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `is_periodic` is a tuple of booleans specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
function VertexMappedMesh(vertex_coordinates::NTuple{NDIMS, Vector{Tv}}, EToV::Array{Ti,2}, rd::RefElemData;
                          is_on_boundary = nothing,
                          is_periodic::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv, Ti}

  md = MeshData(vertex_coordinates..., EToV, rd)
  md = StartUpDG.make_periodic(md, is_periodic)
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)
  return VertexMappedMesh{NDIMS, typeof(rd.elementType), typeof(md), length(boundary_faces)}(md, boundary_faces)
end

"""
    VertexMappedMesh(vertex_coordinates, EToV, dg::DGMulti;
                     is_on_boundary = nothing,
                     is_periodic::NTuple{NDIMS, Bool} = ntuple(_->false, NDIMS)) where {NDIMS, Tv}

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
VertexMappedMesh(vertex_coordinates, EToV, dg::DGMulti; kwargs...) =
  VertexMappedMesh(vertex_coordinates, EToV, dg.basis; kwargs...)

"""
  VertexMappedMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int})

- `triangulateIO` is a `TriangulateIO` mesh representation
- `rd` is a `RefElemData` from `StartUpDG.jl`, and contains information associated with to the
reference element (e.g., quadrature, basis evaluation, differentiation, etc).
- `boundary_dict` is a `Dict{Symbol, Int}` which associates each integer `TriangulateIO` boundary tag with a Symbol
"""
function VertexMappedMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int})

  vertex_coordinates_x, vertex_coordinates_y, EToV = StartUpDG.triangulateIO_to_VXYEToV(triangulateIO)
  md = MeshData(vertex_coordinates_x, vertex_coordinates_y, EToV, rd)
  boundary_faces = StartUpDG.tag_boundary_faces(triangulateIO, rd, md, boundary_dict)
  return VertexMappedMesh{2, typeof(rd.elementType), typeof(md), length(boundary_faces)}(md, boundary_faces)
end

"""
    VertexMappedMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int})

Constructor which uses `dg::DGMulti` instead of `rd::RefElemData`.
"""
VertexMappedMesh(triangulateIO, dg::DGMulti, boundary_dict::Dict{Symbol, Int}) =
  VertexMappedMesh(triangulateIO, dg.basis, boundary_dict)

# old interface
VertexMappedMesh(vertex_coordinates_x, vertex_coordinates_y, EToV, rd, args...; kwargs...) =
  VertexMappedMesh((vertex_coordinates_x, vertex_coordinates_y), EToV, rd, args...; kwargs...)
VertexMappedMesh(vertex_coordinates_x, vertex_coordinates_y, vertex_coordinates_z, EToV, rd, args...; kwargs...) =
  VertexMappedMesh((vertex_coordinates_x, vertex_coordinates_y, vertex_coordinates_z), EToV, rd, args...; kwargs...)
