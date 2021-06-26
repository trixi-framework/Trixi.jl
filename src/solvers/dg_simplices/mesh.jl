# !!! warning "Experimental features"

abstract type AbstractMeshData{Dim, ElemType} end

"""
    VertexMappedMesh{Dim, ElemType, Nboundaries, Tv, Ti} <: AbstractMeshData{Dim, ElemType}

`VertexMappedMesh` describes a mesh which is constructed by an reference-to-physical 
mapping which can be constructed using only the vertex positions. 

Wraps `MeshData` and `boundary_faces` in a dispatchable mesh type.
"""
struct VertexMappedMesh{Dim, ElemType, Nboundaries, Tv, Ti} <: AbstractMeshData{Dim, ElemType}
  md::MeshData{Dim, Tv, Ti}
  boundary_faces::Dict{Symbol, Vector{Int}}
end

Base.ndims(::VertexMappedMesh{Dim}) where {Dim} = Dim

"""
    VertexMappedMesh(VXYZ::NTuple{Dim, Vector{Tv}}, EToV, rd::RefElemData;
                     is_on_boundary = nothing,
                     is_periodic::NTuple{Dim, Bool} = ntuple(_->false, Dim)) where {Dim, Tv}

- `VXYZ` is a tuple of vectors of vertex coordinates
- `EToV` is a matrix containing element-to-vertex connectivities for each element
- `is_on_boundary` specifies boundary using a `Dict{Symbol, <:Function}`
- `is_periodic` is a tuple of booleans specifying periodicity = `true`/`false` in the (x,y,z) direction.
"""
VertexMappedMesh(VX, VY, EToV, rd, args...; kwargs...) = VertexMappedMesh((VX, VY), EToV, rd, args...; kwargs...)

function VertexMappedMesh(VXYZ::NTuple{Dim, Vector{Tv}}, EToV::Matrix{Ti}, rd::RefElemData;
                          is_on_boundary = nothing,
                          is_periodic::NTuple{Dim, Bool} = ntuple(_->false, Dim)) where {Dim, Tv, Ti}

  md = MeshData(VXYZ..., EToV, rd)
  md = StartUpDG.make_periodic(md, is_periodic)
  boundary_faces = StartUpDG.tag_boundary_faces(md, is_on_boundary)  
  return VertexMappedMesh{Dim, length(boundary_faces), typeof(rd.elementType), Tv, Ti}(md, boundary_faces)
end

"""
    VertexMappedMesh(VXYZ::NTuple{Dim, Vector{Tv}}, EToV, rd::RefElemData;
                     triangulateIO, boundary_dict::Dict{Symbol, Int})

- `VXYZ` is a tuple of vectors of vertex coordinates
- `EToV` is a matrix containing element-to-vertex connectivities for each element
- `triangulateIO` is a `TriangulateIO` mesh representation
- `boundary_dict` is a `Dict{Symbol, Int}` which associates each integer `TriangulateIO` boundary tag with a Symbol
"""
function VertexMappedMesh(triangulateIO, rd::RefElemData{2, Tri}, boundary_dict::Dict{Symbol, Int}) 

  VX, VY, EToV = StartUpDG.triangulateIO_to_VXYEToV(triangulateIO)
  md = MeshData(VX, VY, EToV, rd)
  boundary_faces = StartUpDG.tag_boundary_faces(triangulateIO, rd, md, boundary_dict)  
  return VertexMappedMesh{2, length(boundary_faces), typeof(rd.elementType), eltype(VX), eltype(EToV)}(md, boundary_faces)
end
