
abstract type AbstractMeshData{Dim} end

"""
VertexMappedMesh describes a mesh which is constructed by an reference-to-physical 
mapping which can be constructed using only the vertex positions. 

Wraps `MeshData` and `boundary_faces` in a dispatchable mesh type.
"""
struct VertexMappedMesh{Dim,Nboundaries,Tv,Ti} <: AbstractMeshData{Dim}
  md::MeshData{Dim,Tv,Ti}
  boundary_faces::Dict{Symbol,Vector{Int}}
end

Trixi.ndims(::VertexMappedMesh{Dim}) where {Dim} = Dim

"""
  function VertexMappedMesh(VXYZ::NTuple{Dim,Vector{Tv}},EToV,rd::RefElemData;
                            is_on_boundary=nothing,
                            is_periodic::NTuple{Dim,Bool}=ntuple(_->false,Dim)) where {Dim,Tv}

- `VXYZ` is a tuple of vectors of vertex coordinates
- `EToV` is a matrix containing element-to-vertex connectivities for each element
- `is_on_boundary` specifies boundary using Dict{Symbol,<:Function}
- `is_periodic` is a tuple of booleans specifying periodicity = true/false in the (x,y,z) direction.
"""
VertexMappedMesh(VX,VY,EToV,rd,args...;kwargs...) = VertexMappedMesh((VX,VY),EToV,rd,args...;kwargs...)

function VertexMappedMesh(VXYZ::NTuple{Dim,Vector{Tv}}, EToV::Matrix{Ti}, rd::RefElemData;
                          is_on_boundary=nothing,
                          is_periodic::NTuple{Dim,Bool}=ntuple(_->false,Dim)) where {Dim,Tv,Ti}

  md = MeshData(VXYZ...,EToV,rd)
  md = StartUpDG.make_periodic(md,is_periodic)
  boundary_faces = StartUpDG.tag_boundary_faces(md,is_on_boundary)
  return VertexMappedMesh{Dim,length(boundary_faces),Tv,Ti}(md,boundary_faces)
end
