
# todo: fix. These are temporary stopgaps for when the mesh is just a tuple = (VX,EToV) or (VX,VY,EToV)
# todo: remove and replace this with Mesh types for dispatch
Trixi.ndims(mesh::Tuple{Vector{Tv},Vector{Tv},Matrix{Ti}}) where {Tv,Ti} = 2
function StartUpDG.MeshData(mesh::Tuple{Vector{Tv},Matrix{Ti}},rd) where {Tv,Ti} 
    md = MeshData(mesh...,rd)
    return md
end
function StartUpDG.MeshData(mesh::Tuple{Vector{Tv},Vector{Tv},Matrix{Ti}},rd) where {Tv,Ti}
    md = MeshData(mesh...,rd)
    md = StartUpDG.make_periodic(md,rd) # todo: remove
    return md
end

# # # todo: extend StartUpDG.MeshData to CurvedMesh and UnstructuredQuadMesh
# abstract type AbstractAffineMesh{Dim} <: AbstractMesh{Dim} end

# # deals with both triangles/quads
# struct AffineMesh2D{Tv,Ti} <: AbstractAffineMesh{2} 
#     VX::Vector{Tv}
#     VY::Vector{Tv}
#     EToV::Matrix{Ti}
#     periodicity::NTuple{2,Bool}
# end

# StartUpDG.MeshData(mesh::Mesh,rd) where {Mesh <: AbstractAffineMesh{2}} = MeshData(mesh.VX,mesh.VY,mesh.EToV,rd)