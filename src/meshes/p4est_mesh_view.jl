@muladd begin
#! format: noindent

"""
    P4estMeshView{NDIMS, RealT <: Real, IsParallel, P, Ghost, NDIMSP2,
                             NNODES} <:
               AbstractMesh{NDIMS}
A view on a p4est mesh.
"""

mutable struct P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT <: Real, Parent} <: AbstractMesh{NDIMS}
    parent::Parent
    cell_ids::Vector{Int}
#     unsaved_changes::Bool
end

function P4estMeshView(parent::P4estMesh{NDIMS, NDIMS_AMBIENT, RealT}) where {NDIMS, NDIMS_AMBIENT, RealT}
    # SC: number of cells should be corrected.
    cell_ids = Vector{Int}(undef, ncells(parent))
    # SC: do not populate this array. It needs to be given by the user.
    for i in 1:ncells(parent)
        cell_ids[i] = i
    end

    return P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT, typeof(parent)}(parent, cell_ids)#, parent.unsaved_changes)
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
@inline ndims_ambient(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS

@inline balance!(::P4estMeshView) = nothing
@inline ncells(mesh::P4estMeshView) = length(mesh.cell_ids)

end # @muladd
