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
    # SC: After some thought, we might need to create a p4est pointer to p4est data
    #     conatining the data from the view.
#     unsaved_changes::Bool
end

function P4estMeshView(parent::P4estMesh{NDIMS, NDIMS_AMBIENT, RealT}, cell_ids::Vector) where {NDIMS, NDIMS_AMBIENT, RealT}
#     # SC: number of cells should be corrected.
#     cell_ids = Vector{Int}(undef, ncells(parent))
#     # SC: do not populate this array. It needs to be given by the user.
#     for i in 1:ncells(parent)
#         cell_ids[i] = i
#     end

    # SC: Since we need a p4est pointer no the modified (view) p4est data, we might need a function
    #     like connectivity_structured that computes the connectivity.
    return P4estMeshView{NDIMS, NDIMS_AMBIENT, RealT, typeof(parent)}(parent, cell_ids)#, parent.unsaved_changes)
end

@inline Base.ndims(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS
@inline Base.real(::P4estMeshView{NDIMS, RealT}) where {NDIMS, RealT} = RealT
@inline ndims_ambient(::P4estMeshView{NDIMS}) where {NDIMS} = NDIMS

@inline balance!(::P4estMeshView) = nothing
@inline ncells(mesh::P4estMeshView) = length(mesh.cell_ids)

function extract_interfaces!(mesh::P4estMeshView, interfaces)
    u_new = Array{eltype(interfaces.u)}(undef, (size(interfaces.u)[1:3]..., size(mesh.cell_ids)[1]*2))
    u_new[:, :, :, 1:2:end] .= interfaces.u[:, :, :, (mesh.cell_ids.*2 .-1)]
    u_new[:, :, :, 2:2:end] .= interfaces.u[:, :, :, mesh.cell_ids.*2]
    node_indices_new = Array{eltype(interfaces.node_indices)}(undef, (2, size(mesh.cell_ids)[1]*2))
    node_indices_new[:, 1:2:end] .= interfaces.node_indices[:, (mesh.cell_ids.*2 .-1)]
    node_indices_new[:, 2:2:end] .= interfaces.node_indices[:, (mesh.cell_ids.*2)]
    neighbor_ids_new = Array{eltype(interfaces.neighbor_ids)}(undef, (2, size(mesh.cell_ids)[1]*2))
    neighbor_ids_new[:, 1:2:end] .= interfaces.neighbor_ids[:, (mesh.cell_ids.*2 .-1)]
    neighbor_ids_new[:, 2:2:end] .= interfaces.neighbor_ids[:, (mesh.cell_ids.*2)]
    interfaces.u = u_new
    interfaces.node_indices = node_indices_new
    interfaces.neighbor_ids = neighbor_ids_new
    interfaces._u = vec(u_new)
    interfaces._node_indices = vec(node_indices_new)
    interfaces._neighbor_ids = vec(neighbor_ids_new)
end

end # @muladd
