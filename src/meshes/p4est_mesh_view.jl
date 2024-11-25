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

#
function extract_p4est_mesh_view(elements_parent,
                                 interfaces_parent,
                                 boundaries_parent,
                                 mortars_parent,
                                 mesh)
    elements = elements_parent
    elements.inverse_jacobian = elements_parent.inverse_jacobian[.., mesh.cell_ids]
    elements.jacobian_matrix = elements_parent.jacobian_matrix[.., mesh.cell_ids]
    elements.node_coordinates = elements_parent.node_coordinates[.., mesh.cell_ids]
    elements.contravariant_vectors = elements_parent.contravariant_vectors[.., mesh.cell_ids]
    elements.surface_flux_values = elements_parent.surface_flux_values[.., mesh.cell_ids]
    elements._inverse_jacobian = vec(elements.inverse_jacobian)
    elements._jacobian_matrix = vec(elements.jacobian_matrix)
    elements._node_coordinates = vec(elements.node_coordinates)
    elements._node_coordinates = vec(elements.node_coordinates)
    elements._surface_flux_values = vec(elements.surface_flux_values)
    interfaces = extract_interfaces(mesh, interfaces_parent)

    return elements, interfaces, boundaries_parent, mortars_parent
end

function extract_interfaces(mesh::P4estMeshView, interfaces_parent)
    @autoinfiltrate
    interfaces = interfaces_parent
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

    return interfaces
end

# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u,
                             mesh::P4estMeshView{2},
                             equations, surface_integral, dg)
    @unpack interfaces = cache
    index_range = eachnode(dg)

    @threaded for interface in eachinterface(dg, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)
        @autoinfiltrate

        i_primary = i_primary_start
        j_primary = j_primary_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                interfaces.u[1, v, i, interface] = u[v, i_primary, j_primary,
                                                     primary_element]
            end
            i_primary += i_primary_step
            j_primary += j_primary_step
        end

        # Copy solution data from the secondary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        secondary_element = interfaces.neighbor_ids[2, interface]
        secondary_indices = interfaces.node_indices[2, interface]

        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                interfaces.u[2, v, i, interface] = u[v, i_secondary, j_secondary,
                                                     secondary_element]
            end
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    return nothing
end

end # @muladd
