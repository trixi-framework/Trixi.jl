# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize node_indices of MPI interface container
@inline function init_mpi_interface_node_indices!(mpi_interfaces::P4estMPIInterfaceContainer{3},
                                                  faces, local_side, orientation,
                                                  mpi_interface_id)
    # Align interface at the primary element (primary element has surface indices (:i_forward, :j_forward)).
    # The secondary element needs to be indexed differently.
    if local_side == 1
        surface_index1 = :i_forward
        surface_index2 = :j_forward
    else # local_side == 2
        surface_index1, surface_index2 = orientation_to_indices_p4est(faces[2],
                                                                      faces[1],
                                                                      orientation)
    end

    if faces[local_side] == 0
        # Index face in negative x-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (:begin, surface_index1,
                                                         surface_index2)
    elseif faces[local_side] == 1
        # Index face in positive x-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (:end, surface_index1,
                                                         surface_index2)
    elseif faces[local_side] == 2
        # Index face in negative y-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, :begin,
                                                         surface_index2)
    elseif faces[local_side] == 3
        # Index face in positive y-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, :end,
                                                         surface_index2)
    elseif faces[local_side] == 4
        # Index face in negative z-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, surface_index2,
                                                         :begin)
    else # faces[local_side] == 5
        # Index face in positive z-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (surface_index1, surface_index2,
                                                         :end)
    end

    return mpi_interfaces
end

# Initialize node_indices of MPI mortar container. Works the same as for its serial counterpart.
# faces[1] is expected to be the face of the small side.
@inline function init_mortar_node_indices!(mortars::P4estMPIMortarContainer{3},
                                           faces, orientation, mortar_id)
    for side in 1:2
        # Align mortar at small side.
        # The large side needs to be indexed differently.
        if side == 1
            surface_index1 = :i_forward
            surface_index2 = :j_forward
        else
            surface_index1, surface_index2 = orientation_to_indices_p4est(faces[2],
                                                                          faces[1],
                                                                          orientation)
        end

        if faces[side] == 0
            # Index face in negative x-direction
            mortars.node_indices[side, mortar_id] = (:begin, surface_index1,
                                                     surface_index2)
        elseif faces[side] == 1
            # Index face in positive x-direction
            mortars.node_indices[side, mortar_id] = (:end, surface_index1,
                                                     surface_index2)
        elseif faces[side] == 2
            # Index face in negative y-direction
            mortars.node_indices[side, mortar_id] = (surface_index1, :begin,
                                                     surface_index2)
        elseif faces[side] == 3
            # Index face in positive y-direction
            mortars.node_indices[side, mortar_id] = (surface_index1, :end,
                                                     surface_index2)
        elseif faces[side] == 4
            # Index face in negative z-direction
            mortars.node_indices[side, mortar_id] = (surface_index1, surface_index2,
                                                     :begin)
        else # faces[side] == 5
            # Index face in positive z-direction
            mortars.node_indices[side, mortar_id] = (surface_index1, surface_index2,
                                                     :end)
        end
    end

    return mortars
end

# Normal directions of small element surfaces are needed to calculate the mortar fluxes. Initialize
# them for locally available small elements.
function init_normal_directions!(mpi_mortars::P4estMPIMortarContainer{3}, basis,
                                 elements)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = mpi_mortars
    @unpack contravariant_vectors = elements
    index_range = eachnode(basis)

    @threaded for mortar in 1:nmpimortars(mpi_mortars)
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        for (element, position) in zip(local_neighbor_ids[mortar],
                                       local_neighbor_positions[mortar])
            # ignore large elements
            if position == 5
                continue
            end

            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            for j in eachnode(basis)
                for i in eachnode(basis)
                    # Get the normal direction on the small element.
                    # Note, contravariant vectors at interfaces in negative coordinate direction
                    # are pointing inwards. This is handled by `get_normal_direction`.
                    normal_direction = get_normal_direction(small_direction,
                                                            contravariant_vectors,
                                                            i_small, j_small, k_small,
                                                            element)
                    @views mpi_mortars.normal_directions[:, i, j, position, mortar] .= normal_direction

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
                i_small += i_small_step_j
                j_small += j_small_step_j
                k_small += k_small_step_j
            end
        end
    end
end
end # muladd
