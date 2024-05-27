# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize node_indices of MPI interface container
@inline function init_mpi_interface_node_indices!(mpi_interfaces::P4estMPIInterfaceContainer{2},
                                                  faces, local_side, orientation,
                                                  mpi_interface_id)
    # Align interface in positive coordinate direction of primary element.
    # For orientation == 1, the secondary element needs to be indexed backwards
    # relative to the interface.
    if local_side == 1 || orientation == 0
        # Forward indexing
        i = :i_forward
    else
        # Backward indexing
        i = :i_backward
    end

    if faces[local_side] == 0
        # Index face in negative x-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (:begin, i)
    elseif faces[local_side] == 1
        # Index face in positive x-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (:end, i)
    elseif faces[local_side] == 2
        # Index face in negative y-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (i, :begin)
    else # faces[local_side] == 3
        # Index face in positive y-direction
        mpi_interfaces.node_indices[mpi_interface_id] = (i, :end)
    end

    return mpi_interfaces
end

# Normal directions of small element surfaces are needed to calculate the mortar fluxes. Initialize
# them for locally available small elements.
function init_normal_directions!(mpi_mortars::P4estMPIMortarContainer{2}, basis,
                                 elements)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = mpi_mortars
    @unpack contravariant_vectors = elements
    index_range = eachnode(basis)

    @threaded for mortar in 1:nmpimortars(mpi_mortars)
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        for (element, position) in zip(local_neighbor_ids[mortar],
                                       local_neighbor_positions[mortar])
            # ignore large elements
            if position == 3
                continue
            end

            i_small = i_small_start
            j_small = j_small_start
            for node in eachnode(basis)
                # Get the normal direction on the small element.
                # Note, contravariant vectors at interfaces in negative coordinate direction
                # are pointing inwards. This is handled by `get_normal_direction`.
                normal_direction = get_normal_direction(small_direction,
                                                        contravariant_vectors,
                                                        i_small, j_small, element)
                @views mpi_mortars.normal_directions[:, node, position, mortar] .= normal_direction

                i_small += i_small_step
                j_small += j_small_step
            end
        end
    end
end
end # muladd
