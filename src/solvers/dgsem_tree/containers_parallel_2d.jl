# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Data structures and MPI communication of the 2D `TreeMesh`, currently limited to
# Lobatto-Legendre nodes

function start_mpi_send!(backend::Nothing, mpi_cache::TreeMPICache,
                         mesh::TreeMeshParallel{2}, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    for rank in 1:length(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_cache.mpi_send_buffers[rank]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[rank])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[2, :, :,
                                                                             interface])
            else # local element in negative direction
                @views send_buffer[first:last] .= vec(cache.mpi_interfaces.u[1, :, :,
                                                                             interface])
            end
        end

        # Each mortar has a total size of 4 * data_size, set everything to NaN first and overwrite the
        # parts where local data exists
        interfaces_data_size = length(mpi_cache.mpi_neighbor_interfaces[rank]) *
                               data_size
        mortars_data_size = length(mpi_cache.mpi_neighbor_mortars[rank]) * 4 * data_size
        send_buffer[(interfaces_data_size + 1):(interfaces_data_size + mortars_data_size)] .= NaN

        for (index, mortar) in enumerate(mpi_cache.mpi_neighbor_mortars[rank])
            # First and last indices in the send buffer for mortar data obtained from local element
            # in a given position
            index_base = interfaces_data_size + (index - 1) * 4 * data_size
            indices = (
                       # first, last for local element in position 1 (lower element)
                       (index_base + 1,
                        index_base + 1 * data_size),
                       # first, last for local element in position 2 (upper element)
                       (index_base + 1 * data_size + 1,
                        index_base + 2 * data_size),
                       # firsts, lasts for local element in position 3 (large element)
                       (index_base + 2 * data_size + 1,
                        index_base + 3 * data_size,
                        index_base + 3 * data_size + 1,
                        index_base + 4 * data_size))

            for position in cache.mpi_mortars.local_neighbor_positions[mortar]
                # Determine whether the data belongs to the left or right side
                if cache.mpi_mortars.large_sides[mortar] == 1 # large element on left side
                    if position in (1, 2) # small element
                        leftright = 2
                    else # large element
                        leftright = 1
                    end
                else # large element on right side
                    if position in (1, 2) # small element
                        leftright = 1
                    else # large element
                        leftright = 2
                    end
                end
                # copy data to buffer
                if position == 1 # lower element
                    first, last = indices[position]
                    @views send_buffer[first:last] .= vec(cache.mpi_mortars.u_lower[leftright,
                                                                                    :,
                                                                                    :,
                                                                                    mortar])
                elseif position == 2 # upper element
                    first, last = indices[position]
                    @views send_buffer[first:last] .= vec(cache.mpi_mortars.u_upper[leftright,
                                                                                    :,
                                                                                    :,
                                                                                    mortar])
                else # large element
                    first_lower, last_lower, first_upper, last_upper = indices[position]
                    @views send_buffer[first_lower:last_lower] .= vec(cache.mpi_mortars.u_lower[leftright,
                                                                                                :,
                                                                                                :,
                                                                                                mortar])
                    @views send_buffer[first_upper:last_upper] .= vec(cache.mpi_mortars.u_upper[leftright,
                                                                                                :,
                                                                                                :,
                                                                                                mortar])
                end
            end
        end
    end

    # Start sending
    for (index, rank) in enumerate(mpi_cache.mpi_neighbor_ranks)
        mpi_cache.mpi_send_requests[index] = MPI.Isend(mpi_cache.mpi_send_buffers[index],
                                                       rank, mpi_rank(), mpi_comm())
    end

    return nothing
end

function finish_mpi_receive!(backend::Nothing, mpi_cache::TreeMPICache,
                             mesh::TreeMeshParallel{2}, equations, dg, cache)
    data_size = nvariables(equations) * nnodes(dg)^(ndims(mesh) - 1)

    # Start receiving and unpack received data until all communication is finished
    data = MPI.Waitany(mpi_cache.mpi_recv_requests)
    while data !== nothing
        recv_buffer = mpi_cache.mpi_recv_buffers[data]

        for (index, interface) in enumerate(mpi_cache.mpi_neighbor_interfaces[data])
            first = (index - 1) * data_size + 1
            last = (index - 1) * data_size + data_size

            if cache.mpi_interfaces.remote_sides[interface] == 1 # local element in positive direction
                @views vec(cache.mpi_interfaces.u[1, :, :, interface]) .= recv_buffer[first:last]
            else # local element in negative direction
                @views vec(cache.mpi_interfaces.u[2, :, :, interface]) .= recv_buffer[first:last]
            end
        end

        interfaces_data_size = length(mpi_cache.mpi_neighbor_interfaces[data]) *
                               data_size
        for (index, mortar) in enumerate(mpi_cache.mpi_neighbor_mortars[data])
            # First and last indices in the receive buffer for mortar data obtained from remote element
            # in a given position
            index_base = interfaces_data_size + (index - 1) * 4 * data_size
            indices = (
                       # first, last for local element in position 1 (lower element)
                       (index_base + 1,
                        index_base + 1 * data_size),
                       # first, last for local element in position 2 (upper element)
                       (index_base + 1 * data_size + 1,
                        index_base + 2 * data_size),
                       # firsts, lasts for local element in position 3 (large element)
                       (index_base + 2 * data_size + 1,
                        index_base + 3 * data_size,
                        index_base + 3 * data_size + 1,
                        index_base + 4 * data_size))

            for position in 1:3
                # Skip if received data for `pos` is NaN as no real data has been sent for the
                # corresponding element
                if isnan(recv_buffer[Base.first(indices[position])])
                    continue
                end

                # Determine whether the received data belongs to the left or right side
                if cache.mpi_mortars.large_sides[mortar] == 1 # large element on left side
                    if position in (1, 2) # small element
                        leftright = 2
                    else # large element
                        leftright = 1
                    end
                else # large element on right side
                    if position in (1, 2) # small element
                        leftright = 1
                    else # large element
                        leftright = 2
                    end
                end

                if position == 1 # lower element data has been received
                    first, last = indices[position]
                    @views vec(cache.mpi_mortars.u_lower[leftright, :, :, mortar]) .= recv_buffer[first:last]
                elseif position == 2 # upper element data has been received
                    first, last = indices[position]
                    @views vec(cache.mpi_mortars.u_upper[leftright, :, :, mortar]) .= recv_buffer[first:last]
                else # large element data has been received
                    first_lower, last_lower, first_upper, last_upper = indices[position]
                    @views vec(cache.mpi_mortars.u_lower[leftright, :, :, mortar]) .= recv_buffer[first_lower:last_lower]
                    @views vec(cache.mpi_mortars.u_upper[leftright, :, :, mortar]) .= recv_buffer[first_upper:last_upper]
                end
            end
        end

        data = MPI.Waitany(mpi_cache.mpi_recv_requests)
    end

    return nothing
end

# Initialize connectivity between MPI neighbor ranks
function init_mpi_neighbor_connectivity(elements, mpi_interfaces, mpi_mortars,
                                        mesh::TreeMeshParallel{2})
    tree = mesh.tree

    # Determine neighbor ranks and sides for MPI interfaces
    neighbor_ranks_interface = fill(-1, nmpiinterfaces(mpi_interfaces))
    # The global interface id is the smaller of the (globally unique) neighbor cell ids, multiplied by
    # number of directions (2 * ndims) plus direction minus one
    global_interface_ids = fill(-1, nmpiinterfaces(mpi_interfaces))
    for interface_id in 1:nmpiinterfaces(mpi_interfaces)
        orientation = mpi_interfaces.orientations[interface_id]
        remote_side = mpi_interfaces.remote_sides[interface_id]
        # Direction is from local cell to remote cell
        if orientation == 1 # MPI interface in x-direction
            if remote_side == 1 # remote cell on the "left" of MPI interface
                direction = 1
            else # remote cell on the "right" of MPI interface
                direction = 2
            end
        else # MPI interface in y-direction
            if remote_side == 1 # remote cell on the "left" of MPI interface
                direction = 3
            else # remote cell on the "right" of MPI interface
                direction = 4
            end
        end
        local_neighbor_id = mpi_interfaces.local_neighbor_ids[interface_id]
        local_cell_id = elements.cell_ids[local_neighbor_id]
        remote_cell_id = tree.neighbor_ids[direction, local_cell_id]
        neighbor_ranks_interface[interface_id] = tree.mpi_ranks[remote_cell_id]
        if local_cell_id < remote_cell_id
            global_interface_ids[interface_id] = 2 * ndims(tree) * local_cell_id +
                                                 direction - 1
        else
            global_interface_ids[interface_id] = (2 * ndims(tree) * remote_cell_id +
                                                  opposite_direction(direction) - 1)
        end
    end

    # Determine neighbor ranks for MPI mortars
    neighbor_ranks_mortar = Vector{Vector{Int}}(undef, nmpimortars(mpi_mortars))
    # The global mortar id is the (globally unique) large cell id, multiplied by
    # number of directions (2 * ndims) plus direction minus one where
    # direction = 1 for mortars in x-direction where large element is left
    # direction = 2 for mortars in x-direction where large element is right
    # direction = 3 for mortars in y-direction where large element is left
    # direction = 4 for mortars in y-direction where large element is right
    global_mortar_ids = fill(-1, nmpimortars(mpi_mortars))
    for mortar in 1:nmpimortars(mpi_mortars)
        neighbor_ranks_mortar[mortar] = Vector{Int}()

        orientation = mpi_mortars.orientations[mortar]
        large_side = mpi_mortars.large_sides[mortar]
        direction = (orientation - 1) * 2 + large_side

        local_neighbor_ids = mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = mpi_mortars.local_neighbor_positions[mortar]
        if 3 in local_neighbor_positions # large element is on this rank
            large_element_id = local_neighbor_ids[findfirst(pos -> pos == 3,
                                                            local_neighbor_positions)]
            large_cell_id = elements.cell_ids[large_element_id]
        else # large element is remote
            cell_id = elements.cell_ids[first(local_neighbor_ids)]
            large_cell_id = tree.neighbor_ids[direction, tree.parent_ids[cell_id]]
        end

        neighbor_cell_id = tree.neighbor_ids[opposite_direction(direction),
                                             large_cell_id]
        if direction == 1
            lower_cell_id = tree.child_ids[1, neighbor_cell_id]
            upper_cell_id = tree.child_ids[3, neighbor_cell_id]
        elseif direction == 2
            lower_cell_id = tree.child_ids[2, neighbor_cell_id]
            upper_cell_id = tree.child_ids[4, neighbor_cell_id]
        elseif direction == 3
            lower_cell_id = tree.child_ids[1, neighbor_cell_id]
            upper_cell_id = tree.child_ids[2, neighbor_cell_id]
        else
            lower_cell_id = tree.child_ids[3, neighbor_cell_id]
            upper_cell_id = tree.child_ids[4, neighbor_cell_id]
        end

        for cell_id in (lower_cell_id, upper_cell_id, large_cell_id)
            if !is_own_cell(tree, cell_id)
                neighbor_rank = tree.mpi_ranks[cell_id]
                if !(neighbor_rank in neighbor_ranks_mortar[mortar])
                    push!(neighbor_ranks_mortar[mortar], neighbor_rank)
                end
            end
        end

        global_mortar_ids[mortar] = 2 * ndims(tree) * large_cell_id + direction - 1
    end

    # Get sorted, unique neighbor ranks
    mpi_neighbor_ranks = vcat(neighbor_ranks_interface, neighbor_ranks_mortar...) |>
                         sort |> unique

    # Sort interfaces by global interface id
    p = sortperm(global_interface_ids)
    neighbor_ranks_interface .= neighbor_ranks_interface[p]
    interface_ids = collect(1:nmpiinterfaces(mpi_interfaces))[p]

    # Sort mortars by global mortar id
    p = sortperm(global_mortar_ids)
    neighbor_ranks_mortar .= neighbor_ranks_mortar[p]
    mortar_ids = collect(1:nmpimortars(mpi_mortars))[p]

    # For each neighbor rank, init connectivity data structures
    mpi_neighbor_interfaces = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    mpi_neighbor_mortars = Vector{Vector{Int}}(undef, length(mpi_neighbor_ranks))
    for (index, rank) in enumerate(mpi_neighbor_ranks)
        mpi_neighbor_interfaces[index] = interface_ids[findall(x -> (x == rank),
                                                               neighbor_ranks_interface)]
        mpi_neighbor_mortars[index] = mortar_ids[findall(x -> (rank in x),
                                                         neighbor_ranks_mortar)]
    end

    # Sanity checks that we counted all interfaces exactly once
    @assert sum(length(v) for v in mpi_neighbor_interfaces) ==
            nmpiinterfaces(mpi_interfaces)

    return mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars
end
end # @muladd
