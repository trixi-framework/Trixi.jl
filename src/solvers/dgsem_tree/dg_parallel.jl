# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Function to output the mpi rank for visualization
function get_element_variables!(element_variables,
                                mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                            T8codeMeshParallel},
                                dg, cache)
    element_variables[:mpi_rank] = ones(real(dg), nelements(mesh, dg, cache)) *
                                   mpi_rank()
    return nothing
end

# Initialize MPI data structures for the `TreeMesh`. This is dimension-agnostic.
# See `init_mpi_buffers` for the `P4estMesh` and `T8codeMesh`.
function init_mpi_data_structures(mpi_neighbor_interfaces, mpi_neighbor_mortars, n_dims,
                                  nvars, n_nodes, uEltype)
    data_size = nvars * n_nodes^(n_dims - 1)
    n_small_elements = 2^(n_dims - 1)
    mpi_send_buffers = Vector{Vector{uEltype}}(undef, length(mpi_neighbor_interfaces))
    mpi_recv_buffers = Vector{Vector{uEltype}}(undef, length(mpi_neighbor_interfaces))
    for index in 1:length(mpi_neighbor_interfaces)
        mpi_send_buffers[index] = Vector{uEltype}(undef,
                                                  length(mpi_neighbor_interfaces[index]) *
                                                  data_size +
                                                  length(mpi_neighbor_mortars[index]) *
                                                  n_small_elements * 2 * data_size)
        mpi_recv_buffers[index] = Vector{uEltype}(undef,
                                                  length(mpi_neighbor_interfaces[index]) *
                                                  data_size +
                                                  length(mpi_neighbor_mortars[index]) *
                                                  n_small_elements * 2 * data_size)
    end

    mpi_send_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))
    mpi_recv_requests = Vector{MPI.Request}(undef, length(mpi_neighbor_interfaces))

    return mpi_send_buffers, mpi_recv_buffers, mpi_send_requests, mpi_recv_requests
end

# Buffers used by `apply_smoothing_mpi!` to exchange the blending factors of shock
# capturing indicators with neighboring MPI ranks. They are resized in
# `apply_smoothing_mpi!` and remain empty for serial meshes. The requests do not need
# to keep the buffers alive since they are stored here and all requests are completed
# in `apply_smoothing_mpi!`. Reusing the requests avoids allocations.
function create_alpha_mpi_buffers(uEltype)
    send_buffer = Vector{uEltype}()
    recv_buffer = Vector{uEltype}()
    send_requests = MPI.UnsafeMultiRequest()
    recv_requests = MPI.UnsafeMultiRequest()

    return (; send_buffer, recv_buffer, send_requests, recv_requests)
end

# `apply_smoothing!` only considers interfaces and mortars whose elements are all
# local. For MPI-parallel meshes, this function extends the smoothing to MPI interfaces
# and MPI mortars. It must be called after `apply_smoothing!`, which leaves the
# blending factors before smoothing in `alpha_tmp`. Since each update only increases
# `alpha[element]` to `0.5 * alpha_tmp[neighbor]` if this is larger, the result does
# not depend on the order of the updates and is the same as for a serial mesh.
apply_smoothing_mpi!(mesh, alpha, alpha_tmp, indicator_cache, dg, cache) = nothing

function apply_smoothing_mpi!(mesh::Union{TreeMeshParallel, P4estMeshParallel,
                                          T8codeMeshParallel},
                              alpha, alpha_tmp, indicator_cache, dg, cache)
    @unpack mpi_neighbor_ranks, mpi_neighbor_interfaces, mpi_neighbor_mortars = cache.mpi_cache
    @unpack send_buffer, recv_buffer, send_requests, recv_requests = indicator_cache.alpha_mpi_buffers
    @unpack local_neighbor_ids, local_neighbor_positions = cache.mpi_mortars
    # positions `1:n_small_elements` denote the small elements of a mortar,
    # position `n_positions` denotes the large element
    n_positions = 2^(ndims(mesh) - 1) + 1

    # Smoothing between local elements of MPI mortars
    for mortar in eachmpimortar(dg, cache)
        large_index = findfirst(==(n_positions), local_neighbor_positions[mortar])
        if large_index === nothing
            continue
        end
        large = local_neighbor_ids[mortar][large_index]
        for (small, position) in zip(local_neighbor_ids[mortar],
                                     local_neighbor_positions[mortar])
            if position < n_positions
                alpha[small] = max(alpha[small], 0.5f0 * alpha_tmp[large])
                alpha[large] = max(alpha[large], 0.5f0 * alpha_tmp[small])
            end
        end
    end

    # The data exchanged with each neighbor rank consists of the blending factor of
    # the local element of each MPI interface shared with this rank, followed by
    # `n_positions` blending factors for each MPI mortar shared with this rank
    buffer_size = 0
    for index in eachindex(mpi_neighbor_ranks)
        buffer_size += length(mpi_neighbor_interfaces[index]) +
                       n_positions * length(mpi_neighbor_mortars[index])
    end
    resize!(send_buffer, buffer_size)
    resize!(recv_buffer, buffer_size)
    resize!(send_requests, length(mpi_neighbor_ranks))
    resize!(recv_requests, length(mpi_neighbor_ranks))

    # The tag differs from the tags `0:(mpi_nranks() - 1)` of the exchange of the
    # solution in `rhs!`, which may still be in progress when this function is called.
    tag = mpi_nranks()
    offset = 0
    for (index, rank) in enumerate(mpi_neighbor_ranks)
        first = offset + 1
        for interface in mpi_neighbor_interfaces[index]
            offset += 1
            send_buffer[offset] = alpha_tmp[cache.mpi_interfaces.local_neighbor_ids[interface]]
        end
        for mortar in mpi_neighbor_mortars[index]
            # Zero is a neutral value for the smoothing since the blending factors are
            # nonnegative. It is sent for the positions of non-local elements.
            for position in 1:n_positions
                send_buffer[offset + position] = zero(eltype(send_buffer))
            end
            for (element, position) in zip(local_neighbor_ids[mortar],
                                           local_neighbor_positions[mortar])
                send_buffer[offset + position] = alpha_tmp[element]
            end
            offset += n_positions
        end

        MPI.Isend(mpi_buffer(view(send_buffer, first:offset)), rank, tag, mpi_comm(),
                  send_requests[index])
        MPI.Irecv!(mpi_buffer(view(recv_buffer, first:offset)), rank, tag, mpi_comm(),
                   recv_requests[index])
    end

    MPI.Waitall(recv_requests)

    offset = 0
    for index in eachindex(mpi_neighbor_ranks)
        for interface in mpi_neighbor_interfaces[index]
            offset += 1
            element = cache.mpi_interfaces.local_neighbor_ids[interface]
            alpha[element] = max(alpha[element], 0.5f0 * recv_buffer[offset])
        end
        for mortar in mpi_neighbor_mortars[index]
            for (element, position) in zip(local_neighbor_ids[mortar],
                                           local_neighbor_positions[mortar])
                if position == n_positions
                    for small_position in 1:(n_positions - 1)
                        alpha[element] = max(alpha[element],
                                             0.5f0 *
                                             recv_buffer[offset + small_position])
                    end
                else
                    alpha[element] = max(alpha[element],
                                         0.5f0 * recv_buffer[offset + n_positions])
                end
            end
            offset += n_positions
        end
    end

    MPI.Waitall(send_requests)

    return nothing
end
end # muladd
