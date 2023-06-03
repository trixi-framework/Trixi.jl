# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize MPI data structures. This works for both the
# `TreeMesh` and the `P4estMesh` and is dimension-agnostic.
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
end # muladd
