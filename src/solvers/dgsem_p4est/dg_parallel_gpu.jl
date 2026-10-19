# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The MPI send buffer is filled in `prolong2mpiinterfaces!`, and the kernels have
# finished when this is called, see `rhs_hyperbolic!`.
# This function is dimension-agnostic and also used in 3D.
function start_mpi_send!(backend::Backend, mpi_cache::P4estMPICache,
                         mesh::Union{P4estMeshParallel, T8codeMeshParallel},
                         equations, dg, cache)
    # TODO GPU: MPI mortars are not yet implemented
    @assert isempty(eachmpimortar(dg, cache)) "MPI mortars are not yet supported on GPU backends"

    for (index, rank) in enumerate(mpi_cache.mpi_neighbor_ranks)
        send_buffer = mpi_buffer(get_mpi_send_buffer(mpi_cache, index))
        mpi_cache.mpi_send_requests[index] = MPI.Isend(send_buffer, rank, mpi_rank(),
                                                       mpi_comm())
    end

    return nothing
end

# The received data is not unpacked but read directly from the MPI receive buffer
# by `calc_mpi_interface_flux!`. Thus, we can wait for all data at once.
# This function is dimension-agnostic and also used in 3D.
function finish_mpi_receive!(backend::Backend, mpi_cache::P4estMPICache,
                             mesh::Union{P4estMeshParallel, T8codeMeshParallel},
                             equations, dg, cache)
    MPI.Waitall(mpi_cache.mpi_recv_requests)

    return nothing
end
end # muladd