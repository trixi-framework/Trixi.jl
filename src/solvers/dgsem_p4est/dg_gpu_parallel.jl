# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO: Support MPI mortars
@inline function _start_mpi_send!(backend::Backend, mpi_cache::P4estMPICache,
                                  mesh::ParallelP4estMesh{3}, equations, dg, cache)
    @unpack mpi_neighbor_ranks, mpi_neighbor_interfaces = mpi_cache
    @unpack mpi_send_buffers, mpi_send_requests = mpi_cache
    @unpack mpi_interfaces = cache
    @unpack local_sides = mpi_interfaces

    kernel! = copy_to_mpi_send!(backend)

    for (index, d) in enumerate(mpi_neighbor_ranks)
        send_buffer = mpi_send_buffers[index]
        neighbor_interfaces = mpi_neighbor_interfaces[index]
        kernel!(send_buffer, neighbor_interfaces, local_sides, mpi_interfaces.u,
                Val(nvariables(equations)), Val(ndims(mesh)),
                ndrange = (nnodes(dg), nnodes(dg), length(neighbor_interfaces)))
        synchronize(backend)
        mpi_send_requests[index] = MPI.Isend(send_buffer, d, mpi_rank(), mpi_comm())
    end
end

@kernel function copy_to_mpi_send!(send_buffer, neighbor_interfaces, local_sides,
                                   u_mpi_interfaces, ::Val{NVARS},
                                   ::Val{3}) where {NVARS}
    i, j, k = @index(Global, NTuple)
    I = @index(Global, Linear)
    buf_idx = (I - 1) * NVARS
    interface = neighbor_interfaces[k]
    local_side = local_sides[interface]
    for v in 1:NVARS
        send_buffer[buf_idx + v] = u_mpi_interfaces[local_side, v, i, j, interface]
    end
end

@inline function _finish_mpi_receive!(backend::Backend, mpi_cache::P4estMPICache,
                                      mesh, equations, dg, cache)
    @unpack mpi_neighbor_ranks, mpi_neighbor_interfaces = mpi_cache
    @unpack mpi_recv_buffers, mpi_recv_requests = mpi_cache
    @unpack mpi_interfaces = cache
    @unpack local_sides = mpi_interfaces

    kernel! = copy_from_mpi_recv!(backend)

    d = MPI.Waitany(mpi_recv_requests)
    while d !== nothing
        recv_buffer = mpi_recv_buffers[d]
        neighbor_interfaces = mpi_neighbor_interfaces[d]
        kernel!(recv_buffer, neighbor_interfaces, local_sides, mpi_interfaces.u,
                Val(nvariables(equations)), Val(ndims(mesh)),
                ndrange = (nnodes(dg), nnodes(dg), length(neighbor_interfaces)))

        d = MPI.Waitany(mpi_recv_requests)
    end
    synchronize(backend)
end

@kernel function copy_from_mpi_recv!(recv_buffer, neighbor_interfaces, local_sides,
                                     u_mpi_interfaces, ::Val{NVARS},
                                     ::Val{3}) where {NVARS}
    i, j, k = @index(Global, NTuple)
    I = @index(Global, Linear)
    buf_idx = (I - 1) * NVARS
    interface = neighbor_interfaces[k]
    remote_side = local_sides[interface] == 1 ? 2 : 1
    for v in 1:NVARS
        u_mpi_interfaces[remote_side, v, i, j, interface] = recv_buffer[buf_idx + v]
    end
end
end # @muladd