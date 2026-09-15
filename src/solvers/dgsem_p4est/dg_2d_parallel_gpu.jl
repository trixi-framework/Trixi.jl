# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs_hyperbolic!(backend::Backend,
                         du, u, t,
                         mesh::Union{TreeMeshParallel{2}, P4estMeshParallel{2},
                                     T8codeMeshParallel{2}}, equations,
                         boundary_conditions, source_terms::Source,
                         dg::DG, cache) where {Source}
    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(backend, cache, u, mesh, equations, dg.surface_integral,
                               dg)
    end

    # Prolong solution to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars" begin
        prolong2mpimortars!(cache, u, mesh, equations,
                            dg.mortar, dg)
    end

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(backend, cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Calculate interface fluxes
    @trixi_timeit_ext backend timer() "prolong2interfaces + flux" begin
        prolong2interfaces_and_calc_interface_flux!(backend,
                                                    cache.elements.surface_flux_values,
                                                    u, mesh,
                                                    have_nonconservative_terms(equations),
                                                    equations,
                                                    dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u, mesh, equations,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(backend, cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(backend, cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(cache.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals, apply Jacobian from mapping to reference element
    # and calculate source terms
    @trixi_timeit_ext backend timer() "surface, Jacobian + source terms" begin
        calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend, du, u, t,
                                                                   source_terms, mesh,
                                                                   equations,
                                                                   dg.surface_integral,
                                                                   dg, cache)
    end

    # Finish to send MPI data
    @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

    return nothing
end

function prolong2mpiinterfaces!(backend::Backend, cache, u,
                                mesh::Union{P4estMeshParallel{2},
                                            T8codeMeshParallel{2}},
                                equations, surface_integral, dg::DG)
    nmpiinterfaces(dg, cache) == 0 && return nothing
    @unpack local_sides, local_neighbor_ids, node_indices = cache.mpi_interfaces
    index_range = eachnode(dg)
    variables_range = eachvariable(equations)

    kernel! = prolong2mpiinterfaces_KAkernel!(backend)
    kernel!(cache.mpi_interfaces.u, local_sides, local_neighbor_ids, node_indices,
            index_range, variables_range, u,
            ndrange = nmpiinterfaces(dg, cache))
    return nothing
end

@kernel function prolong2mpiinterfaces_KAkernel!(mpi_interfaces_u, local_sides,
                                                 local_neighbor_ids, node_indices,
                                                 index_range, variables_range, u)
    interface = @index(Global)
    prolong2mpiinterfaces_per_interface!(mpi_interfaces_u, interface, local_sides,
                                         local_neighbor_ids, node_indices, index_range,
                                         variables_range, u)
end

function calc_mpi_interface_flux!(backend::Backend, surface_flux_values,
                                  mesh::Union{P4estMeshParallel{2},
                                              T8codeMeshParallel{2}},
                                  have_nonconservative_terms,
                                  equations, surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)

    kernel! = calc_mpi_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, typeof(mesh),
            have_nonconservative_terms, equations,
            surface_integral, typeof(dg),
            local_neighbor_ids, node_indices, local_sides, contravariant_vectors,
            u, index_range,
            ndrange = nmpiinterfaces(dg, cache))
    return nothing
end

@kernel function calc_mpi_interface_flux_KAkernel!(surface_flux_values,
                                                   MeshT::Type{<:Union{P4estMeshParallel{2},
                                                                       T8codeMeshParallel{2}}},
                                                   have_nonconservative_terms,
                                                   equations,
                                                   surface_integral,
                                                   SolverT::Type{<:DG},
                                                   local_neighbor_ids, node_indices,
                                                   local_sides, contravariant_vectors,
                                                   u,
                                                   index_range)
    interface = @index(Global)
    calc_mpi_interface_flux_per_interface!(surface_flux_values, MeshT,
                                           have_nonconservative_terms, equations,
                                           surface_integral, SolverT,
                                           local_neighbor_ids, node_indices,
                                           local_sides, contravariant_vectors, u,
                                           index_range, interface)
end

# TODO GPU: MPI mortars
function start_mpi_send!(backend::Backend, mpi_cache::P4estMPICache,
                         mesh::P4estMeshParallel{2}, equations, dg, cache)
    @unpack mpi_neighbor_ranks, mpi_neighbor_interfaces = mpi_cache
    @unpack mpi_send_buffers, mpi_send_requests = mpi_cache
    @unpack local_sides, u = cache.mpi_interfaces

    kernel! = start_mpi_send_KAkernel!(backend)

    for (rank_index, neighbor_rank) in enumerate(mpi_neighbor_ranks)
        send_buffer = mpi_send_buffers[rank_index]
        neighbor_interfaces = mpi_neighbor_interfaces[rank_index]
        kernel!(send_buffer, neighbor_interfaces, local_sides, u,
                Val(nvariables(equations)), Val(ndims(mesh)),
                ndrange = (nnodes(dg), length(neighbor_interfaces)))

        # wait for the kernel to return before sending the buffer
        KernelAbstractions.synchronize(backend)
        mpi_send_requests[rank_index] = MPI.Isend(send_buffer, neighbor_rank,
                                                  mpi_rank(), mpi_comm())
    end
end

@kernel function start_mpi_send_KAkernel!(send_buffer, neighbor_interfaces, local_sides,
                                          u_mpi_interfaces, ::Val{NVARS},
                                          ::Val{2}) where {NVARS}
    index_node, index_interface = @index(Global, NTuple)
    index_linear = @index(Global, Linear)

    buffer_offset = (index_linear - 1) * NVARS
    interface = neighbor_interfaces[index_interface]
    local_side = local_sides[interface]

    for v in 1:NVARS
        send_buffer[buffer_offset + v] = u_mpi_interfaces[local_side, v, index_node,
                                                          index_interface]
    end
end

# TODO GPU: MPI mortars
function finish_mpi_receive!(backend::Backend, mpi_cache::P4estMPICache,
                             mesh::P4estMeshParallel{2}, equations, dg, cache)
    @unpack mpi_neighbor_interfaces = mpi_cache
    @unpack mpi_recv_buffers, mpi_recv_requests = mpi_cache
    @unpack local_sides, u = cache.mpi_interfaces

    kernel! = finish_mpi_receive_KAkernel!(backend)

    # Start receiving and unpack received data until all communication is finished
    data = MPI.Waitany(mpi_recv_requests)
    while data !== nothing
        recv_buffer = mpi_recv_buffers[data]
        neighbor_interfaces = mpi_neighbor_interfaces[data]
        kernel!(recv_buffer, neighbor_interfaces, local_sides, u,
                Val(nvariables(equations)), Val(ndims(mesh)),
                ndrange = (nnodes(dg), length(neighbor_interfaces)))

        data = MPI.Waitany(mpi_recv_requests)
    end
    # Wait for the last kernel to return ?
    KernelAbstractions.synchronize(backend)
end

@kernel function finish_mpi_receive_KAkernel!(recv_buffer, neighbor_interfaces,
                                              local_sides,
                                              u_mpi_interfaces, ::Val{NVARS},
                                              ::Val{2}) where {NVARS}
    index_node, index_interface = @index(Global, NTuple)
    index_linear = @index(Global, Linear)
    buffer_offset = (index_linear - 1) * NVARS
    interface = neighbor_interfaces[index_interface]
    remote_side = local_sides[interface] == 1 ? 2 : 1
    for v in 1:NVARS
        u_mpi_interfaces[remote_side, v, index_node, interface] = recv_buffer[buffer_offset + v]
    end
end
end
