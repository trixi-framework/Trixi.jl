# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# In contrast to the CPU version, the GPU version does not use the MPI interface
# container `cache.mpi_interfaces.u`. Instead, `prolong2mpiinterfaces!` writes the
# local data directly into the MPI send buffer and `calc_mpi_interface_flux!` reads
# the remote data directly from the MPI receive buffer, while the local data is read
# from `u`. Both kernels handle the MPI interfaces of all neighbor ranks at once and
# parallelize over all nodes of these interfaces.
function rhs_hyperbolic!(backend::Backend,
                         du, u, t,
                         mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2},
                                     P4estMeshParallel{3}, T8codeMeshParallel{3}},
                         equations,
                         boundary_conditions, source_terms::Source,
                         dg::DG, cache) where {Source}
    # Prolong solution to MPI interfaces, i.e., fill the MPI send buffer
    @trixi_timeit_ext backend timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(backend, cache, u, mesh, equations,
                               dg.surface_integral, dg)
    end

    # Prolong solution to MPI mortars
    @trixi_timeit_ext backend timer() "prolong2mpimortars" begin
        prolong2mpimortars!(backend, cache, u, mesh, equations,
                            dg.mortar, dg)
    end

    # Wait for the kernels above to finish before sending their results via MPI.
    # This also guarantees that the MPI interface flux kernel of the previous call,
    # which reads from the MPI receive buffer, has finished. Thus, we can only start
    # to receive MPI data afterwards, in contrast to the CPU version.
    # MPI is not ordered with respect to the kernels queued on the GPU, which run
    # asynchronously. Thus, all kernels writing to the MPI send buffer must be launched
    # before this synchronization (and after `finish_mpi_send!` of the previous call),
    # and all kernels reading from the MPI receive buffer must be launched after
    # `finish_mpi_receive!`. Keep this in mind when adding kernels using the MPI
    # buffers (e.g., for MPI mortars). Missing synchronization cannot be detected by
    # tests with the CPU backend of KernelAbstractions.jl, which runs kernels
    # synchronously.
    @trixi_timeit timer() "synchronize" KernelAbstractions.synchronize(backend)

    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(backend, cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit_ext backend timer() "volume integral" begin
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
    @trixi_timeit_ext backend timer() "prolong2boundaries" begin
        prolong2boundaries!(backend, cache, u, mesh, equations, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit_ext backend timer() "boundary flux" begin
        calc_boundary_flux!(backend, cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    @trixi_timeit_ext backend timer() "prolong2mortars" begin
        prolong2mortars!(backend, cache, u, mesh, equations,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit_ext backend timer() "mortar flux" begin
        calc_mortar_flux!(backend, cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(backend, cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit_ext backend timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(backend, cache.elements.surface_flux_values, u, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit_ext backend timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(backend, cache.elements.surface_flux_values, mesh,
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

# The MPI send buffer is filled in `prolong2mpiinterfaces!`, and the kernels have
# finished when this is called, see `rhs_hyperbolic!`.
# This function is dimension-agnostic and used in both 2D and 3D.
function start_mpi_send!(backend::Backend, mpi_cache::P4estMPICache,
                         mesh::Union{P4estMeshParallel{2}, P4estMeshParallel{3},
                                     T8codeMeshParallel{2}, T8codeMeshParallel{3}},
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
# This function is dimension-agnostic and used in both 2D and 3D.
function finish_mpi_receive!(backend::Backend, mpi_cache::P4estMPICache,
                             mesh::Union{P4estMeshParallel{2}, P4estMeshParallel{3},
                                         T8codeMeshParallel{2}, T8codeMeshParallel{3}},
                             equations, dg, cache)
    MPI.Waitall(mpi_cache.mpi_recv_requests)

    return nothing
end

Base.@propagate_inbounds function get_mpi_buffer_node_vars(mpi_buffer, equations,
                                                           offset)
    # Explicit bounds check, which can be removed by calling this function with `@inbounds`
    @boundscheck checkbounds(mpi_buffer, offset .+ eachvariable(equations))
    # Assume inbounds access now
    return SVector(ntuple(@inline(v->@inbounds mpi_buffer[offset + v]),
                          Val(nvariables(equations))))
end
end # muladd
