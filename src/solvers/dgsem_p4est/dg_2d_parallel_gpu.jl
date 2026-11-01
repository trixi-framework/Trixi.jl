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
                         mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2}},
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

# Copy the local solution data at all MPI interfaces directly into the MPI send buffer.
# The data of each interface node is stored at the same position as in the CPU version,
# which copies `cache.mpi_interfaces.u[local_side, :, i, interface]` into the buffer.
function prolong2mpiinterfaces!(backend::Backend, cache, u,
                                mesh::Union{P4estMeshParallel{2},
                                            T8codeMeshParallel{2}},
                                equations, surface_integral, dg::DG)
    nmpiinterfaces(dg, cache) == 0 && return nothing
    @unpack local_neighbor_ids, node_indices = cache.mpi_interfaces
    @unpack mpi_send_buffer, mpi_interfaces_buffer_offsets = cache.mpi_cache
    index_range = eachnode(dg)

    # Explicit bounds check, which allows us to assume inbounds access in the kernel.
    # The offsets are constructed in `init_mpi_buffers` such that the data of all
    # MPI interfaces fits into the MPI buffers.
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(cache.mpi_interfaces, equations, dg, cache)
        check_axes(mpi_interfaces_buffer_offsets, (eachmpiinterface(dg, cache),))
        checkbounds(mpi_send_buffer,
                    1:(nvariables(equations) * nnodes(dg) * nmpiinterfaces(dg, cache)))
    end

    kernel! = prolong2mpiinterfaces_KAkernel!(backend)
    kernel!(mpi_send_buffer, u, typeof(mesh), equations, dg, local_neighbor_ids,
            node_indices, mpi_interfaces_buffer_offsets, index_range,
            ndrange = (nnodes(dg), nmpiinterfaces(dg, cache)))
    return nothing
end

@kernel inbounds=true function prolong2mpiinterfaces_KAkernel!(mpi_send_buffer, u,
                                                               MeshT::Type{<:Union{P4estMeshParallel{2},
                                                                                   T8codeMeshParallel{2}}},
                                                               equations, dg,
                                                               local_neighbor_ids,
                                                               node_indices,
                                                               mpi_interfaces_buffer_offsets,
                                                               index_range)
    i, interface = @index(Global, NTuple)

    # Copy solution data from the local element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the index of the primary side
    # will always run forwards.
    local_element = local_neighbor_ids[interface]
    local_indices = node_indices[interface]

    i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                             index_range)
    j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                             index_range)

    i_element = delayed_index_2d(i_element_start, i_element_step, i)
    j_element = delayed_index_2d(j_element_start, j_element_step, i)

    u_local = get_node_vars(u, equations, dg, i_element, j_element, local_element)
    offset = mpi_buffer_node_offset(mpi_interfaces_buffer_offsets, equations,
                                    i, interface)
    for v in eachvariable(equations)
        mpi_send_buffer[offset + v] = u_local[v]
    end
end

# Offset of the data of node `i` of an MPI interface in the MPI send/receive buffer
Base.@propagate_inbounds function mpi_buffer_node_offset(mpi_interfaces_buffer_offsets,
                                                         equations, i, interface)
    return mpi_interfaces_buffer_offsets[interface] + (i - 1) * nvariables(equations)
end

Base.@propagate_inbounds function get_mpi_buffer_node_vars(mpi_buffer, equations,
                                                           offset)
    # Explicit bounds check, which can be removed by calling this function with `@inbounds`
    @boundscheck checkbounds(mpi_buffer, offset .+ eachvariable(equations))
    # Assume inbounds access now
    return SVector(ntuple(@inline(v->@inbounds mpi_buffer[offset + v]),
                          Val(nvariables(equations))))
end

# The MPI send buffer is filled in `prolong2mpiinterfaces!`, and the kernels have
# finished when this is called, see `rhs_hyperbolic!`.
function start_mpi_send!(backend::Backend, mpi_cache::P4estMPICache,
                         mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2}},
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
function finish_mpi_receive!(backend::Backend, mpi_cache::P4estMPICache,
                             mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2}},
                             equations, dg, cache)
    MPI.Waitall(mpi_cache.mpi_recv_requests)

    return nothing
end

# Compute the surface fluxes at all MPI interfaces. In contrast to the CPU version,
# the local data is taken directly from `u` and the remote data directly from the
# MPI receive buffer.
function calc_mpi_interface_flux!(backend::Backend, surface_flux_values, u,
                                  mesh::Union{P4estMeshParallel{2},
                                              T8codeMeshParallel{2}},
                                  have_nonconservative_terms,
                                  equations, surface_integral, dg::DG, cache)
    nmpiinterfaces(dg, cache) == 0 && return nothing
    @unpack local_neighbor_ids, node_indices, local_sides, normal_directions = cache.mpi_interfaces
    @unpack mpi_recv_buffer, mpi_interfaces_buffer_offsets = cache.mpi_cache
    index_range = eachnode(dg)

    # Explicit bounds check, which allows us to assume inbounds access in the kernel.
    # The offsets are constructed in `init_mpi_buffers` such that the data of all
    # MPI interfaces fits into the MPI buffers.
    @boundscheck begin
        check_axes(u, mesh, equations, dg, cache)
        check_axes(cache.mpi_interfaces, equations, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations, dg, cache)
        check_axes(mpi_interfaces_buffer_offsets, (eachmpiinterface(dg, cache),))
        checkbounds(mpi_recv_buffer,
                    1:(nvariables(equations) * nnodes(dg) * nmpiinterfaces(dg, cache)))
    end

    kernel! = calc_mpi_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, u, mpi_recv_buffer, typeof(mesh),
            have_nonconservative_terms, equations, surface_integral, dg,
            local_neighbor_ids, node_indices, local_sides,
            mpi_interfaces_buffer_offsets, normal_directions, index_range,
            ndrange = (nnodes(dg), nmpiinterfaces(dg, cache)))
    return nothing
end

@kernel inbounds=true function calc_mpi_interface_flux_KAkernel!(surface_flux_values, u,
                                                                 mpi_recv_buffer,
                                                                 MeshT::Type{<:Union{P4estMeshParallel{2},
                                                                                     T8codeMeshParallel{2}}},
                                                                 have_nonconservative_terms,
                                                                 equations,
                                                                 surface_integral,
                                                                 dg,
                                                                 local_neighbor_ids,
                                                                 node_indices,
                                                                 local_sides,
                                                                 mpi_interfaces_buffer_offsets,
                                                                 normal_directions,
                                                                 index_range)
    i, interface = @index(Global, NTuple)

    # Get element and side index information on the local element
    local_element = local_neighbor_ids[interface]
    local_indices = node_indices[interface]
    local_direction = indices2direction(local_indices)
    local_side = local_sides[interface]

    # Create the local i,j indexing on the local element used to pull the local data
    i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                             index_range)
    j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                             index_range)

    i_element = delayed_index_2d(i_element_start, i_element_step, i)
    j_element = delayed_index_2d(j_element_start, j_element_step, i)

    # The surface flux storage must be indexed in alignment with the local element
    # indexing
    if NodeIndex.IBackward in local_indices
        surface_node = last(index_range) + 1 - i
    else
        surface_node = i
    end

    # The interface is "aligned at the primary element", i.e., `u_ll` is the solution
    # on the primary side and `u_rr` the solution on the secondary side
    u_local = get_node_vars(u, equations, dg, i_element, j_element, local_element)
    offset = mpi_buffer_node_offset(mpi_interfaces_buffer_offsets, equations,
                                    i, interface)
    u_remote = get_mpi_buffer_node_vars(mpi_recv_buffer, equations, offset)
    if local_side == 1
        u_ll, u_rr = u_local, u_remote
    else # local_side == 2
        u_ll, u_rr = u_remote, u_local
    end

    # Use the normal direction of the primary element, which is available on both
    # MPI ranks. Thus, both ranks compute the same numerical flux, as in the serial
    # `calc_interface_flux!`, see `exchange_normal_directions!`.
    normal_direction = get_surface_normal(normal_directions, i, interface)

    calc_mpi_interface_flux_per_node!(surface_flux_values, MeshT,
                                      have_nonconservative_terms, equations,
                                      surface_integral, u_ll, u_rr, normal_direction,
                                      local_side, surface_node, local_direction,
                                      local_element)
end

# Inlined version of the interface flux computation for conservation laws
Base.@propagate_inbounds function calc_mpi_interface_flux_per_node!(surface_flux_values,
                                                                    MeshT::Type{<:Union{P4estMeshParallel{2},
                                                                                        T8codeMeshParallel{2}}},
                                                                    have_nonconservative_terms::False,
                                                                    equations,
                                                                    surface_integral,
                                                                    u_ll, u_rr,
                                                                    normal_direction,
                                                                    local_side,
                                                                    surface_node_index,
                                                                    local_direction_index,
                                                                    local_element_index)
    @unpack surface_flux = surface_integral

    # `normal_direction` is the normal direction of the primary element.
    # The flux entering the primary element leaves the secondary element
    # (note the `-` for the `else` case).
    if local_side == 1
        flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
    else # local_side == 2
        flux_ = -surface_flux(u_ll, u_rr, normal_direction, equations)
    end

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_index, local_direction_index, local_element_index] = flux_[v]
    end

    return nothing
end

# Inlined version of the interface flux computation for non-conservative equations.
# As for the other surface fluxes on GPUs, see `prolong2interfaces_and_calc_interface_flux_per_node!`,
# only surface fluxes combining the conservative and nonconservative terms are supported.
Base.@propagate_inbounds function calc_mpi_interface_flux_per_node!(surface_flux_values,
                                                                    MeshT::Type{<:Union{P4estMeshParallel{2},
                                                                                        T8codeMeshParallel{2}}},
                                                                    have_nonconservative_terms::True,
                                                                    equations,
                                                                    surface_integral,
                                                                    u_ll, u_rr,
                                                                    normal_direction,
                                                                    local_side,
                                                                    surface_node_index,
                                                                    local_direction_index,
                                                                    local_element_index)
    calc_mpi_interface_flux_per_node!(surface_flux_values, MeshT,
                                      have_nonconservative_terms,
                                      combine_conservative_and_nonconservative_fluxes(surface_integral.surface_flux,
                                                                                      equations),
                                      equations, surface_integral, u_ll, u_rr,
                                      normal_direction, local_side, surface_node_index,
                                      local_direction_index, local_element_index)
    return nothing
end

Base.@propagate_inbounds function calc_mpi_interface_flux_per_node!(surface_flux_values,
                                                                    MeshT::Type{<:Union{P4estMeshParallel{2},
                                                                                        T8codeMeshParallel{2}}},
                                                                    have_nonconservative_terms::True,
                                                                    combine_conservative_and_nonconservative_fluxes::True,
                                                                    equations,
                                                                    surface_integral,
                                                                    u_ll, u_rr,
                                                                    normal_direction,
                                                                    local_side,
                                                                    surface_node_index,
                                                                    local_direction_index,
                                                                    local_element_index)
    @unpack surface_flux = surface_integral

    # Compute the flux including the non-conservative term for this side of the interface.
    # `normal_direction` is the normal direction of the primary element
    # (note the `-` for the `else` case).
    flux_left, flux_right = surface_flux(u_ll, u_rr, normal_direction, equations)
    if local_side == 1
        flux_ = flux_left
    else # local_side == 2
        flux_ = -flux_right
    end

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_index, local_direction_index, local_element_index] = flux_[v]
    end

    return nothing
end

# For GPU backends MPI mortars are not yet implemented
function prolong2mpimortars!(backend::Backend, cache, u,
                             mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2}},
                             equations, mortar, dg::DG)
    @assert isempty(eachmpimortar(dg, cache)) "MPI mortars are not yet supported on GPU backends"
    return nothing
end

# For GPU backends MPI mortars are not yet implemented
function calc_mpi_mortar_flux!(backend::Backend, surface_flux_values,
                               mesh::Union{P4estMeshParallel{2},
                                           T8codeMeshParallel{2}},
                               have_nonconservative_terms, equations, mortar,
                               surface_integral, dg::DG, cache)
    @assert isempty(eachmpimortar(dg, cache)) "MPI mortars are not yet supported on GPU backends"
    return nothing
end
end # @muladd
