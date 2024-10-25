# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs!(du, u, t,
              mesh::Union{ParallelP4estMesh{3}, ParallelT8codeMesh{3}}, equations,
              boundary_conditions, source_terms::Source,
              dg::DG, cache) where {Source}
    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache, u, mesh, equations, dg.surface_integral, dg)
    end

    # Prolong solution to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars" begin
        prolong2mpimortars!(cache, u, mesh, equations,
                            dg.mortar, dg.surface_integral, dg)
    end

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u, mesh, equations, dg.surface_integral, dg)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg, cache)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u, mesh, equations, dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Prolong solution to mortars
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u, mesh, equations,
                         dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache.elements.surface_flux_values, mesh,
                          have_nonconservative_terms(equations), equations,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(cache.mpi_cache, mesh, equations, dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations), equations,
                                 dg.surface_integral, dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(cache.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations, dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    # Finish to send MPI data
    @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

    return nothing
end

function prolong2mpiinterfaces!(cache, u,
                                mesh::Union{ParallelP4estMesh{3},
                                            ParallelT8codeMesh{3}},
                                equations, surface_integral, dg::DG)
    @unpack mpi_interfaces = cache
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_side = mpi_interfaces.local_sides[interface]
        local_element = mpi_interfaces.local_neighbor_ids[interface]
        local_indices = mpi_interfaces.node_indices[interface]

        i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                     index_range)
        j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                     index_range)
        k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                     index_range)

        i_element = i_element_start
        j_element = j_element_start
        k_element = k_element_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    mpi_interfaces.u[local_side, v, i, j, interface] = u[v, i_element,
                                                                         j_element,
                                                                         k_element,
                                                                         local_element]
                end
                i_element += i_element_step_i
                j_element += j_element_step_i
                k_element += k_element_step_i
            end
            i_element += i_element_step_j
            j_element += j_element_step_j
            k_element += k_element_step_j
        end
    end

    return nothing
end

function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::Union{ParallelP4estMesh{3},
                                              ParallelT8codeMesh{3}},
                                  nonconservative_terms,
                                  equations, surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get element and side index information on the local element
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        # Create the local i,j,k indexing on the local element used to pull normal direction information
        i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                     index_range)
        j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                     index_range)
        k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                     index_range)

        i_element = i_element_start
        j_element = j_element_start
        k_element = k_element_start

        # Initiate the node indices to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        local_surface_indices = surface_indices(local_indices)
        i_surface_start, i_surface_step_i, i_surface_step_j = index_to_start_step_3d(local_surface_indices[1],
                                                                                     index_range)
        j_surface_start, j_surface_step_i, j_surface_step_j = index_to_start_step_3d(local_surface_indices[2],
                                                                                     index_range)
        i_surface = i_surface_start
        j_surface = j_surface_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                # Get the normal direction on the local element
                # Contravariant vectors at interfaces in negative coordinate direction
                # are pointing inwards. This is handled by `get_normal_direction`.
                normal_direction = get_normal_direction(local_direction,
                                                        contravariant_vectors,
                                                        i_element, j_element, k_element,
                                                        local_element)

                calc_mpi_interface_flux!(surface_flux_values, mesh,
                                         nonconservative_terms, equations,
                                         surface_integral, dg, cache,
                                         interface, normal_direction,
                                         i, j, local_side,
                                         i_surface, j_surface, local_direction,
                                         local_element)

                # Increment local element indices to pull the normal direction
                i_element += i_element_step_i
                j_element += j_element_step_i
                k_element += k_element_step_i
                # Increment the surface node indices along the local element
                i_surface += i_surface_step_i
                j_surface += j_surface_step_i
            end
            # Increment local element indices to pull the normal direction
            i_element += i_element_step_j
            j_element += j_element_step_j
            k_element += k_element_step_j
            # Increment the surface node indices along the local element
            i_surface += i_surface_step_j
            j_surface += j_surface_step_j
        end
    end

    return nothing
end

# Inlined version of the interface flux computation for conservation laws
@inline function calc_mpi_interface_flux!(surface_flux_values,
                                          mesh::Union{ParallelP4estMesh{3},
                                                      ParallelT8codeMesh{3}},
                                          nonconservative_terms::False, equations,
                                          surface_integral, dg::DG, cache,
                                          interface_index, normal_direction,
                                          interface_i_node_index,
                                          interface_j_node_index, local_side,
                                          surface_i_node_index, surface_j_node_index,
                                          local_direction_index, local_element_index)
    @unpack u = cache.mpi_interfaces
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg,
                                       interface_i_node_index, interface_j_node_index,
                                       interface_index)

    if local_side == 1
        flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
    else # local_side == 2
        flux_ = -surface_flux(u_ll, u_rr, -normal_direction, equations)
    end

    for v in eachvariable(equations)
        surface_flux_values[v, surface_i_node_index, surface_j_node_index,
        local_direction_index, local_element_index] = flux_[v]
    end
end

function prolong2mpimortars!(cache, u,
                             mesh::Union{ParallelP4estMesh{3}, ParallelT8codeMesh{3}},
                             equations,
                             mortar_l2::LobattoLegendreMortarL2,
                             surface_integral, dg::DGSEM)
    @unpack node_indices = cache.mpi_mortars
    index_range = eachnode(dg)

    @threaded for mortar in eachmpimortar(dg, cache)
        local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

        # Get start value and step size for indices on both sides to get the correct face
        # and orientation
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 5 # -> large element
                # Buffer to copy solution values of the large element in the correct orientation
                # before interpolating
                u_buffer = cache.u_threaded[Threads.threadid()]
                # temporary buffer for projections
                fstar_tmp = cache.fstar_tmp_threaded[Threads.threadid()]

                i_large = i_large_start
                j_large = j_large_start
                k_large = k_large_start
                for j in eachnode(dg)
                    for i in eachnode(dg)
                        for v in eachvariable(equations)
                            u_buffer[v, i, j] = u[v, i_large, j_large, k_large, element]
                        end

                        i_large += i_large_step_i
                        j_large += j_large_step_i
                        k_large += k_large_step_i
                    end
                    i_large += i_large_step_j
                    j_large += j_large_step_j
                    k_large += k_large_step_j
                end

                # Interpolate large element face data from buffer to small face locations
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 1, :, :,
                                             mortar),
                                        mortar_l2.forward_lower,
                                        mortar_l2.forward_lower,
                                        u_buffer,
                                        fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 2, :, :,
                                             mortar),
                                        mortar_l2.forward_upper,
                                        mortar_l2.forward_lower,
                                        u_buffer,
                                        fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 3, :, :,
                                             mortar),
                                        mortar_l2.forward_lower,
                                        mortar_l2.forward_upper,
                                        u_buffer,
                                        fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 4, :, :,
                                             mortar),
                                        mortar_l2.forward_upper,
                                        mortar_l2.forward_upper,
                                        u_buffer,
                                        fstar_tmp)
            else # position in (1, 2, 3, 4) -> small element
                # Copy solution data from the small elements
                i_small = i_small_start
                j_small = j_small_start
                k_small = k_small_start
                for j in eachnode(dg)
                    for i in eachnode(dg)
                        for v in eachvariable(equations)
                            cache.mpi_mortars.u[1, v, position, i, j, mortar] = u[v,
                                                                                  i_small,
                                                                                  j_small,
                                                                                  k_small,
                                                                                  element]
                        end
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

    return nothing
end

function calc_mpi_mortar_flux!(surface_flux_values,
                               mesh::Union{ParallelP4estMesh{3}, ParallelT8codeMesh{3}},
                               nonconservative_terms, equations,
                               mortar_l2::LobattoLegendreMortarL2,
                               surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_threaded, fstar_tmp_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmpimortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = fstar_threaded[Threads.threadid()]
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]

        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        for position in 1:4
            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    # Get the normal direction on the small element.
                    normal_direction = get_normal_direction(cache.mpi_mortars, i, j,
                                                            position, mortar)

                    calc_mpi_mortar_flux!(fstar, mesh, nonconservative_terms, equations,
                                          surface_integral, dg, cache,
                                          mortar, position, normal_direction,
                                          i, j)

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
            end
            i_small += i_small_step_j
            j_small += j_small_step_j
            k_small += k_small_step_j
        end

        # Buffer to interpolate flux values of the large element to before
        # copying in the correct orientation
        u_buffer = cache.u_threaded[Threads.threadid()]

        mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                       mesh, equations, mortar_l2, dg, cache,
                                       mortar, fstar, u_buffer, fstar_tmp)
    end

    return nothing
end

# Inlined version of the mortar flux computation on small elements for conservation laws
@inline function calc_mpi_mortar_flux!(fstar,
                                       mesh::Union{ParallelP4estMesh{3},
                                                   ParallelT8codeMesh{3}},
                                       nonconservative_terms::False, equations,
                                       surface_integral, dg::DG, cache,
                                       mortar_index, position_index, normal_direction,
                                       i_node_index, j_node_index)
    @unpack u = cache.mpi_mortars
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, i_node_index,
                                       j_node_index, mortar_index)

    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Copy flux to buffer
    set_node_vars!(fstar, flux, equations, dg, i_node_index, j_node_index,
                   position_index)
end

@inline function mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                                mesh::Union{ParallelP4estMesh{3},
                                                            ParallelT8codeMesh{3}},
                                                equations,
                                                mortar_l2::LobattoLegendreMortarL2,
                                                dg::DGSEM, cache, mortar, fstar,
                                                u_buffer, fstar_tmp)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    index_range = eachnode(dg)

    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)
    large_surface_indices = surface_indices(large_indices)

    i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_surface_indices[1],
                                                                           index_range)
    j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_surface_indices[2],
                                                                           index_range)

    for (element, position) in zip(local_neighbor_ids[mortar],
                                   local_neighbor_positions[mortar])
        if position == 5 # -> large element
            # Project small fluxes to large element.
            multiply_dimensionwise!(u_buffer,
                                    mortar_l2.reverse_lower, mortar_l2.reverse_lower,
                                    view(fstar, .., 1),
                                    fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_lower,
                                        view(fstar, .., 2),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_lower,
                                        mortar_l2.reverse_upper,
                                        view(fstar, .., 3),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_upper,
                                        view(fstar, .., 4),
                                        fstar_tmp)
            # The flux is calculated in the outward direction of the small elements,
            # so the sign must be switched to get the flux in outward direction
            # of the large element.
            # The contravariant vectors of the large element (and therefore the normal
            # vectors of the large element as well) are four times as large as the
            # contravariant vectors of the small elements. Therefore, the flux needs
            # to be scaled by a factor of 4 to obtain the flux of the large element.
            u_buffer .*= -4
            # Copy interpolated flux values from buffer to large element face in the
            # correct orientation.
            # Note that the index of the small sides will always run forward but
            # the index of the large side might need to run backwards for flipped sides.
            i_large = i_large_start
            j_large = j_large_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i_large, j_large, large_direction, element] = u_buffer[v,
                                                                                                      i,
                                                                                                      j]
                    end
                    i_large += i_large_step_i
                    j_large += j_large_step_i
                end
                i_large += i_large_step_j
                j_large += j_large_step_j
            end
        else # position in (1, 2, 3, 4) -> small element
            # Copy solution small to small
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i, j, small_direction, element] = fstar[v,
                                                                                       i,
                                                                                       j,
                                                                                       position]
                    end
                end
            end
        end
    end

    return nothing
end
end # muladd
