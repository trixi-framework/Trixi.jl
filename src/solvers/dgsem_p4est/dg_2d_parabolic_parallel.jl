# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent    
function rhs_parabolic!(du, u, t,
                        mesh::Union{P4estMeshParallel{2}, P4estMeshParallel{3}},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms_parabolic,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack parabolic_container = cache_parabolic
    @unpack u_transformed, gradients, flux_parabolic = parabolic_container

    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, cache)
    end

    ### Gradient computation ###

    # Start gradient MPI receive
    @trixi_timeit timer() "start MPI receive gradient" begin
        start_mpi_receive!(cache.mpi_cache)
    end

    # Prolong transformed variables to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars gradient" begin
        prolong2mpimortars!(cache, u_transformed, mesh, equations_parabolic,
                            dg.mortar, dg)
    end

    # Prolong transformed variables to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces gradient" begin
        prolong2mpiinterfaces!(cache, u_transformed, mesh, equations_parabolic,
                               dg.surface_integral, dg)
    end

    # Start gradient MPI send
    @trixi_timeit timer() "start MPI send gradient" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
    end

    # Local gradient computation
    @trixi_timeit timer() "calculate gradient local" begin
        calc_gradient_local!(gradients, u_transformed, t, mesh,
                             equations_parabolic, boundary_conditions_parabolic,
                             dg, parabolic_scheme, cache)
    end

    # Finish gradient MPI receive
    @trixi_timeit timer() "finish MPI receive gradient" begin
        finish_mpi_receive!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
    end

    # MPI interface fluxes for gradients
    @trixi_timeit timer() "MPI interface flux gradient" begin
        calc_mpi_interface_flux_gradient!(cache.elements.surface_flux_values,
                                          mesh, equations_parabolic,
                                          dg, parabolic_scheme, cache)
    end

    # MPI mortar fluxes for gradients
    @trixi_timeit timer() "MPI mortar flux gradient" begin
        calc_mpi_mortar_flux_gradient!(cache.elements.surface_flux_values,
                                       mesh, equations_parabolic, dg.mortar,
                                       dg, parabolic_scheme, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral_gradient!(gradients, mesh, equations_parabolic,
                                        dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients, mesh, equations_parabolic, dg,
                                  cache)
    end

    # Finish gradient MPI send
    @trixi_timeit timer() "finish MPI send gradient" begin
        finish_mpi_send!(cache.mpi_cache)
    end

    # Local parabolic flux construction
    @trixi_timeit timer() "calculate parabolic fluxes" begin
        calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                               equations_parabolic, dg, cache)
    end

    ### Divergence of parabolic/gradient fluxes ###

    # Start divergence MPI receive
    @trixi_timeit timer() "start MPI receive divergence" begin
        start_mpi_receive!(cache.mpi_cache)
    end

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" begin
        set_zero!(du, dg, cache)
    end

    # Local volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_parabolic, mesh, equations_parabolic, dg, cache)
    end

    # Prolong parabolic fluxes to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars divergence" begin
        prolong2mpimortars_divergence!(cache, flux_parabolic, mesh, equations_parabolic,
                                       dg.mortar, dg)
    end

    # Prolong parabolic fluxes to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces divergence" begin
        prolong2mpiinterfaces!(cache, flux_parabolic, mesh, equations_parabolic, dg)
    end

    # Start divergence MPI send
    @trixi_timeit timer() "start MPI send divergence" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
    end

    # Local interface fluxes
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, flux_parabolic, mesh, equations_parabolic, dg)
    end

    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             equations_parabolic, dg, parabolic_scheme, cache)
    end

    # Local boundary fluxes
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, flux_parabolic, mesh, equations_parabolic, dg)
    end

    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache, t,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic,
                                       dg.surface_integral, dg)
    end

    # Local mortar fluxes
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars_divergence!(cache, flux_parabolic, mesh, equations_parabolic,
                                    dg.mortar, dg)
    end

    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux_divergence!(cache.elements.surface_flux_values,
                                     mesh, equations_parabolic, dg.mortar,
                                     dg, parabolic_scheme, cache)
    end

    # Finish divergence MPI receive
    @trixi_timeit timer() "finish MPI receive divergence" begin
        finish_mpi_receive!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
    end

    # MPI interface fluxes for divergence
    @trixi_timeit timer() "MPI interface flux divergence" begin
        calc_mpi_interface_flux_divergence!(cache.elements.surface_flux_values,
                                            mesh, equations_parabolic,
                                            dg, parabolic_scheme, cache)
    end

    # MPI mortar fluxes for divergence
    @trixi_timeit timer() "MPI mortar flux divergence" begin
        calc_mpi_mortar_flux_divergence!(cache.elements.surface_flux_values,
                                         mesh, equations_parabolic, dg.mortar,
                                         dg, parabolic_scheme, cache)
    end

    # Finish divergence MPI send
    @trixi_timeit timer() "finish MPI send divergence" begin
        finish_mpi_send!(cache.mpi_cache)
    end

    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(nothing, du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache)
    end

    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    end

    @trixi_timeit timer() "source terms parabolic" begin
        calc_sources_parabolic!(du, u, gradients, t, source_terms_parabolic,
                                equations_parabolic, dg, cache)
    end

    return nothing
end

function calc_gradient_local!(gradients, u_transformed, t,
                              mesh::Union{P4estMeshParallel{2}, P4estMeshParallel{3}},
                              equations_parabolic, boundary_conditions_parabolic,
                              dg::DG, parabolic_scheme, cache)
    # Reset gradients
    @trixi_timeit timer() "reset gradients" begin
        reset_gradients!(gradients, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral_gradient!(gradients, u_transformed,
                                       mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces.
    # This reuses `prolong2interfaces` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(nothing, cache, u_transformed, mesh,
                            equations_parabolic, dg)
    end

    # Calculate interface fluxes for the gradients
    @trixi_timeit timer() "interface flux" begin
        @unpack surface_flux_values = cache.elements
        calc_interface_flux_gradient!(surface_flux_values, mesh, equations_parabolic,
                                      dg, parabolic_scheme, cache)
    end

    # Prolong solution to boundaries.
    # This reuses `prolong2boundaries` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache, u_transformed, mesh,
                            equations_parabolic, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_gradient!(cache, t, boundary_conditions_parabolic,
                                     mesh, equations_parabolic, dg.surface_integral,
                                     dg)
    end

    # Prolong solution to mortars.
    # This reuses `prolong2mortars` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u_transformed, mesh, equations_parabolic,
                         dg.mortar, dg)
    end

    # Calculate mortar fluxes
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux_gradient!(cache.elements.surface_flux_values,
                                   mesh, equations_parabolic, dg.mortar,
                                   dg, parabolic_scheme, cache)
    end

    return nothing
end

function prolong2mpiinterfaces!(cache, flux_parabolic::Tuple,
                                mesh::P4estMeshParallel{2},
                                equations_parabolic, dg::DG)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_parabolic_x, flux_parabolic_y = flux_parabolic

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]
        # store the normal flux with respect to the primary normal direction, 
        # which is the negative of the secondary normal direction
        orientation_factor = local_side == 1 ? 1 : -1

        i_start, i_step = index_to_start_step_2d(local_indices[1], index_range)
        j_start, j_step = index_to_start_step_2d(local_indices[2], index_range)

        i_elem = i_start
        j_elem = j_start

        for i in eachnode(dg)
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_elem, j_elem,
                                                    local_element)

            for v in eachvariable(equations_parabolic)
                flux_visc = SVector(flux_parabolic_x[v, i_elem, j_elem, local_element],
                                    flux_parabolic_y[v, i_elem, j_elem, local_element])
                # Side 1 and 2 must be consistent, i.e., with their outward-pointing normals.
                # Thus, the `orientation_factor` changes the logic such that the
                # flux which enters side 1 leaves side 2. 
                cache.mpi_interfaces.u[local_side, v, i, interface] = orientation_factor *
                                                                      dot(flux_visc,
                                                                          normal_direction)
            end

            i_elem += i_step
            j_elem += j_step
        end
    end

    return nothing
end

function calc_mpi_interface_flux_gradient!(surface_flux_values,
                                           mesh::P4estMeshParallel{2},
                                           equations_parabolic,
                                           dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        # Create the local i,j indexing on the local element used to pull normal direction information
        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start

        # Initiate the node index to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        if :i_backward in local_indices
            surface_node = index_end
            surface_node_step = -1
        else
            surface_node = 1
            surface_node_step = 1
        end

        for i in eachnode(dg)
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_element, j_element,
                                                    local_element)

            u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg,
                                               i, interface)

            flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                                   equations_parabolic, parabolic_scheme)

            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, surface_node,
                local_direction, local_element] = flux_[v]
            end

            # Increment local element indices to pull the normal direction
            # from the element data
            i_element += i_element_step
            j_element += j_element_step

            # Increment the surface node index along the local element
            surface_node += surface_node_step
        end
    end

    return nothing
end

function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                             mesh::P4estMeshParallel{2},
                                             equations_parabolic,
                                             dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start

        # Initiate the node index to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        if :i_backward in local_indices
            surface_node = index_end
            surface_node_step = -1
        else
            surface_node = 1
            surface_node_step = 1
        end

        for i in eachnode(dg)
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_element, j_element,
                                                    local_element)

            parabolic_flux_normal_ll, parabolic_flux_normal_rr = get_surface_node_vars(u,
                                                                                       equations_parabolic,
                                                                                       dg,
                                                                                       i,
                                                                                       interface)

            # Sign flip for `local_side = 2` required for divergence calculation since
            # the divergence interface flux involves the normal direction.
            # `local_side=2` is thus flipped (opposite of primary side)
            orientation_factor = local_side == 1 ? 1 : -1
            flux_ = flux_parabolic(parabolic_flux_normal_ll, parabolic_flux_normal_rr,
                                   orientation_factor * normal_direction, Divergence(),
                                   equations_parabolic, parabolic_scheme)

            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, surface_node,
                local_direction, local_element] = orientation_factor * flux_[v]
            end

            i_element += i_element_step
            j_element += j_element_step

            surface_node += surface_node_step
        end
    end

    return nothing
end

function calc_mpi_mortar_flux_gradient!(surface_flux_values,
                                        mesh::Union{P4estMeshParallel{2},
                                                    T8codeMeshParallel{2}},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DG, parabolic_scheme, cache)
    @unpack (fstar_primary_upper_threaded, fstar_primary_lower_threaded,
    fstar_secondary_upper_threaded, fstar_secondary_lower_threaded) = cache
    @unpack u = cache.mpi_mortars
    @threaded for mortar in eachmpimortar(dg, cache)
        fstar_primary = (fstar_primary_lower_threaded[Threads.threadid()],
                         fstar_primary_upper_threaded[Threads.threadid()])

        fstar_secondary = (fstar_secondary_lower_threaded[Threads.threadid()],
                           fstar_secondary_upper_threaded[Threads.threadid()])

        for position in 1:2
            for i in eachnode(dg)
                normal_direction = get_normal_direction(cache.mpi_mortars, i,
                                                        position, mortar)

                u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg,
                                                   position, i, mortar)

                flux = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                                      equations_parabolic, parabolic_scheme)

                set_node_vars!(fstar_primary[position], flux, equations_parabolic, dg,
                               i)
                set_node_vars!(fstar_secondary[position], flux, equations_parabolic, dg,
                               i)
            end
        end

        u_buffer = cache.u_threaded[Threads.threadid()]
        mpi_mortar_fluxes_to_elements_gradient!(surface_flux_values,
                                                mesh, equations_parabolic, mortar_l2,
                                                dg, cache,
                                                mortar, fstar_primary, fstar_secondary,
                                                u_buffer)
    end

    return nothing
end

@inline function mpi_mortar_fluxes_to_elements_gradient!(surface_flux_values,
                                                         mesh::Union{P4estMeshParallel{2},
                                                                     T8codeMeshParallel{2}},
                                                         equations_parabolic,
                                                         mortar_l2::LobattoLegendreMortarL2,
                                                         dg::DGSEM, cache, mortar,
                                                         fstar_primary, fstar_secondary,
                                                         u_buffer)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    index_range = eachnode(dg)
    index_end = last(index_range)

    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)

    for (element, position) in zip(local_neighbor_ids[mortar],
                                   local_neighbor_positions[mortar])

        # In 2D the large side is the third mortar neighbor slot.
        if position == 3
            # Project the two small-side traces to the large side.
            multiply_dimensionwise!(u_buffer,
                                    mortar_l2.reverse_upper,
                                    fstar_secondary[2],
                                    mortar_l2.reverse_lower,
                                    fstar_secondary[1])

            # Gradient stage: no extra sign flip / scale factor
            # (same as local 2D parabolic mortar_fluxes_to_elements!)
            if :i_backward in large_indices
                for i in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, index_end + 1 - i,
                        large_direction, element] = u_buffer[v, i]
                    end
                end
            else
                for i in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, i,
                        large_direction, element] = u_buffer[v, i]
                    end
                end
            end
        else
            # Small sides copy directly
            for i in eachnode(dg)
                for v in eachvariable(equations_parabolic)
                    surface_flux_values[v, i,
                    small_direction, element] = fstar_primary[position][v, i]
                end
            end
        end
    end

    return nothing
end

function prolong2mpimortars_divergence!(cache, flux_parabolic,
                                        mesh::Union{P4estMeshParallel{2},
                                                    T8codeMeshParallel{2}},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DGSEM)
    @unpack node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_parabolic_x, flux_parabolic_y = flux_parabolic

    @threaded for mortar in eachmpimortar(dg, cache)
        local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

        # Small side indexing
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        # Large side indexing
        large_indices = node_indices[2, mortar]
        large_direction = indices2direction(large_indices)

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 3
                # =========================
                # LARGE ELEMENT
                # =========================
                u_buffer = cache.u_threaded[Threads.threadid()]

                i_large = i_large_start
                j_large = j_large_start

                for i in eachnode(dg)
                    normal_direction = get_normal_direction(large_direction,
                                                            contravariant_vectors,
                                                            i_large, j_large,
                                                            element)

                    for v in eachvariable(equations_parabolic)
                        flux_node = SVector(flux_parabolic_x[v, i_large, j_large,
                                                             element],
                                            flux_parabolic_y[v, i_large, j_large,
                                                             element])

                        # Same convention as local 2D code:
                        # prolong flux dotted with outward normal on the small element.
                        # The large-element normal is -2x the small-element normal,
                        # hence the factor -1/2 here.
                        u_buffer[v, i] = -0.5f0 * dot(flux_node, normal_direction)
                    end

                    i_large += i_large_step
                    j_large += j_large_step
                end

                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 1, :, mortar),
                                        mortar_l2.forward_lower, u_buffer)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 2, :, mortar),
                                        mortar_l2.forward_upper, u_buffer)

            else
                # =========================
                # SMALL ELEMENT (1–2)
                # =========================
                i_small = i_small_start
                j_small = j_small_start

                for i in eachnode(dg)
                    normal_direction = get_normal_direction(small_direction,
                                                            contravariant_vectors,
                                                            i_small, j_small,
                                                            element)

                    for v in eachvariable(equations_parabolic)
                        flux_node = SVector(flux_parabolic_x[v, i_small, j_small,
                                                             element],
                                            flux_parabolic_y[v, i_small, j_small,
                                                             element])

                        cache.mpi_mortars.u[1, v, position, i, mortar] = dot(flux_node,
                                                                             normal_direction)
                    end

                    i_small += i_small_step
                    j_small += j_small_step
                end
            end
        end
    end

    return nothing
end

function calc_mpi_mortar_flux_divergence!(surface_flux_values,
                                          mesh::Union{P4estMeshParallel{2},
                                                      T8codeMeshParallel{2}},
                                          equations_parabolic,
                                          mortar_l2::LobattoLegendreMortarL2,
                                          dg::DG, parabolic_scheme, cache)
    @unpack fstar_primary_upper_threaded, fstar_primary_lower_threaded = cache
    @unpack u = cache.mpi_mortars
    @threaded for mortar in eachmpimortar(dg, cache)
        # Match local 2D structure as one tuple
        fstar = (fstar_primary_lower_threaded[Threads.threadid()],
                 fstar_primary_upper_threaded[Threads.threadid()])

        for position in 1:2
            for i in eachnode(dg)
                normal_direction = get_normal_direction(cache.mpi_mortars, i,
                                                        position, mortar)

                for v in eachvariable(equations_parabolic)
                    viscous_flux_normal_ll = u[1, v, position, i, mortar]
                    viscous_flux_normal_rr = u[2, v, position, i, mortar]

                    flux = flux_parabolic(viscous_flux_normal_ll,
                                          viscous_flux_normal_rr,
                                          normal_direction, Divergence(),
                                          equations_parabolic, parabolic_scheme)

                    # Same convention as local 2D: sign flip / scaling already handled in prolongation
                    fstar[position][v, i] = flux
                end
            end
        end

        u_buffer = cache.u_threaded[Threads.threadid()]

        # Reuse hyperbolic MPI mortar-to-element transfer, same as local 2D 
        mpi_mortar_fluxes_to_elements!(surface_flux_values, mesh,
                                       equations_parabolic, mortar_l2, dg, cache,
                                       mortar, fstar, fstar, u_buffer)
    end

    return nothing
end
end #muladd
