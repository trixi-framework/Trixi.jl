# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    function rhs_parabolic!(du, u, t,
                            mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2}},
                            equations_parabolic::AbstractEquationsParabolic,
                            boundary_conditions_parabolic, source_terms_parabolic,
                            dg::DG, parabolic_scheme, cache, cache_parabolic)
        @unpack viscous_container = cache_parabolic
        @unpack u_transformed, gradients, flux_viscous = viscous_container

        # Stage 0: local variable transform
        #
        @trixi_timeit timer() "transform variables" begin
            transform_variables!(u_transformed, u, mesh, equations_parabolic,
                                 dg, cache)
        end

        #
        # Stage 1: gradient computation
        #

        # Start gradient-stage MPI receive
        @trixi_timeit timer() "start MPI receive gradient" begin
            start_mpi_receive!(cache.mpi_cache)
        end

        # Prolong transformed variables to MPI mortars
        # @trixi_timeit timer() "prolong2mpimortars gradient" begin
        #     prolong2mpimortars!(cache, u_transformed, mesh, equations_parabolic,
        #                         dg.mortar, dg)
        # end

        # Prolong transformed variables to MPI interfaces
        @trixi_timeit timer() "prolong2mpiinterfaces gradient" begin
            prolong2mpiinterfaces!(cache, u_transformed, mesh, equations_parabolic,
                                   dg.surface_integral, dg)
        end

        # Start gradient-stage MPI send
        @trixi_timeit timer() "start MPI send gradient" begin
            start_mpi_send!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
        end

        # Local gradient computation
        @trixi_timeit timer() "calculate gradient local" begin
            calc_gradient_local!(gradients, u_transformed, t, mesh,
                                 equations_parabolic, boundary_conditions_parabolic,
                                 dg, parabolic_scheme, cache)
        end

        # Finish gradient-stage MPI receive
        @trixi_timeit timer() "finish MPI receive gradient" begin
            finish_mpi_receive!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
        end
        # Finish gradient-stage MPI send
        @trixi_timeit timer() "finish MPI send gradient" begin
            finish_mpi_send!(cache.mpi_cache)
        end
        # MPI interface fluxes for gradient stage
        @trixi_timeit timer() "MPI interface flux gradient" begin
            calc_mpi_interface_flux_gradient!(cache.elements.surface_flux_values,
                                              mesh, equations_parabolic,
                                              dg, parabolic_scheme, cache)
        end

        # MPI mortar fluxes for gradient stage
        # @trixi_timeit timer() "MPI mortar flux gradient" begin
        #     calc_mpi_mortar_flux_gradient!(cache.elements.surface_flux_values,
        #                                    mesh, equations_parabolic, dg.mortar,
        #                                    dg, parabolic_scheme, cache)
        # end

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
        # Stage 2: local viscous flux construction
        #
        @trixi_timeit timer() "calculate viscous fluxes" begin
            calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                                 equations_parabolic, dg, cache)
        end

        #
        # Stage 3: divergence of viscous fluxes
        #

        # Start divergence-stage MPI receive
        @trixi_timeit timer() "start MPI receive divergence" begin
            start_mpi_receive!(cache.mpi_cache)
        end

        # Reset du
        @trixi_timeit timer() "reset ∂u/∂t" begin
            set_zero!(du, dg, cache)
        end

        # Local volume integral
        @trixi_timeit timer() "volume integral" begin
            calc_volume_integral!(du, flux_viscous, mesh, equations_parabolic, dg, cache)
        end

        # Prolong viscous fluxes to MPI mortars
        # @trixi_timeit timer() "prolong2mpimortars divergence" begin
        #     prolong2mpimortars_divergence!(cache, flux_viscous, mesh, equations_parabolic,
        #                                    dg.mortar, dg)
        # end

        # Prolong viscous fluxes to MPI interfaces
        @trixi_timeit timer() "prolong2mpiinterfaces divergence" begin
            prolong2mpiinterfaces!(cache, flux_viscous, mesh, equations_parabolic, dg)
        end
        ########################## Divergence #################################
        # Start divergence-stage MPI send
        @trixi_timeit timer() "start MPI send divergence" begin
            start_mpi_send!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
        end

        # Local interface fluxes
        @trixi_timeit timer() "prolong2interfaces" begin
            prolong2interfaces!(cache, flux_viscous, mesh, equations_parabolic, dg)
        end

        @trixi_timeit timer() "interface flux" begin
            calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                                 equations_parabolic, dg, parabolic_scheme, cache)
        end

        # Local boundary fluxes
        @trixi_timeit timer() "prolong2boundaries" begin
            prolong2boundaries!(cache, flux_viscous, mesh, equations_parabolic, dg)
        end

        @trixi_timeit timer() "boundary flux" begin
            calc_boundary_flux_divergence!(cache, t,
                                           boundary_conditions_parabolic, mesh,
                                           equations_parabolic,
                                           dg.surface_integral, dg)
        end

        # Local mortar fluxes
        @trixi_timeit timer() "prolong2mortars" begin
            prolong2mortars_divergence!(cache, flux_viscous, mesh, equations_parabolic,
                                        dg.mortar, dg)
        end

        @trixi_timeit timer() "mortar flux" begin
            calc_mortar_flux_divergence!(cache.elements.surface_flux_values,
                                         mesh, equations_parabolic, dg.mortar,
                                         dg, parabolic_scheme, cache)
        end

        # Finish divergence-stage MPI receive
        @trixi_timeit timer() "finish MPI receive divergence" begin
            finish_mpi_receive!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
        end

        # MPI interface fluxes for divergence stage
        @trixi_timeit timer() "MPI interface flux divergence" begin
            calc_mpi_interface_flux_divergence!(cache.elements.surface_flux_values,
                                                mesh, equations_parabolic,
                                                dg, parabolic_scheme, cache)
        end

        # MPI mortar fluxes for divergence stage
        # @trixi_timeit timer() "MPI mortar flux divergence" begin
        #     calc_mpi_mortar_flux_divergence!(cache.elements.surface_flux_values,
        #                                      mesh, equations_parabolic, dg.mortar,
        #                                      dg, parabolic_scheme, cache)
        # end

        # Finish divergence-stage MPI send
        @trixi_timeit timer() "finish MPI send divergence" begin
            finish_mpi_send!(cache.mpi_cache)
        end

        #
        # Stage 4: final assembly
        #
        @trixi_timeit timer() "surface integral" begin
            calc_surface_integral!(du, u, mesh, equations_parabolic,
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
                                  mesh::Union{P4estMeshParallel{2}, T8codeMeshParallel{2}},
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
            prolong2interfaces!(cache, u_transformed, mesh,
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

    function prolong2mpiinterfaces!(cache, flux_viscous::Tuple,
                                    mesh::Union{P4estMeshParallel{2},
                                                T8codeMeshParallel{2}},
                                    equations_parabolic, dg::DG)
        @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
        @unpack contravariant_vectors = cache.elements
        index_range = eachnode(dg)

        flux_viscous_x, flux_viscous_y = flux_viscous

        @threaded for interface in eachmpiinterface(dg, cache)
            local_element = local_neighbor_ids[interface]
            local_indices = node_indices[interface]
            local_direction = indices2direction(local_indices)
            local_side = local_sides[interface]
            orientationFactor = local_side == 1 ? 1 : -1

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
                    flux_visc = SVector(flux_viscous_x[v, i_elem, j_elem, local_element],
                                        flux_viscous_y[v, i_elem, j_elem, local_element])

                    cache.mpi_interfaces.u[local_side, v, i, interface] = orientationFactor *
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
                                               mesh::Union{P4estMeshParallel{2},
                                                           T8codeMeshParallel{2}},
                                               equations_parabolic,
                                               dg::DG, parabolic_scheme, cache)
        @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
        @unpack contravariant_vectors = cache.elements
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

                calc_mpi_interface_flux_gradient!(surface_flux_values, mesh,
                                                  equations_parabolic,
                                                  dg, parabolic_scheme, cache,
                                                  interface, normal_direction,
                                                  i, local_side,
                                                  surface_node,
                                                  local_direction, local_element)

                # Increment local element indices to pull the normal direction
                i_element += i_element_step
                j_element += j_element_step

                # Increment the surface node index along the local element
                surface_node += surface_node_step
            end
        end

        return nothing
    end

    @inline function calc_mpi_interface_flux_gradient!(surface_flux_values,
                                                       mesh::Union{P4estMeshParallel{2},
                                                                   T8codeMeshParallel{2}},
                                                       equations_parabolic,
                                                       dg::DG, parabolic_scheme, cache,
                                                       interface_index, normal_direction,
                                                       interface_node_index, local_side,
                                                       surface_node_index,
                                                       local_direction_index,
                                                       local_element_index)
        @unpack u = cache.mpi_interfaces

        u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg,
                                           interface_node_index,
                                           interface_index)

        flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                               equations_parabolic, parabolic_scheme)

        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, surface_node_index,
            local_direction_index, local_element_index] = flux_[v]
        end

        return nothing
    end

    function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                                 mesh::Union{P4estMeshParallel{2},
                                                             T8codeMeshParallel{2}},
                                                 equations_parabolic,
                                                 dg::DG, parabolic_scheme, cache)
        @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
        @unpack contravariant_vectors = cache.elements
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

                calc_mpi_interface_flux_divergence!(surface_flux_values, mesh,
                                                    equations_parabolic,
                                                    dg, parabolic_scheme, cache,
                                                    interface, normal_direction,
                                                    i, local_side,
                                                    surface_node,
                                                    local_direction, local_element)

                i_element += i_element_step
                j_element += j_element_step

                surface_node += surface_node_step
            end
        end

        return nothing
    end

    @inline function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                                         mesh::Union{P4estMeshParallel{2},
                                                                     T8codeMeshParallel{2}},
                                                         equations_parabolic,
                                                         dg::DG, parabolic_scheme, cache,
                                                         interface_index, normal_direction,
                                                         interface_node_index, local_side,
                                                         surface_node_index,
                                                         local_direction_index,
                                                         local_element_index)
        @unpack u = cache.mpi_interfaces

        viscous_flux_normal_ll, viscous_flux_normal_rr = get_surface_node_vars(u,
                                                                               equations_parabolic,
                                                                               dg,
                                                                               interface_node_index,
                                                                               interface_index)

        flux_ = flux_parabolic(viscous_flux_normal_ll, viscous_flux_normal_rr,
                               normal_direction, Divergence(),
                               equations_parabolic, parabolic_scheme)

        if local_side == 1
            flux_ = flux_parabolic(viscous_flux_normal_ll, viscous_flux_normal_rr,
                                   normal_direction, Divergence(),
                                   equations_parabolic, parabolic_scheme)
            sign_ = 1
        else
            # side 2 must also use primary-oriented normal
            flux_ = flux_parabolic(viscous_flux_normal_ll, viscous_flux_normal_rr,
                                   -normal_direction, Divergence(),
                                   equations_parabolic, parabolic_scheme)
            sign_ = -1
        end

        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, surface_node_index,
            local_direction_index, local_element_index] = sign_ * flux_[v]
        end

        return nothing
    end
end #muladd
