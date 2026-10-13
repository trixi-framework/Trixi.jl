# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prolong2mpiinterfaces!(cache, flux_parabolic::Tuple,
                                mesh::P4estMeshParallel{3},
                                equations_parabolic, dg::DG)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)

    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(flux_parabolic_x, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_y, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_z, mesh, equations_parabolic, dg, cache)
        check_axes(cache.mpi_interfaces, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end

    @threaded for interface in eachmpiinterface(dg, cache)
        @inbounds begin
            # Copy solution data from the local element using "delayed indexing" with
            # a start value and a step size to get the correct face and orientation.
            # Note that in the current implementation, the interface will be
            # "aligned at the primary element", i.e., the index of the primary side
            # will always run forwards.
            local_element = local_neighbor_ids[interface]
            local_indices = node_indices[interface]
            local_direction = indices2direction(local_indices)
            local_side = local_sides[interface]
            # Store the normal flux with respect to the primary normal direction,
            # which is the negative of the secondary normal direction
            orientation_factor = local_side == 1 ? 1 : -1

            i_start, i_step_i, i_step_j = index_to_start_step_3d(local_indices[1],
                                                                 index_range)
            j_start, j_step_i, j_step_j = index_to_start_step_3d(local_indices[2],
                                                                 index_range)
            k_start, k_step_i, k_step_j = index_to_start_step_3d(local_indices[3],
                                                                 index_range)

            i_elem = i_start
            j_elem = j_start
            k_elem = k_start

            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(local_direction,
                                                            contravariant_vectors,
                                                            i_elem, j_elem, k_elem,
                                                            local_element)

                    for v in eachvariable(equations_parabolic)
                        f_parabolic = SVector(flux_parabolic_x[v,
                                                               i_elem,
                                                               j_elem,
                                                               k_elem,
                                                               local_element],
                                              flux_parabolic_y[v,
                                                               i_elem,
                                                               j_elem,
                                                               k_elem,
                                                               local_element],
                                              flux_parabolic_z[v,
                                                               i_elem,
                                                               j_elem,
                                                               k_elem,
                                                               local_element])
                        # Side 1 and 2 must be consistent, i.e., with their
                        # outward-pointing normals. Thus, the `orientation_factor` changes
                        # the logic such that the flux which enters side 1 leaves side 2.
                        flux_normal = dot(f_parabolic, normal_direction)
                        u[local_side, v, i, j, interface] = orientation_factor *
                                                            flux_normal
                    end

                    i_elem += i_step_i
                    j_elem += j_step_i
                    k_elem += k_step_i
                end

                i_elem += i_step_j
                j_elem += j_step_j
                k_elem += k_step_j
            end
        end
    end

    return nothing
end

function calc_mpi_interface_flux_gradient!(surface_flux_values,
                                           mesh::P4estMeshParallel{3},
                                           equations_parabolic,
                                           dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(cache.mpi_interfaces, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations_parabolic,
                                       dg, cache)
    end

    @threaded for interface in eachmpiinterface(dg, cache)
        @inbounds begin
            local_element = local_neighbor_ids[interface]
            local_indices = node_indices[interface]
            local_direction = indices2direction(local_indices)
            local_side = local_sides[interface]
            # The gradient flux must be evaluated with the normal direction of the
            # primary side (as in the serial `calc_interface_flux_gradient!`) since it
            # may depend on the normal direction, e.g., for `ParabolicFormulationLocalDG`.
            # The secondary side's normal direction is the negative of the primary one.
            orientation_factor = local_side == 1 ? 1 : -1

            # Create the local i,j,k indexing on the local element used to pull normal
            # direction information
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
            # the surface flux storage must be indexed in alignment with the local
            # element indexing
            local_surface_indices = surface_indices(local_indices)
            i_surface_start, i_surface_step_i, i_surface_step_j = index_to_start_step_3d(local_surface_indices[1],
                                                                                         index_range)
            j_surface_start, j_surface_step_i, j_surface_step_j = index_to_start_step_3d(local_surface_indices[2],
                                                                                         index_range)
            i_surface = i_surface_start
            j_surface = j_surface_start

            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(local_direction,
                                                            contravariant_vectors,
                                                            i_element,
                                                            j_element,
                                                            k_element,
                                                            local_element)

                    u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg,
                                                       i, j, interface)

                    flux_ = flux_parabolic(u_ll, u_rr,
                                           orientation_factor * normal_direction,
                                           Gradient(),
                                           equations_parabolic, parabolic_scheme)

                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, i_surface, j_surface,
                        local_direction, local_element] = flux_[v]
                    end

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
    end

    return nothing
end

function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                             mesh::P4estMeshParallel{3},
                                             equations_parabolic,
                                             dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(cache.mpi_interfaces, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations_parabolic,
                                       dg, cache)
    end

    @threaded for interface in eachmpiinterface(dg, cache)
        @inbounds begin
            local_element = local_neighbor_ids[interface]
            local_indices = node_indices[interface]
            local_direction = indices2direction(local_indices)
            local_side = local_sides[interface]
            # Sign flip for `local_side = 2` required for divergence calculation since
            # the divergence interface flux involves the normal direction.
            # `local_side=2` is thus flipped (opposite of primary side)
            orientation_factor = local_side == 1 ? 1 : -1

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
            # the surface flux storage must be indexed in alignment with the local
            # element indexing
            local_surface_indices = surface_indices(local_indices)
            i_surface_start, i_surface_step_i, i_surface_step_j = index_to_start_step_3d(local_surface_indices[1],
                                                                                         index_range)
            j_surface_start, j_surface_step_i, j_surface_step_j = index_to_start_step_3d(local_surface_indices[2],
                                                                                         index_range)
            i_surface = i_surface_start
            j_surface = j_surface_start

            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(local_direction,
                                                            contravariant_vectors,
                                                            i_element,
                                                            j_element,
                                                            k_element,
                                                            local_element)

                    (parabolic_flux_normal_ll,
                    parabolic_flux_normal_rr) = get_surface_node_vars(u,
                                                                      equations_parabolic,
                                                                      dg,
                                                                      i, j, interface)

                    flux_ = flux_parabolic(parabolic_flux_normal_ll,
                                           parabolic_flux_normal_rr,
                                           orientation_factor * normal_direction,
                                           Divergence(),
                                           equations_parabolic, parabolic_scheme)

                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, i_surface, j_surface,
                        local_direction, local_element] = orientation_factor * flux_[v]
                    end

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
    end

    return nothing
end

function calc_mpi_mortar_flux_gradient!(surface_flux_values,
                                        mesh::P4estMeshParallel{3},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DG, parabolic_scheme, cache)
    @unpack fstar_primary_threaded, fstar_secondary_threaded, fstar_tmp_threaded = cache

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(cache.mpi_mortars, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations_parabolic,
                                       dg, cache)
    end

    @threaded for mortar in eachmpimortar(dg, cache)
        @inbounds begin
            # Choose thread-specific pre-allocated container
            fstar_primary = fstar_primary_threaded[Threads.threadid()]
            fstar_secondary = fstar_secondary_threaded[Threads.threadid()]
            fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

            for position in 1:4
                for j in eachnode(dg)
                    for i in eachnode(dg)
                        # Use the normal direction of the small element, which is
                        # available on all MPI ranks of this mortar
                        normal_direction = get_normal_direction(cache.mpi_mortars,
                                                                i, j, position, mortar)

                        # This reuses the serial version with the MPI mortar data
                        calc_mortar_flux_gradient!(fstar_primary, fstar_secondary,
                                                   mesh, equations_parabolic,
                                                   dg, parabolic_scheme,
                                                   cache.mpi_mortars.u,
                                                   mortar, position, normal_direction,
                                                   i, j)
                    end
                end
            end

            # Buffer to interpolate flux values of the large element to before
            # copying in the correct orientation
            u_buffer = cache.u_threaded[Threads.threadid()]

            mpi_mortar_fluxes_to_elements_gradient!(surface_flux_values,
                                                    mesh, equations_parabolic,
                                                    mortar_l2, dg, cache,
                                                    mortar,
                                                    fstar_primary, fstar_secondary,
                                                    u_buffer, fstar_tmp)
        end
    end

    return nothing
end

# This version is called during `calc_gradients!` and must be specialized because the
# MAGNITUDE of the flux does NOT depend on the outward normal.
# Thus, you don't need to scale by 4 and flip the sign when storing the mortar fluxes
# back into `surface_flux_values` (as in `mortar_fluxes_to_elements_gradient!`).
Base.@propagate_inbounds function mpi_mortar_fluxes_to_elements_gradient!(surface_flux_values,
                                                                          mesh::P4estMeshParallel{3},
                                                                          equations_parabolic,
                                                                          mortar_l2::LobattoLegendreMortarL2,
                                                                          dg::DGSEM,
                                                                          cache, mortar,
                                                                          fstar_primary,
                                                                          fstar_secondary,
                                                                          u_buffer,
                                                                          fstar_tmp)
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
                                    view(fstar_secondary, .., 1), fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_lower,
                                        view(fstar_secondary, .., 2), fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_lower,
                                        mortar_l2.reverse_upper,
                                        view(fstar_secondary, .., 3), fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_upper,
                                        view(fstar_secondary, .., 4), fstar_tmp)

            # Copy interpolated flux values from buffer to large element face in the
            # correct orientation.
            # Note that the index of the small sides will always run forward but
            # the index of the large side might need to run backwards for flipped sides.
            i_large = i_large_start
            j_large = j_large_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, i_large, j_large,
                        large_direction, element] = u_buffer[v, i, j]
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
                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, i, j,
                        small_direction, element] = fstar_primary[v, i, j, position]
                    end
                end
            end
        end
    end

    return nothing
end

function prolong2mpimortars_divergence!(cache, flux_parabolic,
                                        mesh::P4estMeshParallel{3},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DGSEM)
    @unpack node_indices, u = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(flux_parabolic_x, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_y, mesh, equations_parabolic, dg, cache)
        check_axes(flux_parabolic_z, mesh, equations_parabolic, dg, cache)
        check_axes(cache.mpi_mortars, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end

    @threaded for mortar in eachmpimortar(dg, cache)
        @inbounds begin
            local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
            local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

            # Get start value and step size for indices on both sides to get the correct
            # face and orientation
            small_indices = node_indices[1, mortar]
            small_direction = indices2direction(small_indices)

            i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                                   index_range)
            j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                                   index_range)
            k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                                   index_range)

            large_indices = node_indices[2, mortar]
            large_direction = indices2direction(large_indices)

            i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                                   index_range)
            j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                                   index_range)
            k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                                   index_range)

            for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
                if position == 5 # -> large element
                    # Buffer to copy solution values of the large element in the correct
                    # orientation before interpolating
                    u_buffer = cache.u_threaded[Threads.threadid()]
                    # temporary buffer for projections
                    fstar_tmp = cache.fstar_tmp_threaded[Threads.threadid()]

                    i_large = i_large_start
                    j_large = j_large_start
                    k_large = k_large_start
                    for j in eachnode(dg)
                        for i in eachnode(dg)
                            normal_direction = get_normal_direction(large_direction,
                                                                    contravariant_vectors,
                                                                    i_large, j_large,
                                                                    k_large,
                                                                    element)

                            for v in eachvariable(equations_parabolic)
                                flux_parabolic_node = SVector(flux_parabolic_x[v,
                                                                               i_large,
                                                                               j_large,
                                                                               k_large,
                                                                               element],
                                                              flux_parabolic_y[v,
                                                                               i_large,
                                                                               j_large,
                                                                               k_large,
                                                                               element],
                                                              flux_parabolic_z[v,
                                                                               i_large,
                                                                               j_large,
                                                                               k_large,
                                                                               element])

                                # We prolong the parabolic flux dotted with respect the
                                # outward normal on the small element. We scale by -1/4
                                # here because the normal direction on the large element
                                # is negative 4x that of the small element (these normal
                                # directions are "scaled" by the surface Jacobian)
                                u_buffer[v, i, j] = -0.25f0 *
                                                    dot(flux_parabolic_node,
                                                        normal_direction)
                            end

                            i_large += i_large_step_i
                            j_large += j_large_step_i
                            k_large += k_large_step_i
                        end
                        i_large += i_large_step_j
                        j_large += j_large_step_j
                        k_large += k_large_step_j
                    end

                    # Interpolate large element face data from buffer to small face
                    # locations
                    multiply_dimensionwise!(view(u, 2, :, 1, :, :, mortar),
                                            mortar_l2.forward_lower,
                                            mortar_l2.forward_lower,
                                            u_buffer, fstar_tmp)
                    multiply_dimensionwise!(view(u, 2, :, 2, :, :, mortar),
                                            mortar_l2.forward_upper,
                                            mortar_l2.forward_lower,
                                            u_buffer, fstar_tmp)
                    multiply_dimensionwise!(view(u, 2, :, 3, :, :, mortar),
                                            mortar_l2.forward_lower,
                                            mortar_l2.forward_upper,
                                            u_buffer, fstar_tmp)
                    multiply_dimensionwise!(view(u, 2, :, 4, :, :, mortar),
                                            mortar_l2.forward_upper,
                                            mortar_l2.forward_upper,
                                            u_buffer, fstar_tmp)
                else # position in (1, 2, 3, 4) -> small element
                    # Copy solution data from the small elements
                    i_small = i_small_start
                    j_small = j_small_start
                    k_small = k_small_start
                    for j in eachnode(dg)
                        for i in eachnode(dg)
                            normal_direction = get_normal_direction(small_direction,
                                                                    contravariant_vectors,
                                                                    i_small, j_small,
                                                                    k_small,
                                                                    element)

                            for v in eachvariable(equations_parabolic)
                                flux_parabolic_node = SVector(flux_parabolic_x[v,
                                                                               i_small,
                                                                               j_small,
                                                                               k_small,
                                                                               element],
                                                              flux_parabolic_y[v,
                                                                               i_small,
                                                                               j_small,
                                                                               k_small,
                                                                               element],
                                                              flux_parabolic_z[v,
                                                                               i_small,
                                                                               j_small,
                                                                               k_small,
                                                                               element])

                                u[1, v, position, i, j, mortar] = dot(flux_parabolic_node,
                                                                      normal_direction)
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
    end

    return nothing
end

function calc_mpi_mortar_flux_divergence!(surface_flux_values,
                                          mesh::P4estMeshParallel{3},
                                          equations_parabolic,
                                          mortar_l2::LobattoLegendreMortarL2,
                                          dg::DG, parabolic_scheme, cache)
    @unpack fstar_primary_threaded, fstar_tmp_threaded = cache
    @unpack u = cache.mpi_mortars

    # Explicit bounds check, which allows us to assume inbounds access below
    @boundscheck begin
        check_axes(cache.mpi_mortars, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh, equations_parabolic,
                                       dg, cache)
    end

    @threaded for mortar in eachmpimortar(dg, cache)
        @inbounds begin
            # Choose thread-specific pre-allocated container
            fstar = fstar_primary_threaded[Threads.threadid()]
            fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

            for position in 1:4
                for j in eachnode(dg)
                    for i in eachnode(dg)
                        normal_direction = get_normal_direction(cache.mpi_mortars,
                                                                i, j, position, mortar)

                        for v in eachvariable(equations_parabolic)
                            parabolic_flux_normal_ll = u[1, v, position, i, j, mortar]
                            parabolic_flux_normal_rr = u[2, v, position, i, j, mortar]

                            flux_ = flux_parabolic(parabolic_flux_normal_ll,
                                                   parabolic_flux_normal_rr,
                                                   normal_direction, Divergence(),
                                                   equations_parabolic,
                                                   parabolic_scheme)

                            # Sign flip (and scaling by 0.25) already handled above in
                            # `prolong2mpimortars_divergence!`
                            fstar[v, i, j, position] = flux_
                        end
                    end
                end
            end

            # Buffer to interpolate flux values of the large element to before
            # copying in the correct orientation
            u_buffer = cache.u_threaded[Threads.threadid()]

            # This reuses the hyperbolic version of `mpi_mortar_fluxes_to_elements!`
            mpi_mortar_fluxes_to_elements!(surface_flux_values, mesh,
                                           equations_parabolic, mortar_l2, dg, cache,
                                           mortar, fstar, fstar, u_buffer, fstar_tmp)
        end
    end

    return nothing
end
end # @muladd
