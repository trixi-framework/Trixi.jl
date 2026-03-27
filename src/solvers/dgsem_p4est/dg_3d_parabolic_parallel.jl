# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function calc_mpi_interface_flux_gradient!(surface_flux_values,
                                           mesh::Union{P4estMeshParallel{3},
                                                       T8codeMeshParallel{3}},
                                           equations_parabolic,
                                           dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
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
                normal_direction = get_normal_direction(local_direction,
                                                        contravariant_vectors,
                                                        i_element, j_element, k_element,
                                                        local_element)

                calc_mpi_interface_flux_gradient!(surface_flux_values, mesh,
                                                  equations_parabolic,
                                                  dg, parabolic_scheme, cache,
                                                  interface, normal_direction,
                                                  i, j, local_side,
                                                  i_surface, j_surface,
                                                  local_direction, local_element)
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

@inline function calc_mpi_interface_flux_gradient!(surface_flux_values,
                                                   mesh::Union{P4estMeshParallel{3},
                                                               T8codeMeshParallel{3}},
                                                   equations_parabolic,
                                                   dg::DG, parabolic_scheme, cache,
                                                   interface_index, normal_direction,
                                                   interface_i_node_index,
                                                   interface_j_node_index, local_side,
                                                   surface_i_node_index,
                                                   surface_j_node_index,
                                                   local_direction_index,
                                                   local_element_index)
    @unpack u = cache.mpi_interfaces

    u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg,
                                       interface_i_node_index,
                                       interface_j_node_index,
                                       interface_index)

    flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                           equations_parabolic, parabolic_scheme)

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, surface_i_node_index, surface_j_node_index,
        local_direction_index, local_element_index] = flux_[v]
    end

    return nothing
end

function prolong2mpimortars_divergence!(cache, flux_parabolic,
                                        mesh::Union{P4estMeshParallel{3},
                                                    T8codeMeshParallel{3}},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DGSEM)
    @unpack node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic

    @threaded for mortar in eachmpimortar(dg, cache)
        local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

        # Small side indexing
        small_indices = node_indices[1, mortar]
        direction_index = indices2direction(small_indices)

        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        # Large side indexing
        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 5
                # =========================
                # LARGE ELEMENT
                # =========================
                u_buffer = cache.u_threaded[Threads.threadid()]
                fstar_tmp = cache.fstar_tmp_threaded[Threads.threadid()]

                i_large = i_large_start
                j_large = j_large_start
                k_large = k_large_start

                for j in eachnode(dg)
                    for i in eachnode(dg)
                        normal_direction = get_normal_direction(direction_index,
                                                                contravariant_vectors,
                                                                i_large, j_large,
                                                                k_large,
                                                                element)

                        for v in eachvariable(equations_parabolic)
                            flux_node = SVector(flux_parabolic_x[v, i_large, j_large,
                                                                 k_large, element],
                                                flux_parabolic_y[v, i_large, j_large,
                                                                 k_large, element],
                                                flux_parabolic_z[v, i_large, j_large,
                                                                 k_large, element])

                            # same convention as local code
                            u_buffer[v, i, j] = -0.5f0 *
                                                dot(flux_node, normal_direction)
                        end

                        i_large += i_large_step_i
                        j_large += j_large_step_i
                        k_large += k_large_step_i
                    end
                    i_large += i_large_step_j
                    j_large += j_large_step_j
                    k_large += k_large_step_j
                end

                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 1, :, :,
                                             mortar),
                                        mortar_l2.forward_lower,
                                        mortar_l2.forward_lower,
                                        u_buffer, fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 2, :, :,
                                             mortar),
                                        mortar_l2.forward_upper,
                                        mortar_l2.forward_lower,
                                        u_buffer, fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 3, :, :,
                                             mortar),
                                        mortar_l2.forward_lower,
                                        mortar_l2.forward_upper,
                                        u_buffer, fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 4, :, :,
                                             mortar),
                                        mortar_l2.forward_upper,
                                        mortar_l2.forward_upper,
                                        u_buffer, fstar_tmp)

            else
                # =========================
                # SMALL ELEMENT (1–4)
                # =========================
                i_small = i_small_start
                j_small = j_small_start
                k_small = k_small_start

                for j in eachnode(dg)
                    for i in eachnode(dg)
                        normal_direction = get_normal_direction(direction_index,
                                                                contravariant_vectors,
                                                                i_small, j_small,
                                                                k_small,
                                                                element)

                        for v in eachvariable(equations_parabolic)
                            flux_node = SVector(flux_parabolic_x[v, i_small, j_small,
                                                                 k_small, element],
                                                flux_parabolic_y[v, i_small, j_small,
                                                                 k_small, element],
                                                flux_parabolic_z[v, i_small, j_small,
                                                                 k_small, element])

                            cache.mpi_mortars.u[1, v, position, i, j, mortar] = dot(flux_node,
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

    return nothing
end

function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                             mesh::Union{P4estMeshParallel{3},
                                                         T8codeMeshParallel{3}},
                                             equations_parabolic,
                                             dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                     index_range)
        j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                     index_range)
        k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                     index_range)

        i_element = i_element_start
        j_element = j_element_start
        k_element = k_element_start

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
                                                        i_element, j_element, k_element,
                                                        local_element)

                calc_mpi_interface_flux_divergence!(surface_flux_values, mesh,
                                                    equations_parabolic,
                                                    dg, parabolic_scheme, cache,
                                                    interface, normal_direction,
                                                    i, j, local_side,
                                                    i_surface, j_surface,
                                                    local_direction, local_element)

                i_element += i_element_step_i
                j_element += j_element_step_i
                k_element += k_element_step_i

                i_surface += i_surface_step_i
                j_surface += j_surface_step_i
            end

            i_element += i_element_step_j
            j_element += j_element_step_j
            k_element += k_element_step_j

            i_surface += i_surface_step_j
            j_surface += j_surface_step_j
        end
    end

    return nothing
end

@inline function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                                     mesh::Union{P4estMeshParallel{3},
                                                                 T8codeMeshParallel{3}},
                                                     equations_parabolic,
                                                     dg::DG, parabolic_scheme, cache,
                                                     interface_index, normal_direction,
                                                     interface_i_node_index,
                                                     interface_j_node_index, local_side,
                                                     surface_i_node_index,
                                                     surface_j_node_index,
                                                     local_direction_index,
                                                     local_element_index)
    @unpack u = cache.mpi_interfaces

    parabolic_flux_normal_ll, parabolic_flux_normal_rr = get_surface_node_vars(u,
                                                                               equations_parabolic,
                                                                               dg,
                                                                               interface_i_node_index,
                                                                               interface_j_node_index,
                                                                               interface_index)

    flux_ = flux_parabolic(parabolic_flux_normal_ll, parabolic_flux_normal_rr,
                           normal_direction, Divergence(),
                           equations_parabolic, parabolic_scheme)

    dirFactor = (local_side == 1) ? 1 : -1

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, surface_i_node_index, surface_j_node_index,
        local_direction_index, local_element_index] = dirFactor .* flux_[v]
    end

    return nothing
end

function calc_mpi_mortar_flux_divergence!(surface_flux_values,
                                          mesh::Union{P4estMeshParallel{3},
                                                      T8codeMeshParallel{3}},
                                          equations_parabolic,
                                          mortar_l2::LobattoLegendreMortarL2,
                                          dg::DG, parabolic_scheme, cache)
    @unpack fstar_primary_threaded, fstar_tmp_threaded = cache

    @threaded for mortar in eachmpimortar(dg, cache)
        fstar = fstar_primary_threaded[Threads.threadid()]
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        for position in 1:4
            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(cache.mpi_mortars, i, j,
                                                            position, mortar)

                    calc_mpi_mortar_flux_divergence!(fstar, mesh,
                                                     equations_parabolic,
                                                     dg, parabolic_scheme, cache,
                                                     mortar, position,
                                                     normal_direction, i, j)
                end
            end
        end

        u_buffer = cache.u_threaded[Threads.threadid()]

        mpi_mortar_fluxes_to_elements_divergence!(surface_flux_values, mesh,
                                                  equations_parabolic, mortar_l2,
                                                  dg, cache,
                                                  mortar, fstar, u_buffer, fstar_tmp)
    end

    return nothing
end

@inline function calc_mpi_mortar_flux_divergence!(fstar,
                                                  mesh::Union{P4estMeshParallel{3},
                                                              T8codeMeshParallel{3}},
                                                  equations_parabolic,
                                                  dg::DG, parabolic_scheme, cache,
                                                  mortar_index, position_index,
                                                  normal_direction,
                                                  i_node_index, j_node_index)
    @unpack u = cache.mpi_mortars

    for v in eachvariable(equations_parabolic)
        parabolic_flux_normal_ll = u[1, v, position_index, i_node_index, j_node_index,
                                     mortar_index]
        parabolic_flux_normal_rr = u[2, v, position_index, i_node_index, j_node_index,
                                     mortar_index]

        flux_ = flux_parabolic(parabolic_flux_normal_ll, parabolic_flux_normal_rr,
                               normal_direction, Divergence(),
                               equations_parabolic, parabolic_scheme)

        fstar[v, i_node_index, j_node_index, position_index] = flux_
    end

    return nothing
end

@inline function mpi_mortar_fluxes_to_elements_divergence!(surface_flux_values,
                                                           mesh::Union{P4estMeshParallel{3},
                                                                       T8codeMeshParallel{3}},
                                                           equations_parabolic,
                                                           mortar_l2::LobattoLegendreMortarL2,
                                                           dg::DGSEM, cache, mortar,
                                                           fstar, u_buffer, fstar_tmp)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars

    mpi_mortar_fluxes_to_elements!(surface_flux_values, mesh,
                                   equations_parabolic, mortar_l2, dg, cache,
                                   mortar, fstar, fstar, u_buffer, fstar_tmp)

    return nothing
end

function calc_mpi_mortar_flux_gradient!(surface_flux_values,
                                        mesh::Union{P4estMeshParallel{3},
                                                    T8codeMeshParallel{3}},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DG, parabolic_scheme, cache)
    @unpack fstar_primary_threaded, fstar_secondary_threaded, fstar_tmp_threaded = cache

    @threaded for mortar in eachmpimortar(dg, cache)
        fstar_primary = fstar_primary_threaded[Threads.threadid()]
        fstar_secondary = fstar_secondary_threaded[Threads.threadid()]
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        for position in 1:4
            for j in eachnode(dg)
                for i in eachnode(dg)
                    normal_direction = get_normal_direction(cache.mpi_mortars, i, j,
                                                            position, mortar)

                    calc_mpi_mortar_flux_gradient!(fstar_primary, fstar_secondary,
                                                   mesh, equations_parabolic,
                                                   dg, parabolic_scheme, cache,
                                                   mortar, position,
                                                   normal_direction, i, j)
                end
            end
        end

        u_buffer = cache.u_threaded[Threads.threadid()]

        mpi_mortar_fluxes_to_elements_gradient!(surface_flux_values,
                                                mesh, equations_parabolic, mortar_l2,
                                                dg, cache,
                                                mortar, fstar_primary, fstar_secondary,
                                                u_buffer, fstar_tmp)
    end

    return nothing
end

@inline function calc_mpi_mortar_flux_gradient!(fstar_primary, fstar_secondary,
                                                mesh::Union{P4estMeshParallel{3},
                                                            T8codeMeshParallel{3}},
                                                equations_parabolic,
                                                dg::DG, parabolic_scheme, cache,
                                                mortar_index, position_index,
                                                normal_direction,
                                                i_node_index, j_node_index)
    @unpack u = cache.mpi_mortars

    u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg, position_index,
                                       i_node_index, j_node_index, mortar_index)

    flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                           equations_parabolic, parabolic_scheme)

    set_node_vars!(fstar_primary, flux_, equations_parabolic, dg,
                   i_node_index, j_node_index, position_index)
    set_node_vars!(fstar_secondary, flux_, equations_parabolic, dg,
                   i_node_index, j_node_index, position_index)

    return nothing
end

@inline function mpi_mortar_fluxes_to_elements_gradient!(surface_flux_values,
                                                         mesh::Union{P4estMeshParallel{3},
                                                                     T8codeMeshParallel{3}},
                                                         equations_parabolic,
                                                         mortar_l2::LobattoLegendreMortarL2,
                                                         dg::DGSEM, cache, mortar,
                                                         fstar_primary, fstar_secondary,
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
        if position == 5
            multiply_dimensionwise!(u_buffer,
                                    mortar_l2.reverse_lower, mortar_l2.reverse_lower,
                                    view(fstar_secondary, .., 1),
                                    fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_lower,
                                        view(fstar_secondary, .., 2),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_lower,
                                        mortar_l2.reverse_upper,
                                        view(fstar_secondary, .., 3),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_upper,
                                        view(fstar_secondary, .., 4),
                                        fstar_tmp)

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
        else
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations_parabolic)
                        surface_flux_values[v, i, j, small_direction, element] = fstar_primary[v,
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

function prolong2mpiinterfaces!(cache, flux_parabolic::Tuple,
                                mesh::Union{P4estMeshParallel{3},
                                            T8codeMeshParallel{3}},
                                equations_parabolic, dg::DG)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]
        orientationFactor = local_side == 1 ? 1 : -1

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
                    flux_parabolic = SVector(flux_parabolic_x[v, i_elem, j_elem, k_elem,
                                                              local_element],
                                             flux_parabolic_y[v, i_elem, j_elem, k_elem,
                                                              local_element],
                                             flux_parabolic_z[v, i_elem, j_elem, k_elem,
                                                              local_element])

                    cache.mpi_interfaces.u[local_side, v, i, j, interface] = orientationFactor .*
                                                                             dot(flux_parabolic,
                                                                                 normal_direction)
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

    return nothing
end
end # @muladd
