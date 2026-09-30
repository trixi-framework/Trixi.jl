# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

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
        # Sign flip for `local_side = 2` required for divergence calculation since
        # the divergence interface flux involves the normal direction.
        # `local_side=2` is thus flipped (opposite of primary side)
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
                    flux_parabolic = SVector(flux_parabolic_x[v, i_elem, j_elem, k_elem,
                                                              local_element],
                                             flux_parabolic_y[v, i_elem, j_elem, k_elem,
                                                              local_element],
                                             flux_parabolic_z[v, i_elem, j_elem, k_elem,
                                                              local_element])

                    cache.mpi_interfaces.u[local_side, v, i, j, interface] = orientation_factor .*
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

function calc_mpi_interface_flux_gradient!(surface_flux_values,
                                           mesh::Union{P4estMeshParallel{3},
                                                       T8codeMeshParallel{3}},
                                           equations_parabolic,
                                           dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)

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

                u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg,
                                                   i, j, interface)

                flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
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

    return nothing
end

function calc_mpi_interface_flux_divergence!(surface_flux_values,
                                             mesh::Union{P4estMeshParallel{3},
                                                         T8codeMeshParallel{3}},
                                             equations_parabolic,
                                             dg::DG, parabolic_scheme, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    @unpack u = cache.mpi_interfaces
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

                parabolic_flux_normal_ll, parabolic_flux_normal_rr = get_surface_node_vars(u,
                                                                                           equations_parabolic,
                                                                                           dg,
                                                                                           i,
                                                                                           j,
                                                                                           interface)

                # Sign flip for `local_side = 2` required for divergence calculation since
                # the divergence interface flux involves the normal direction.
                # `local_side=2` is thus flipped (opposite of primary side)
                orientation_factor = (local_side == 1) ? 1 : -1
                flux_ = flux_parabolic(parabolic_flux_normal_ll,
                                       parabolic_flux_normal_rr,
                                       orientation_factor * normal_direction,
                                       Divergence(),
                                       equations_parabolic, parabolic_scheme)

                for v in eachvariable(equations_parabolic)
                    surface_flux_values[v, i_surface, j_surface,
                    local_direction, local_element] = orientation_factor * flux_[v]
                end

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

function calc_mpi_mortar_flux_gradient!(surface_flux_values,
                                        mesh::Union{P4estMeshParallel{3},
                                                    T8codeMeshParallel{3}},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DG, parabolic_scheme, cache)
    @assert nmpimortars(dg, cache)==0 "Mortars are not yet implemented for 3D parabolic p4est simulations"

    return nothing
end

function prolong2mpimortars_divergence!(cache, flux_parabolic,
                                        mesh::Union{P4estMeshParallel{3},
                                                    T8codeMeshParallel{3}},
                                        equations_parabolic,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        dg::DGSEM)
    @assert nmpimortars(dg, cache)==0 "Mortars are not yet implemented for 3D parabolic p4est simulations"

    return nothing
end

function calc_mpi_mortar_flux_divergence!(surface_flux_values,
                                          mesh::Union{P4estMeshParallel{3},
                                                      T8codeMeshParallel{3}},
                                          equations_parabolic,
                                          mortar_l2::LobattoLegendreMortarL2,
                                          dg::DG, parabolic_scheme, cache)
    @assert nmpimortars(dg, cache)==0 "Mortars are not yet implemented for 3D parabolic p4est simulations"

    return nothing
end

end # @muladd
