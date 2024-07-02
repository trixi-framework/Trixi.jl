# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function _prolong2mpiinterfaces!(backend::Backend, cache, u,
                                         mesh::P4estMesh{3},
                                         equations, surface_integral, dg::DG)
    @unpack mpi_interfaces = cache
    nmpiinterfaces(mpi_interfaces) == 0 && return nothing

    nodes = eachnode(dg)
    kernel! = prolong2mpiinterfaces_kernel!(backend)

    kernel!(mpi_interfaces.u, mpi_interfaces.local_sides,
            mpi_interfaces.local_neighbor_ids,
            mpi_interfaces.node_indices, u, Val(nvariables(equations)), nodes,
            ndrange = nmpiinterfaces(mpi_interfaces))
    return nothing
end

@kernel function prolong2mpiinterfaces_kernel!(u_mpi_interfaces, local_sides,
                                               local_neighbor_ids,
                                               node_indices, u, ::Val{NVARS},
                                               nodes) where {NVARS}
    interface = @index(Global, Linear)
    # Copy solution data from the local element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the index of the primary side
    # will always run forwards.
    local_side = local_sides[interface]
    local_element = local_neighbor_ids[interface]
    local_indices = node_indices[interface]

    i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                 nodes)
    j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                 nodes)
    k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                 nodes)

    i_element = i_element_start
    j_element = j_element_start
    k_element = k_element_start
    for j in nodes
        for i in nodes
            for v in 1:NVARS
                u_mpi_interfaces[local_side, v, i, j, interface] = u[v, i_element,
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

@inline function _calc_mpi_interface_flux!(backend::Backend, surface_flux_values,
                                           mesh::ParallelP4estMesh{3},
                                           nonconservative_terms::False,
                                           equations, surface_integral, dg::DG, cache)
    @unpack mpi_interfaces = cache
    nmpiinterfaces(mpi_interfaces) == 0 && return nothing

    @unpack local_neighbor_ids, node_indices, local_sides = mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    nodes = eachnode(dg)
    kernel! = mpi_interface_flux_kernel!(backend)

    kernel!(surface_flux_values, equations, surface_integral.surface_flux, nodes,
            mpi_interfaces.u, local_neighbor_ids, node_indices, local_sides,
            contravariant_vectors, ndrange = nmpiinterfaces(mpi_interfaces))
    return nothing
end

@kernel function mpi_interface_flux_kernel!(surface_flux_values, equations,
                                            surface_flux, nodes,
                                            u_mpi_interfaces, local_neighbor_ids,
                                            node_indices, local_sides,
                                            contravariant_vectors)
    interface = @index(Global, Linear)
    NVARS = Val(nvariables(equations))

    # Get element and side index information on the local element
    local_element = local_neighbor_ids[interface]
    local_indices = node_indices[interface]
    local_direction = indices2direction(local_indices)
    local_side = local_sides[interface]

    # Create the local i,j,k indexing on the local element used to pull normal direction information
    i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                 nodes)
    j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                 nodes)
    k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                 nodes)

    i_element = i_element_start
    j_element = j_element_start
    k_element = k_element_start

    # Initiate the node indices to be used in the surface for loop,
    # the surface flux storage must be indexed in alignment with the local element indexing
    local_surface_indices = surface_indices(local_indices)
    i_surface_start, i_surface_step_i, i_surface_step_j = index_to_start_step_3d(local_surface_indices[1],
                                                                                 nodes)
    j_surface_start, j_surface_step_i, j_surface_step_j = index_to_start_step_3d(local_surface_indices[2],
                                                                                 nodes)
    i_surface = i_surface_start
    j_surface = j_surface_start

    for j in nodes
        for i in nodes
            # Get the normal direction on the local element
            # Contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_element, j_element, k_element,
                                                    local_element)
            u_ll, u_rr = get_svectors(u_mpi_interfaces, NVARS, i, j, interface)

            if local_side == 1
                flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
            else # local_side == 2
                flux_ = -surface_flux(u_ll, u_rr, -normal_direction, equations)
            end

            for v in 1:nvariables(equations)
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

@inline function _prolong2mpimortars!(backend::Backend, cache, u, 
                                      mesh::ParallelP4estMesh{3},
                                      equations,
                                      mortar_l2::LobattoLegendreMortarL2,
                                      surface_integral, dg::DGSEM)
    if nmpimortars(dg, cache) > 0
        error("mortars currently not supported by KA.jl P4estMesh solver")
    end
    return nothing
end

@inline function _calc_mpi_mortar_flux!(backend::Backend, surface_flux_values,
                                        mesh::ParallelP4estMesh{3},
                                        nonconservative_terms, equations,
                                        mortar_l2::LobattoLegendreMortarL2,
                                        surface_integral, dg::DG, cache)
    return nothing
end
end # @muladd