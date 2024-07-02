# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function _calc_volume_integral!(backend::Backend, du, u, mesh::P4estMesh{3},
                                        nonconservative_terms::False, equations,
                                        volume_integral::VolumeIntegralWeakForm,
                                        dg::DGSEM,
                                        cache)
    nelements(dg, cache) == 0 && return nothing
    
    @unpack derivative_dhat = dg.basis
    @unpack contravariant_vectors = cache.elements
    nodes = eachnode(dg)
    kernel! = _weak_form_kernel!(backend)

    kernel!(du, u, equations, nodes, derivative_dhat, contravariant_vectors,
            ndrange = nelements(dg, cache))
    return nothing
end

@kernel function _weak_form_kernel!(du, u, equations, nodes, derivative_dhat,
                                    contravariant_vectors, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    element = @index(Global)
    NVARS = Val(nvariables(equations))

    for k in nodes, j in nodes, i in nodes
        u_node = get_svector(u, NVARS, i, j, k, element)

        flux1 = flux(u_node, 1, equations)
        flux2 = flux(u_node, 2, equations)
        flux3 = flux(u_node, 3, equations)

        # Compute the contravariant flux by taking the scalar product of the
        # first contravariant vector Ja^1 and the flux vector
        Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j, k,
                                                    element)
        contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2 + Ja13 * flux3
        for ii in nodes
            multiply_add_to_first_axis!(du, alpha * derivative_dhat[ii, i],
                                        contravariant_flux1, ii, j, k,
                                        element)
        end

        # Compute the contravariant flux by taking the scalar product of the
        # second contravariant vector Ja^2 and the flux vector
        Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j, k,
                                                    element)
        contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2 + Ja23 * flux3
        for jj in nodes
            multiply_add_to_first_axis!(du, alpha * derivative_dhat[jj, j],
                                        contravariant_flux2, i, jj, k,
                                        element)
        end

        # Compute the contravariant flux by taking the scalar product of the
        # third contravariant vector Ja^3 and the flux vector
        Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j, k,
                                                    element)
        contravariant_flux3 = Ja31 * flux1 + Ja32 * flux2 + Ja33 * flux3
        for kk in nodes
            multiply_add_to_first_axis!(du, alpha * derivative_dhat[kk, k],
                                        contravariant_flux3, i, j, kk,
                                        element)
        end
    end
end

@inline function _calc_volume_integral!(backend::Backend, du, u,
                                        mesh::P4estMesh{3},
                                        nonconservative_terms::False, equations,
                                        volume_integral::VolumeIntegralFluxDifferencing,
                                        dg::DGSEM, cache)
    @unpack derivative_split = dg.basis
    @unpack contravariant_vectors = cache.elements
    nodes = eachnode(dg)
    kernel! = _flux_differencing_kernel!(backend)

    kernel!(du, u, equations, volume_integral.volume_flux, nodes, derivative_split,
            contravariant_vectors,
            ndrange = nelements(dg, cache))
    return nothing
end

@kernel function _flux_differencing_kernel!(du, u, equations,
                                            volume_flux, nodes, derivative_split,
                                            contravariant_vectors, alpha = true)
    # true * [some floating point value] == [exactly the same floating point value]
    # This can (hopefully) be optimized away due to constant propagation.
    element = @index(Global, Linear)
    NVARS = Val(nvariables(equations))
    num_nodes = length(nodes)

    # Calculate volume integral in one element
    for k in nodes, j in nodes, i in nodes
        u_node = get_svector(u, NVARS, i, j, k, element)

        # pull the contravariant vectors in each coordinate direction
        Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
        Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
        Ja3_node = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of the `volume_flux` to save half of the possible two-point flux
        # computations.

        # x direction
        for ii in (i + 1):num_nodes
            u_node_ii = get_svector(u, NVARS, ii, j, k, element)
            # pull the contravariant vectors and compute the average
            Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                   ii, j, k, element)
            Ja1_avg = 0.5 * (Ja1_node + Ja1_node_ii)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
            multiply_add_to_first_axis!(du, alpha * derivative_split[i, ii], fluxtilde1,
                                        i, j, k, element)
            multiply_add_to_first_axis!(du, alpha * derivative_split[ii, i], fluxtilde1,
                                        ii, j, k, element)
        end

        # y direction
        for jj in (j + 1):num_nodes
            u_node_jj = get_svector(u, NVARS, i, jj, k, element)
            # pull the contravariant vectors and compute the average
            Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                   i, jj, k, element)
            Ja2_avg = 0.5 * (Ja2_node + Ja2_node_jj)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
            multiply_add_to_first_axis!(du, alpha * derivative_split[j, jj], fluxtilde2,
                                        i, j, k, element)
            multiply_add_to_first_axis!(du, alpha * derivative_split[jj, j], fluxtilde2,
                                        i, jj, k, element)
        end

        # z direction
        for kk in (k + 1):num_nodes
            u_node_kk = get_svector(u, NVARS, i, j, kk, element)
            # pull the contravariant vectors and compute the average
            Ja3_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                   i, j, kk, element)
            Ja3_avg = 0.5 * (Ja3_node + Ja3_node_kk)
            # compute the contravariant sharp flux in the direction of the
            # averaged contravariant vector
            fluxtilde3 = volume_flux(u_node, u_node_kk, Ja3_avg, equations)
            multiply_add_to_first_axis!(du, alpha * derivative_split[k, kk], fluxtilde3,
                                        i, j, k, element)
            multiply_add_to_first_axis!(du, alpha * derivative_split[kk, k], fluxtilde3,
                                        i, j, kk, element)
        end
    end
end

@inline function _prolong2interfaces!(backend::Backend, cache, u,
                                      mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                      equations, surface_integral, dg::DG)
    @unpack interfaces = cache
    ninterfaces(interfaces) == 0 && return nothing

    nodes = eachnode(dg)
    kernel! = prolong2interfaces_kernel!(backend)

    kernel!(interfaces.u, interfaces.neighbor_ids, interfaces.node_indices, u,
            Val(nvariables(equations)), nodes,
            ndrange = ninterfaces(interfaces))
    return nothing
end

@kernel function prolong2interfaces_kernel!(u_interfaces, neighbor_ids, node_indices, u,
                                            ::Val{NVARS}, nodes) where {NVARS}
    interface = @index(Global, Linear)
    # Copy solution data from the primary element using "delayed indexing" with
    # a start value and two step sizes to get the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the indices of the primary side
    # will always run forwards.
    primary_element = neighbor_ids[1, interface]
    primary_indices = node_indices[1, interface]

    i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                 nodes)
    j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                 nodes)
    k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                 nodes)

    i_primary = i_primary_start
    j_primary = j_primary_start
    k_primary = k_primary_start
    for j in nodes
        for i in nodes
            for v in 1:NVARS
                u_interfaces[1, v, i, j, interface] = u[v, i_primary, j_primary,
                                                        k_primary, primary_element]
            end
            i_primary += i_primary_step_i
            j_primary += j_primary_step_i
            k_primary += k_primary_step_i
        end
        i_primary += i_primary_step_j
        j_primary += j_primary_step_j
        k_primary += k_primary_step_j
    end

    # Copy solution data from the secondary element using "delayed indexing" with
    # a start value and two step sizes to get the correct face and orientation.
    secondary_element = neighbor_ids[2, interface]
    secondary_indices = node_indices[2, interface]

    i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                       nodes)
    j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                       nodes)
    k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                       nodes)

    i_secondary = i_secondary_start
    j_secondary = j_secondary_start
    k_secondary = k_secondary_start
    for j in nodes
        for i in nodes
            for v in 1:NVARS
                u_interfaces[2, v, i, j, interface] = u[v, i_secondary, j_secondary,
                                                        k_secondary,
                                                        secondary_element]
            end
            i_secondary += i_secondary_step_i
            j_secondary += j_secondary_step_i
            k_secondary += k_secondary_step_i
        end
        i_secondary += i_secondary_step_j
        j_secondary += j_secondary_step_j
        k_secondary += k_secondary_step_j
    end
end

@inline function _calc_interface_flux!(backend::Backend, surface_flux_values,
                                       mesh::P4estMesh{3},
                                       nonconservative_terms::False,
                                       equations, surface_integral, dg::DG, cache)
    @unpack interfaces = cache
    ninterfaces(interfaces) == 0 && return nothing

    @unpack neighbor_ids, node_indices = interfaces
    @unpack contravariant_vectors = cache.elements
    nodes = eachnode(dg)
    kernel! = interface_flux_kernel!(backend)

    kernel!(surface_flux_values, equations, surface_integral.surface_flux, nodes,
            interfaces.u, neighbor_ids, node_indices, contravariant_vectors,
            ndrange = ninterfaces(interfaces))
    return nothing
end

@kernel function interface_flux_kernel!(surface_flux_values, equations, surface_flux, nodes,
                                        u_interfaces, neighbor_ids, node_indices,
                                        contravariant_vectors)
    interface = @index(Global, Linear)
    NVARS = Val(nvariables(equations))

    # Get element and side information on the primary element
    primary_element = neighbor_ids[1, interface]
    primary_indices = node_indices[1, interface]
    primary_direction = indices2direction(primary_indices)

    i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                 nodes)
    j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                 nodes)
    k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                 nodes)

    i_primary = i_primary_start
    j_primary = j_primary_start
    k_primary = k_primary_start

    # Get element and side information on the secondary element
    secondary_element = neighbor_ids[2, interface]
    secondary_indices = node_indices[2, interface]
    secondary_direction = indices2direction(secondary_indices)
    secondary_surface_indices = surface_indices(secondary_indices)

    # Get the surface indexing on the secondary element.
    # Note that the indices of the primary side will always run forward but
    # the secondary indices might need to run backwards for flipped sides.
    i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_surface_indices[1],
                                                                                       nodes)
    j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_surface_indices[2],
                                                                                       nodes)
    i_secondary = i_secondary_start
    j_secondary = j_secondary_start

    for j in nodes
        for i in nodes
            # Get the normal direction from the primary element.
            # Note, contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(primary_direction,
                                                    contravariant_vectors,
                                                    i_primary, j_primary, k_primary,
                                                    primary_element)
            u_ll, u_rr = get_svectors(u_interfaces, NVARS, i, j, interface)

            flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)

            for v in eachvariable(equations)
                surface_flux_values[v, i, j, primary_direction, primary_element] = flux_[v]
                surface_flux_values[v, i_secondary, j_secondary,
                secondary_direction, secondary_element] = -flux_[v]
            end

            # Increment the primary element indices
            i_primary += i_primary_step_i
            j_primary += j_primary_step_i
            k_primary += k_primary_step_i
            # Increment the secondary element surface indices
            i_secondary += i_secondary_step_i
            j_secondary += j_secondary_step_i
        end
        # Increment the primary element indices
        i_primary += i_primary_step_j
        j_primary += j_primary_step_j
        k_primary += k_primary_step_j
        # Increment the secondary element surface indices
        i_secondary += i_secondary_step_j
        j_secondary += j_secondary_step_j
    end
end

@inline function _prolong2boundaries!(backend::Backend, cache, u, mesh::P4estMesh{3},
                                      equations, surface_integral, dg::DG)
    @unpack boundaries = cache
    nboundaries(boundaries) == 0 && return nothing

    nodes = eachnode(dg)
    kernel! = prolong2boundaries_kernel!(backend)

    kernel!(boundaries.u, boundaries.neighbor_ids, boundaries.node_indices, u,
            Val(nvariables(equations)), nodes,
            ndrange = nboundaries(boundaries))
    return nothing
end

@kernel function prolong2boundaries_kernel!(u_boundaries, neighbor_ids, _node_indices, u,
                                            ::Val{NVARS}, nodes) where {NVARS}
    boundary = @index(Global, Linear)

    # Copy solution data from the element using "delayed indexing" with
    # a start value and two step sizes to get the correct face and orientation.
    element = neighbor_ids[boundary]
    node_indices = _node_indices[boundary]

    i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                        nodes)
    j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                        nodes)
    k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                        nodes)

    i_node = i_node_start
    j_node = j_node_start
    k_node = k_node_start
    for j in nodes
        for i in nodes
            for v in 1:NVARS
                u_boundaries[v, i, j, boundary] = u[v, i_node, j_node, k_node,
                                                    element]
            end
            i_node += i_node_step_i
            j_node += j_node_step_i
            k_node += k_node_step_i
        end
        i_node += i_node_step_j
        j_node += j_node_step_j
        k_node += k_node_step_j
    end
end

@inline function _calc_boundary_flux!(backend::Backend, cache, t,
                                      boundary_condition, boundary_indexing,
                                      mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                      equations, surface_integral, dg::DG)
    @unpack boundaries, elements = cache
    nboundaries(boundaries) == 0 && return nothing

    @unpack neighbor_ids, node_indices = boundaries
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = elements
    nodes = eachnode(dg)
    kernel! = boundary_flux_kernel!(backend)

    kernel!(surface_flux_values, t, boundary_condition, boundary_indexing, equations,
            surface_integral.surface_flux, nodes,
            boundaries.u, neighbor_ids, node_indices,
            node_coordinates, contravariant_vectors,
            ndrange = nboundaries(boundaries))
    return nothing
end

@kernel function boundary_flux_kernel!(surface_flux_values, t,
                                       boundary_condition, boundary_indexing, equations,
                                       surface_flux, nodes,
                                       u_boundaries, neighbor_ids, _node_indices,
                                       node_coordinates, contravariant_vectors)
    local_index = @index(Global, Linear)
    NVARS = Val(nvariables(equations))
    boundary = boundary_indexing[local_index]

    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element = neighbor_ids[boundary]
    node_indices = _node_indices[boundary]
    direction = indices2direction(node_indices)

    i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                        nodes)
    j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                        nodes)
    k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                        nodes)

    i_node = i_node_start
    j_node = j_node_start
    k_node = k_node_start
    for j in nodes
        for i in nodes
            # Extract solution data from boundary container
            u_inner = get_svector(u_boundaries, NVARS, i, j, boundary)

            # Outward-pointing normal direction (not normalized)
            normal_direction = get_normal_direction(direction,
                                                    contravariant_vectors,
                                                    i_node, j_node, k_node, element)

            # Coordinates at boundary node
            x = get_svector(node_coordinates, Val(3), i_node, j_node, k_node, element)

            flux_ = boundary_condition(u_inner, normal_direction, x, t,
                                       surface_flux, equations)

            # Copy flux to element storage in the correct orientation
            for v in eachvariable(equations)
                surface_flux_values[v, i, j, direction, element] = flux_[v]
            end

            i_node += i_node_step_i
            j_node += j_node_step_i
            k_node += k_node_step_i
        end
        i_node += i_node_step_j
        j_node += j_node_step_j
        k_node += k_node_step_j
    end
end

@inline function _prolong2mortars!(backend::Backend, cache, u,
                                   mesh::P4estMesh{3}, equations,
                                   mortar_l2::LobattoLegendreMortarL2,
                                   surface_integral, dg::DGSEM)
    if nmortars(dg, cache) > 0
        error("mortars currently not supported by KA.jl P4estMesh solver")
    end
    return nothing
end

@inline function _calc_mortar_flux!(backend::Backend, surface_flux_values,
                                    mesh::P4estMesh{3},
                                    nonconservative_terms, equations,
                                    mortar_l2::LobattoLegendreMortarL2,
                                    surface_integral, dg::DG, cache)
    return nothing
end

@inline function _calc_surface_integral!(backend::Backend, du, u,
                                         mesh::P4estMesh{3},
                                         equations,
                                         surface_integral::SurfaceIntegralWeakForm,
                                         dg::DGSEM, cache)
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache.elements
    nodes = eachnode(dg)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1, 1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]
    kernel! = surface_integral_kernel!(backend)

    kernel!(du, u, Val(nvariables(equations)), factor_1, factor_2, nodes,
            surface_flux_values, ndrange = nelements(cache.elements))
    return nothing
end

@kernel function surface_integral_kernel!(du, u, ::Val{NVARS},
                                          boundary_interp_factor_1, boundary_interp_factor_2,
                                          nodes, surface_flux_values) where {NVARS}
    element = @index(Global, Linear)
    num_nodes = length(nodes)
    for m in nodes, l in nodes
        for v in 1:NVARS
            # surface at -x
            du[v, 1, l, m, element] = (du[v, 1, l, m, element] +
                                        surface_flux_values[v, l, m, 1, element] *
                                        boundary_interp_factor_1)

            # surface at +x
            du[v, num_nodes, l, m, element] = (du[v, num_nodes, l, m, element] +
                                                surface_flux_values[v, l, m, 2,
                                                                    element] *
                                                boundary_interp_factor_2)

            # surface at -y
            du[v, l, 1, m, element] = (du[v, l, 1, m, element] +
                                        surface_flux_values[v, l, m, 3, element] *
                                        boundary_interp_factor_1)

            # surface at +y
            du[v, l, num_nodes, m, element] = (du[v, l, num_nodes, m, element] +
                                                surface_flux_values[v, l, m, 4,
                                                                    element] *
                                                boundary_interp_factor_2)

            # surface at -z
            du[v, l, m, 1, element] = (du[v, l, m, 1, element] +
                                        surface_flux_values[v, l, m, 5, element] *
                                        boundary_interp_factor_1)

            # surface at +z
            du[v, l, m, num_nodes, element] = (du[v, l, m, num_nodes, element] +
                                                surface_flux_values[v, l, m, 6,
                                                                    element] *
                                                boundary_interp_factor_2)
        end
    end
end

@inline function _apply_jacobian!(backend::Backend, du, mesh::P4estMesh{3},
                                  equations, dg::DG, cache)
    NVARS = Val(nvariables(equations))
    nodes = eachnode(dg)
    @unpack inverse_jacobian = cache.elements
    kernel! = _apply_jacobian_kernel!(backend)

    kernel!(du, inverse_jacobian, NVARS, nodes, ndrange = nelements(cache.elements))
    return nothing
end

@kernel function _apply_jacobian_kernel!(du, inverse_jacobian,
                                         ::Val{NVARS}, nodes) where {NVARS}
    element = @index(Global, Linear)
    for k in nodes, j in nodes, i in nodes
        factor = -inverse_jacobian[i, j, k, element]
        for v in 1:NVARS
            du[v, i, j, k, element] *= factor
        end
    end
end

@inline function _calc_sources!(backend::Backend, du, u, t, source_terms,
                                equations::AbstractEquations{3}, dg::DG, cache)
    @unpack node_coordinates = cache.elements
    NVARS = Val(nvariables(equations))
    nodes = eachnode(dg)
    kernel! = _calc_sources_kernel!(backend)

    kernel!(du, u, t, source_terms, equations, NVARS, nodes, node_coordinates;
            ndrange = nelements(cache.elements))

    return nothing
end

@kernel function _calc_sources_kernel!(du, u, t, source_terms, equations, NVARS,
                                       nodes, node_coordinates)
    element = @index(Global, Linear)
    for k in nodes, j in nodes, i in nodes
        u_local = get_svector(u, NVARS, i, j, k, element)
        x_local = get_svector(node_coordinates, Val(3), i, j, k, element)
        du_local = source_terms(u_local, x_local, t, equations)
        add_to_first_axis!(du, du_local, i, j, k, element)
    end
end
end # @muladd