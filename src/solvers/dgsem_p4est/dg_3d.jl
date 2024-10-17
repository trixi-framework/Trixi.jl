# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{P4estMesh{3}, T8codeMesh{3}}, equations,
                      mortar_l2::LobattoLegendreMortarL2, uEltype)
    # TODO: Taal compare performance of different types
    fstar_threaded = [Array{uEltype, 4}(undef, nvariables(equations), nnodes(mortar_l2),
                                        nnodes(mortar_l2), 4)
                      for _ in 1:Threads.nthreads()]

    fstar_tmp_threaded = [Array{uEltype, 3}(undef, nvariables(equations),
                                            nnodes(mortar_l2), nnodes(mortar_l2))
                          for _ in 1:Threads.nthreads()]
    u_threaded = [Array{uEltype, 3}(undef, nvariables(equations), nnodes(mortar_l2),
                                    nnodes(mortar_l2))
                  for _ in 1:Threads.nthreads()]

    (; fstar_threaded, fstar_tmp_threaded, u_threaded)
end

#     index_to_start_step_3d(index::Symbol, index_range)
#
# Given a symbolic `index` and an `indexrange` (usually `eachnode(dg)`),
# return `index_start, index_step_i, index_step_j`, i.e., a tuple containing
# - `index_start`,   an index value to begin a loop
# - `index_step_i`,  an index step to update during an `i` loop
# - `index_step_j`,  an index step to update during a `j` loop
# The resulting indices translate surface indices to volume indices.
#
# !!! warning
#     This assumes that loops using the return values are written as
#
#     i_volume_start, i_volume_step_i, i_volume_step_j = index_to_start_step_3d(symbolic_index_i, index_range)
#     j_volume_start, j_volume_step_i, j_volume_step_j = index_to_start_step_3d(symbolic_index_j, index_range)
#     k_volume_start, k_volume_step_i, k_volume_step_j = index_to_start_step_3d(symbolic_index_k, index_range)
#
#     i_volume, j_volume, k_volume = i_volume_start, j_volume_start, k_volume_start
#     for j_surface in index_range
#       for i_surface in index_range
#         # do stuff with `(i_surface, j_surface)` and `(i_volume, j_volume, k_volume)`
#
#         i_volume += i_volume_step_i
#         j_volume += j_volume_step_i
#         k_volume += k_volume_step_i
#       end
#       i_volume += i_volume_step_j
#       j_volume += j_volume_step_j
#       k_volume += k_volume_step_j
#     end
@inline function index_to_start_step_3d(index::Symbol, index_range)
    index_begin = first(index_range)
    index_end = last(index_range)

    if index === :begin
        return index_begin, 0, 0
    elseif index === :end
        return index_end, 0, 0
    elseif index === :i_forward
        return index_begin, 1, index_begin - index_end - 1
    elseif index === :i_backward
        return index_end, -1, index_end + 1 - index_begin
    elseif index === :j_forward
        return index_begin, 0, 1
    else # if index === :j_backward
        return index_end, 0, -1
    end
end

# Extract the two varying indices from a symbolic index tuple.
# For example, `surface_indices((:i_forward, :end, :j_forward)) == (:i_forward, :j_forward)`.
@inline function surface_indices(indices::NTuple{3, Symbol})
    i1, i2, i3 = indices
    index = i1
    (index === :begin || index === :end) && return (i2, i3)

    index = i2
    (index === :begin || index === :end) && return (i1, i3)

    # i3 in (:begin, :end)
    return (i1, i2)
end

# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u,
                             mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                             equations, surface_integral, dg::DG)
    @unpack interfaces = cache
    index_range = eachnode(dg)

    @threaded for interface in eachinterface(dg, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the indices of the primary side
        # will always run forwards.
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]

        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        k_primary = k_primary_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    interfaces.u[1, v, i, j, interface] = u[v, i_primary, j_primary,
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
        secondary_element = interfaces.neighbor_ids[2, interface]
        secondary_indices = interfaces.node_indices[2, interface]

        i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1],
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2],
                                                                                           index_range)
        k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3],
                                                                                           index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        k_secondary = k_secondary_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    interfaces.u[2, v, i, j, interface] = u[v, i_secondary, j_secondary,
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

    return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                              nonconservative_terms,
                              equations, surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    @threaded for interface in eachinterface(dg, cache)
        # Get element and side information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1],
                                                                                     index_range)
        j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2],
                                                                                     index_range)
        k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3],
                                                                                     index_range)

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
                                                                                           index_range)
        j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_surface_indices[2],
                                                                                           index_range)
        i_secondary = i_secondary_start
        j_secondary = j_secondary_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                # Get the normal direction from the primary element.
                # Note, contravariant vectors at interfaces in negative coordinate direction
                # are pointing inwards. This is handled by `get_normal_direction`.
                normal_direction = get_normal_direction(primary_direction,
                                                        contravariant_vectors,
                                                        i_primary, j_primary, k_primary,
                                                        primary_element)

                calc_interface_flux!(surface_flux_values, mesh, nonconservative_terms,
                                     equations,
                                     surface_integral, dg, cache,
                                     interface, normal_direction,
                                     i, j, primary_direction, primary_element,
                                     i_secondary, j_secondary, secondary_direction,
                                     secondary_element)

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

    return nothing
end

# Inlined function for interface flux computation for conservative flux terms
@inline function calc_interface_flux!(surface_flux_values,
                                      mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                      nonconservative_terms::False, equations,
                                      surface_integral, dg::DG, cache,
                                      interface_index, normal_direction,
                                      primary_i_node_index, primary_j_node_index,
                                      primary_direction_index, primary_element_index,
                                      secondary_i_node_index, secondary_j_node_index,
                                      secondary_direction_index,
                                      secondary_element_index)
    @unpack u = cache.interfaces
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_i_node_index,
                                       primary_j_node_index, interface_index)

    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, primary_i_node_index, primary_j_node_index,
        primary_direction_index, primary_element_index] = flux_[v]
        surface_flux_values[v, secondary_i_node_index, secondary_j_node_index,
        secondary_direction_index, secondary_element_index] = -flux_[v]
    end
end

# Inlined function for interface flux computation for flux + nonconservative terms
@inline function calc_interface_flux!(surface_flux_values,
                                      mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                      nonconservative_terms::True, equations,
                                      surface_integral, dg::DG, cache,
                                      interface_index, normal_direction,
                                      primary_i_node_index, primary_j_node_index,
                                      primary_direction_index, primary_element_index,
                                      secondary_i_node_index, secondary_j_node_index,
                                      secondary_direction_index,
                                      secondary_element_index)
    @unpack u = cache.interfaces
    surface_flux, nonconservative_flux = surface_integral.surface_flux

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_i_node_index,
                                       primary_j_node_index, interface_index)

    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Compute both nonconservative fluxes
    noncons_primary = nonconservative_flux(u_ll, u_rr, normal_direction, equations)
    noncons_secondary = nonconservative_flux(u_rr, u_ll, normal_direction, equations)

    # Store the flux with nonconservative terms on the primary and secondary elements
    for v in eachvariable(equations)
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        surface_flux_values[v, primary_i_node_index, primary_j_node_index,
        primary_direction_index, primary_element_index] = flux_[v] +
                                                          0.5f0 * noncons_primary[v]
        surface_flux_values[v, secondary_i_node_index, secondary_j_node_index,
        secondary_direction_index, secondary_element_index] = -(flux_[v] +
                                                                0.5f0 *
                                                                noncons_secondary[v])
    end
end

function prolong2boundaries!(cache, u,
                             mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                             equations, surface_integral, dg::DG)
    @unpack boundaries = cache
    index_range = eachnode(dg)

    @threaded for boundary in eachboundary(dg, cache)
        # Copy solution data from the element using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]

        i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                            index_range)
        j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                            index_range)
        k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                            index_range)

        i_node = i_node_start
        j_node = j_node_start
        k_node = k_node_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    boundaries.u[v, i, j, boundary] = u[v, i_node, j_node, k_node,
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

    return nothing
end

function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
                             mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                             equations, surface_integral, dg::DG)
    @unpack boundaries = cache
    @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
    @unpack surface_flux = surface_integral
    index_range = eachnode(dg)

    @threaded for local_index in eachindex(boundary_indexing)
        # Use the local index to get the global boundary index from the
        # pre-sorted list
        boundary = boundary_indexing[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                            index_range)
        j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                            index_range)
        k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                            index_range)

        i_node = i_node_start
        j_node = j_node_start
        k_node = k_node_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                # Extract solution data from boundary container
                u_inner = get_node_vars(boundaries.u, equations, dg, i, j, boundary)

                # Outward-pointing normal direction (not normalized)
                normal_direction = get_normal_direction(direction,
                                                        contravariant_vectors,
                                                        i_node, j_node, k_node, element)

                # Coordinates at boundary node
                x = get_node_coords(node_coordinates, equations, dg,
                                    i_node, j_node, k_node, element)

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
end

function prolong2mortars!(cache, u,
                          mesh::Union{P4estMesh{3}, T8codeMesh{3}}, equations,
                          mortar_l2::LobattoLegendreMortarL2,
                          surface_integral, dg::DGSEM)
    @unpack fstar_tmp_threaded = cache
    @unpack neighbor_ids, node_indices = cache.mortars
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        # Copy solution data from the small elements using "delayed indexing" with
        # a start value and two step sizes to get the correct face and orientation.
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
            element = neighbor_ids[position, mortar]
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.mortars.u[1, v, position, i, j, mortar] = u[v, i_small,
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

        # Buffer to copy solution values of the large element in the correct orientation
        # before interpolating
        u_buffer = cache.u_threaded[Threads.threadid()]
        # temporary buffer for projections
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        # Copy solution of large element face to buffer in the
        # correct orientation
        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        i_large = i_large_start
        j_large = j_large_start
        k_large = k_large_start
        element = neighbor_ids[5, mortar]
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
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, :, mortar),
                                mortar_l2.forward_lower,
                                mortar_l2.forward_lower,
                                u_buffer,
                                fstar_tmp)
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, :, mortar),
                                mortar_l2.forward_upper,
                                mortar_l2.forward_lower,
                                u_buffer,
                                fstar_tmp)
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 3, :, :, mortar),
                                mortar_l2.forward_lower,
                                mortar_l2.forward_upper,
                                u_buffer,
                                fstar_tmp)
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 4, :, :, mortar),
                                mortar_l2.forward_upper,
                                mortar_l2.forward_upper,
                                u_buffer,
                                fstar_tmp)
    end

    return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                           nonconservative_terms, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_threaded, fstar_tmp_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = fstar_threaded[Threads.threadid()]
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

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
            element = neighbor_ids[position, mortar]
            for j in eachnode(dg)
                for i in eachnode(dg)
                    # Get the normal direction on the small element.
                    # Note, contravariant vectors at interfaces in negative coordinate direction
                    # are pointing inwards. This is handled by `get_normal_direction`.
                    normal_direction = get_normal_direction(small_direction,
                                                            contravariant_vectors,
                                                            i_small, j_small, k_small,
                                                            element)

                    calc_mortar_flux!(fstar, mesh, nonconservative_terms, equations,
                                      surface_integral, dg, cache,
                                      mortar, position, normal_direction,
                                      i, j)

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
                i_small += i_small_step_j
                j_small += j_small_step_j
                k_small += k_small_step_j
            end
        end

        # Buffer to interpolate flux values of the large element to before
        # copying in the correct orientation
        u_buffer = cache.u_threaded[Threads.threadid()]

        # in calc_interface_flux!, the interface flux is computed once over each
        # interface using the normal from the "primary" element. The result is then
        # passed back to the "secondary" element, flipping the sign to account for the
        # change in the normal direction. For mortars, this sign flip occurs in
        # "mortar_fluxes_to_elements!" instead.
        mortar_fluxes_to_elements!(surface_flux_values,
                                   mesh, equations, mortar_l2, dg, cache,
                                   mortar, fstar, u_buffer, fstar_tmp)
    end

    return nothing
end

# Inlined version of the mortar flux computation on small elements for conservation fluxes
@inline function calc_mortar_flux!(fstar,
                                   mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                   nonconservative_terms::False, equations,
                                   surface_integral, dg::DG, cache,
                                   mortar_index, position_index, normal_direction,
                                   i_node_index, j_node_index)
    @unpack u = cache.mortars
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, i_node_index,
                                       j_node_index, mortar_index)

    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Copy flux to buffer
    set_node_vars!(fstar, flux, equations, dg, i_node_index, j_node_index,
                   position_index)
end

# Inlined version of the mortar flux computation on small elements for conservation fluxes
# with nonconservative terms
@inline function calc_mortar_flux!(fstar,
                                   mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                   nonconservative_terms::True, equations,
                                   surface_integral, dg::DG, cache,
                                   mortar_index, position_index, normal_direction,
                                   i_node_index, j_node_index)
    @unpack u = cache.mortars
    surface_flux, nonconservative_flux = surface_integral.surface_flux

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, i_node_index,
                                       j_node_index, mortar_index)

    # Compute conservative flux
    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Compute nonconservative flux and add it to the flux scaled by a factor of 0.5 based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    noncons = nonconservative_flux(u_ll, u_rr, normal_direction, equations)
    flux_plus_noncons = flux + 0.5f0 * noncons

    # Copy to buffer
    set_node_vars!(fstar, flux_plus_noncons, equations, dg, i_node_index, j_node_index,
                   position_index)
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                            equations,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache, mortar, fstar, u_buffer,
                                            fstar_tmp)
    @unpack neighbor_ids, node_indices = cache.mortars
    index_range = eachnode(dg)

    # Copy solution small to small
    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    for position in 1:4
        element = neighbor_ids[position, mortar]
        for j in eachnode(dg), i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, i, j, small_direction, element] = fstar[v, i, j,
                                                                               position]
            end
        end
    end

    # Project small fluxes to large element.
    multiply_dimensionwise!(u_buffer,
                            mortar_l2.reverse_lower, mortar_l2.reverse_lower,
                            view(fstar, .., 1),
                            fstar_tmp)
    add_multiply_dimensionwise!(u_buffer,
                                mortar_l2.reverse_upper, mortar_l2.reverse_lower,
                                view(fstar, .., 2),
                                fstar_tmp)
    add_multiply_dimensionwise!(u_buffer,
                                mortar_l2.reverse_lower, mortar_l2.reverse_upper,
                                view(fstar, .., 3),
                                fstar_tmp)
    add_multiply_dimensionwise!(u_buffer,
                                mortar_l2.reverse_upper, mortar_l2.reverse_upper,
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
    large_element = neighbor_ids[5, mortar]
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)
    large_surface_indices = surface_indices(large_indices)

    i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_surface_indices[1],
                                                                           index_range)
    j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_surface_indices[2],
                                                                           index_range)

    # Note that the indices of the small sides will always run forward but
    # the large indices might need to run backwards for flipped sides.
    i_large = i_large_start
    j_large = j_large_start
    for j in eachnode(dg)
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, i_large, j_large, large_direction, large_element] = u_buffer[v,
                                                                                                    i,
                                                                                                    j]
            end
            i_large += i_large_step_i
            j_large += j_large_step_i
        end
        i_large += i_large_step_j
        j_large += j_large_step_j
    end

    return nothing
end

function calc_surface_integral!(du, u,
                                mesh::Union{P4estMesh{3}, T8codeMesh{3}},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM, cache)
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache.elements

    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1, 1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]
    @threaded for element in eachelement(dg, cache)
        for m in eachnode(dg), l in eachnode(dg)
            for v in eachvariable(equations)
                # surface at -x
                du[v, 1, l, m, element] = (du[v, 1, l, m, element] +
                                           surface_flux_values[v, l, m, 1, element] *
                                           factor_1)

                # surface at +x
                du[v, nnodes(dg), l, m, element] = (du[v, nnodes(dg), l, m, element] +
                                                    surface_flux_values[v, l, m, 2,
                                                                        element] *
                                                    factor_2)

                # surface at -y
                du[v, l, 1, m, element] = (du[v, l, 1, m, element] +
                                           surface_flux_values[v, l, m, 3, element] *
                                           factor_1)

                # surface at +y
                du[v, l, nnodes(dg), m, element] = (du[v, l, nnodes(dg), m, element] +
                                                    surface_flux_values[v, l, m, 4,
                                                                        element] *
                                                    factor_2)

                # surface at -z
                du[v, l, m, 1, element] = (du[v, l, m, 1, element] +
                                           surface_flux_values[v, l, m, 5, element] *
                                           factor_1)

                # surface at +z
                du[v, l, m, nnodes(dg), element] = (du[v, l, m, nnodes(dg), element] +
                                                    surface_flux_values[v, l, m, 6,
                                                                        element] *
                                                    factor_2)
            end
        end
    end

    return nothing
end
end # @muladd
