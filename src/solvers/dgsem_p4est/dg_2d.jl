# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::Union{P4estMesh{2}, T8codeMesh{2}}, equations,
                      mortar_l2::LobattoLegendreMortarL2, uEltype)
    # TODO: Taal performance using different types
    MA2d = MArray{Tuple{nvariables(equations), nnodes(mortar_l2)},
                  uEltype, 2,
                  nvariables(equations) * nnodes(mortar_l2)}
    fstar_upper_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
    fstar_lower_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
    u_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]

    (; fstar_upper_threaded, fstar_lower_threaded, u_threaded)
end

#     index_to_start_step_2d(index::Symbol, index_range)
#
# Given a symbolic `index` and an `indexrange` (usually `eachnode(dg)`),
# return `index_start, index_step`, i.e., a tuple containing
# - `index_start`, an index value to begin a loop
# - `index_step`,  an index step to update during a loop
# The resulting indices translate surface indices to volume indices.
#
# !!! warning
#     This assumes that loops using the return values are written as
#
#     i_volume_start, i_volume_step = index_to_start_step_2d(symbolic_index_i, index_range)
#     j_volume_start, j_volume_step = index_to_start_step_2d(symbolic_index_j, index_range)
#
#     i_volume, j_volume = i_volume_start, j_volume_start
#     for i_surface in index_range
#       # do stuff with `i_surface` and `(i_volume, j_volume)`
#
#       i_volume += i_volume_step
#       j_volume += j_volume_step
#     end
@inline function index_to_start_step_2d(index::Symbol, index_range)
    index_begin = first(index_range)
    index_end = last(index_range)

    if index === :begin
        return index_begin, 0
    elseif index === :end
        return index_end, 0
    elseif index === :i_forward
        return index_begin, 1
    else # if index === :i_backward
        return index_end, -1
    end
end

# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache, u,
                             mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                             equations, surface_integral, dg::DG)
    @unpack interfaces = cache
    index_range = eachnode(dg)

    @threaded for interface in eachinterface(dg, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                interfaces.u[1, v, i, interface] = u[v, i_primary, j_primary,
                                                     primary_element]
            end
            i_primary += i_primary_step
            j_primary += j_primary_step
        end

        # Copy solution data from the secondary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        secondary_element = interfaces.neighbor_ids[2, interface]
        secondary_indices = interfaces.node_indices[2, interface]

        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                interfaces.u[2, v, i, interface] = u[v, i_secondary, j_secondary,
                                                     secondary_element]
            end
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                              nonconservative_terms,
                              equations, surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachinterface(dg, cache)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        # Create the local i,j indexing on the primary element used to pull normal direction information
        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]
        secondary_direction = indices2direction(secondary_indices)

        # Initiate the secondary index to be used in the surface for loop.
        # This index on the primary side will always run forward but
        # the secondary index might need to run backwards for flipped sides.
        if :i_backward in secondary_indices
            node_secondary = index_end
            node_secondary_step = -1
        else
            node_secondary = 1
            node_secondary_step = 1
        end

        for node in eachnode(dg)
            # Get the normal direction on the primary element.
            # Contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(primary_direction,
                                                    contravariant_vectors,
                                                    i_primary, j_primary,
                                                    primary_element)

            calc_interface_flux!(surface_flux_values, mesh, nonconservative_terms,
                                 equations,
                                 surface_integral, dg, cache,
                                 interface, normal_direction,
                                 node, primary_direction, primary_element,
                                 node_secondary, secondary_direction, secondary_element)

            # Increment primary element indices to pull the normal direction
            i_primary += i_primary_step
            j_primary += j_primary_step
            # Increment the surface node index along the secondary element
            node_secondary += node_secondary_step
        end
    end

    return nothing
end

# Inlined version of the interface flux computation for conservation laws
@inline function calc_interface_flux!(surface_flux_values,
                                      mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                      nonconservative_terms::False, equations,
                                      surface_integral, dg::DG, cache,
                                      interface_index, normal_direction,
                                      primary_node_index, primary_direction_index,
                                      primary_element_index,
                                      secondary_node_index, secondary_direction_index,
                                      secondary_element_index)
    @unpack u = cache.interfaces
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_node_index,
                                       interface_index)

    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, primary_node_index, primary_direction_index, primary_element_index] = flux_[v]
        surface_flux_values[v, secondary_node_index, secondary_direction_index, secondary_element_index] = -flux_[v]
    end
end

# Inlined version of the interface flux computation for equations with conservative and nonconservative terms
@inline function calc_interface_flux!(surface_flux_values,
                                      mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                      nonconservative_terms::True, equations,
                                      surface_integral, dg::DG, cache,
                                      interface_index, normal_direction,
                                      primary_node_index, primary_direction_index,
                                      primary_element_index,
                                      secondary_node_index, secondary_direction_index,
                                      secondary_element_index)
    @unpack u = cache.interfaces
    surface_flux, nonconservative_flux = surface_integral.surface_flux

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_node_index,
                                       interface_index)

    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Compute both nonconservative fluxes
    noncons_primary = nonconservative_flux(u_ll, u_rr, normal_direction, equations)
    noncons_secondary = nonconservative_flux(u_rr, u_ll, normal_direction, equations)

    # Store the flux with nonconservative terms on the primary and secondary elements
    for v in eachvariable(equations)
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        surface_flux_values[v, primary_node_index, primary_direction_index, primary_element_index] = (flux_[v] +
                                                                                                      0.5f0 *
                                                                                                      noncons_primary[v])
        surface_flux_values[v, secondary_node_index, secondary_direction_index, secondary_element_index] = -(flux_[v] +
                                                                                                             0.5f0 *
                                                                                                             noncons_secondary[v])
    end
end

function prolong2boundaries!(cache, u,
                             mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                             equations, surface_integral, dg::DG)
    @unpack boundaries = cache
    index_range = eachnode(dg)

    @threaded for boundary in eachboundary(dg, cache)
        # Copy solution data from the element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for i in eachnode(dg)
            for v in eachvariable(equations)
                boundaries.u[v, i, boundary] = u[v, i_node, j_node, element]
            end
            i_node += i_node_step
            j_node += j_node_step
        end
    end

    return nothing
end

function calc_boundary_flux!(cache, t, boundary_condition::BC, boundary_indexing,
                             mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                             equations, surface_integral, dg::DG) where {BC}
    @unpack boundaries = cache
    @unpack surface_flux_values = cache.elements
    index_range = eachnode(dg)

    @threaded for local_index in eachindex(boundary_indexing)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = boundary_indexing[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node in eachnode(dg)
            calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                mesh, have_nonconservative_terms(equations),
                                equations, surface_integral, dg, cache,
                                i_node, j_node,
                                node, direction, element, boundary)

            i_node += i_node_step
            j_node += j_node_step
        end
    end
end

# inlined version of the boundary flux calculation along a physical interface
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                     nonconservative_terms::False, equations,
                                     surface_integral, dg::DG, cache,
                                     i_index, j_index,
                                     node_index, direction_index, element_index,
                                     boundary_index)
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    @unpack surface_flux = surface_integral

    # Extract solution data from boundary container
    u_inner = get_node_vars(boundaries.u, equations, dg, node_index, boundary_index)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, element_index)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg, i_index, j_index,
                        element_index)

    flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        surface_flux_values[v, node_index, direction_index, element_index] = flux_[v]
    end
end

# inlined version of the boundary flux with nonconservative terms calculation along a physical interface
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                     nonconservative_terms::True, equations,
                                     surface_integral, dg::DG, cache,
                                     i_index, j_index,
                                     node_index, direction_index, element_index,
                                     boundary_index)
    @unpack boundaries = cache
    @unpack node_coordinates, contravariant_vectors = cache.elements
    surface_flux, nonconservative_flux = surface_integral.surface_flux

    # Extract solution data from boundary container
    u_inner = get_node_vars(boundaries.u, equations, dg, node_index, boundary_index)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction_index, contravariant_vectors,
                                            i_index, j_index, element_index)

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations, dg, i_index, j_index,
                        element_index)

    # Call pointwise numerical flux function for the conservative part
    # in the normal direction on the boundary
    flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

    # Compute pointwise nonconservative numerical flux at the boundary.
    noncons_ = boundary_condition(u_inner, normal_direction, x, t, nonconservative_flux,
                                  equations)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations)
        # Note the factor 0.5 necessary for the nonconservative fluxes based on
        # the interpretation of global SBP operators coupled discontinuously via
        # central fluxes/SATs
        surface_flux_values[v, node_index, direction_index, element_index] = flux_[v] +
                                                                             0.5f0 *
                                                                             noncons_[v]
    end
end

function prolong2mortars!(cache, u,
                          mesh::Union{P4estMesh{2}, T8codeMesh{2}}, equations,
                          mortar_l2::LobattoLegendreMortarL2,
                          surface_integral, dg::DGSEM)
    @unpack neighbor_ids, node_indices = cache.mortars
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        # Copy solution data from the small elements using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        small_indices = node_indices[1, mortar]

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    cache.mortars.u[1, v, position, i, mortar] = u[v, i_small, j_small,
                                                                   element]
                end
                i_small += i_small_step
                j_small += j_small_step
            end
        end

        # Buffer to copy solution values of the large element in the correct orientation
        # before interpolating
        u_buffer = cache.u_threaded[Threads.threadid()]

        # Copy solution of large element face to buffer in the
        # correct orientation
        large_indices = node_indices[2, mortar]

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_large = i_large_start
        j_large = j_large_start
        element = neighbor_ids[3, mortar]
        for i in eachnode(dg)
            for v in eachvariable(equations)
                u_buffer[v, i] = u[v, i_large, j_large, element]
            end
            i_large += i_large_step
            j_large += j_large_step
        end

        # Interpolate large element face data from buffer to small face locations
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, mortar),
                                mortar_l2.forward_lower,
                                u_buffer)
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, mortar),
                                mortar_l2.forward_upper,
                                u_buffer)
    end

    return nothing
end

function calc_mortar_flux!(surface_flux_values,
                           mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                           nonconservative_terms, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_upper_threaded, fstar_lower_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = (fstar_lower_threaded[Threads.threadid()],
                 fstar_upper_threaded[Threads.threadid()])

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]
            for node in eachnode(dg)
                # Get the normal direction on the small element.
                # Note, contravariant vectors at interfaces in negative coordinate direction
                # are pointing inwards. This is handled by `get_normal_direction`.
                normal_direction = get_normal_direction(small_direction,
                                                        contravariant_vectors,
                                                        i_small, j_small, element)

                calc_mortar_flux!(fstar, mesh, nonconservative_terms, equations,
                                  surface_integral, dg, cache,
                                  mortar, position, normal_direction,
                                  node)

                i_small += i_small_step
                j_small += j_small_step
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
                                   mortar, fstar, u_buffer)
    end

    return nothing
end

# Inlined version of the mortar flux computation on small elements for conservation laws
@inline function calc_mortar_flux!(fstar,
                                   mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                   nonconservative_terms::False, equations,
                                   surface_integral, dg::DG, cache,
                                   mortar_index, position_index, normal_direction,
                                   node_index)
    @unpack u = cache.mortars
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, node_index,
                                       mortar_index)

    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Copy flux to buffer
    set_node_vars!(fstar[position_index], flux, equations, dg, node_index)
end

# Inlined version of the mortar flux computation on small elements for equations with conservative and
# nonconservative terms
@inline function calc_mortar_flux!(fstar,
                                   mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                   nonconservative_terms::True, equations,
                                   surface_integral, dg::DG, cache,
                                   mortar_index, position_index, normal_direction,
                                   node_index)
    @unpack u = cache.mortars
    surface_flux, nonconservative_flux = surface_integral.surface_flux

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, node_index,
                                       mortar_index)

    # Compute conservative flux
    flux = surface_flux(u_ll, u_rr, normal_direction, equations)

    # Compute nonconservative flux and add it to the conservative flux.
    # The nonconservative flux is scaled by a factor of 0.5 based on
    # the interpretation of global SBP operators coupled discontinuously via
    # central fluxes/SATs
    noncons = nonconservative_flux(u_ll, u_rr, normal_direction, equations)

    flux_plus_noncons = flux + 0.5f0 * noncons

    # Copy to buffer
    set_node_vars!(fstar[position_index], flux_plus_noncons, equations, dg, node_index)
end

@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                            equations,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache, mortar, fstar, u_buffer)
    @unpack neighbor_ids, node_indices = cache.mortars

    # Copy solution small to small
    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    for position in 1:2
        element = neighbor_ids[position, mortar]
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, i, small_direction, element] = fstar[position][v,
                                                                                      i]
            end
        end
    end

    # Project small fluxes to large element.
    multiply_dimensionwise!(u_buffer,
                            mortar_l2.reverse_upper, fstar[2],
                            mortar_l2.reverse_lower, fstar[1])

    # The flux is calculated in the outward direction of the small elements,
    # so the sign must be switched to get the flux in outward direction
    # of the large element.
    # The contravariant vectors of the large element (and therefore the normal
    # vectors of the large element as well) are twice as large as the
    # contravariant vectors of the small elements. Therefore, the flux needs
    # to be scaled by a factor of 2 to obtain the flux of the large element.
    u_buffer .*= -2

    # Copy interpolated flux values from buffer to large element face in the
    # correct orientation.
    # Note that the index of the small sides will always run forward but
    # the index of the large side might need to run backwards for flipped sides.
    large_element = neighbor_ids[3, mortar]
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)

    if :i_backward in large_indices
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, end + 1 - i, large_direction, large_element] = u_buffer[v,
                                                                                               i]
            end
        end
    else
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, i, large_direction, large_element] = u_buffer[v,
                                                                                     i]
            end
        end
    end

    return nothing
end

function calc_surface_integral!(du, u,
                                mesh::Union{P4estMesh{2}, T8codeMesh{2}},
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
        for l in eachnode(dg)
            for v in eachvariable(equations)
                # surface at -x
                du[v, 1, l, element] = (du[v, 1, l, element] +
                                        surface_flux_values[v, l, 1, element] *
                                        factor_1)

                # surface at +x
                du[v, nnodes(dg), l, element] = (du[v, nnodes(dg), l, element] +
                                                 surface_flux_values[v, l, 2, element] *
                                                 factor_2)

                # surface at -y
                du[v, l, 1, element] = (du[v, l, 1, element] +
                                        surface_flux_values[v, l, 3, element] *
                                        factor_1)

                # surface at +y
                du[v, l, nnodes(dg), element] = (du[v, l, nnodes(dg), element] +
                                                 surface_flux_values[v, l, 4, element] *
                                                 factor_2)
            end
        end
    end

    return nothing
end
end # @muladd
