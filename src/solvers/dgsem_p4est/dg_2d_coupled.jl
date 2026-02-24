# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    prolong2coupledmortars!(cache, u, mesh::P4estMeshView, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)

Prolong solution values from elements to coupled mortar faces.
For large elements, interpolate to small element positions.
For small elements, copy directly.

This function only handles LOCAL elements - remote elements will be filled
during flux computation from the global solution vector.
"""
function prolong2coupledmortars!(cache, u, mesh::P4estMeshView{2}, equations,
                                mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)
    @unpack node_indices = cache.coupled_mortars
    index_range = eachnode(dg)

    @threaded for mortar in eachcoupledmortar(dg, cache)
        local_neighbor_ids = cache.coupled_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.coupled_mortars.local_neighbor_positions[mortar]

        # Get start value and step size for indices
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 3 # Large element
                # Buffer to copy solution values
                u_buffer = cache.u_threaded[Threads.threadid()]
                i_large = i_large_start
                j_large = j_large_start
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        u_buffer[v, i] = u[v, i_large, j_large, element]
                    end
                    i_large += i_large_step
                    j_large += j_large_step
                end

                # Interpolate large element face data to small face locations
                multiply_dimensionwise!(view(cache.coupled_mortars.u, 2, :, 1, :,
                                            mortar),
                                       mortar_l2.forward_lower, u_buffer)
                multiply_dimensionwise!(view(cache.coupled_mortars.u, 2, :, 2, :,
                                            mortar),
                                       mortar_l2.forward_upper, u_buffer)
            else # position in (1, 2) - Small element
                # Copy solution data from the small elements
                i_small = i_small_start
                j_small = j_small_start
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        cache.coupled_mortars.u[1, v, position, i, mortar] = u[v,
                                                                                i_small,
                                                                                j_small,
                                                                                element]
                    end
                    i_small += i_small_step
                    j_small += j_small_step
                end
            end
        end
    end

    return nothing
end

"""
    fill_coupled_mortar_from_global!(coupled_mortars, mortar, u_global, semi,
                                     mesh, equations, dg, mortar_l2)

Fill remote element data in coupled mortar from global solution vector.
The global solution vector `u_global` uses per-view offsets with layout
`[var, i_node, j_node, local_element]` within each view, accessed via
`semi.element_offset[view_id]`.
"""
function fill_coupled_mortar_from_global!(coupled_mortars, mortar, u_global, semi,
                                         mesh, equations, dg, mortar_l2)
    global_neighbor_ids = coupled_mortars.global_neighbor_ids[mortar]
    local_neighbor_positions = coupled_mortars.local_neighbor_positions[mortar]
    node_indices = coupled_mortars.node_indices[:, mortar]

    # Determine which positions are remote (not in local list)
    all_positions = (1, 2, 3)  # small_1, small_2, large in 2D
    remote_positions = setdiff(all_positions, local_neighbor_positions)

    n_nodes = nnodes(dg)
    n_vars = nvariables(equations)
    index_range = eachnode(dg)

    for (global_id, pos) in zip(global_neighbor_ids, all_positions)
        if pos in remote_positions
            # Determine which view owns this element and get the local element ID
            view_id = semi.mesh_ids[global_id]
            semi_other = semi.semis[view_id]
            local_id = global_cell_id_to_local(global_id, semi_other.mesh)
            offset = semi.element_offset[view_id]

            if pos <= 2  # Small element
                small_indices = node_indices[1]
                i_start, i_step = index_to_start_step_2d(small_indices[1], index_range)
                j_start, j_step = index_to_start_step_2d(small_indices[2], index_range)

                i_node = i_start
                j_node = j_start
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        coupled_mortars.u[1, v, pos, i, mortar] = u_global[offset +
                                                                           (v - 1) +
                                                                           n_vars * (i_node - 1) +
                                                                           n_vars * n_nodes * (j_node - 1) +
                                                                           n_vars * n_nodes^2 * (local_id - 1)]
                    end
                    i_node += i_step
                    j_node += j_step
                end
            else  # pos == 3: Large element
                large_indices = node_indices[2]
                i_start, i_step = index_to_start_step_2d(large_indices[1], index_range)
                j_start, j_step = index_to_start_step_2d(large_indices[2], index_range)

                u_buffer = zeros(eltype(u_global), n_vars, n_nodes)
                i_node = i_start
                j_node = j_start
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        u_buffer[v, i] = u_global[offset +
                                                  (v - 1) +
                                                  n_vars * (i_node - 1) +
                                                  n_vars * n_nodes * (j_node - 1) +
                                                  n_vars * n_nodes^2 * (local_id - 1)]
                    end
                    i_node += i_step
                    j_node += j_step
                end

                # Interpolate large face data to small element positions
                multiply_dimensionwise!(view(coupled_mortars.u, 2, :, 1, :, mortar),
                                       mortar_l2.forward_lower, u_buffer)
                multiply_dimensionwise!(view(coupled_mortars.u, 2, :, 2, :, mortar),
                                       mortar_l2.forward_upper, u_buffer)
            end
        end
    end

    return nothing
end

"""
    calc_coupled_mortar_flux!(surface_flux_values, mesh, equations, mortar_l2,
                              surface_integral, dg, cache, u_global, semi)

Compute fluxes at coupled mortar boundaries and distribute to elements.
"""
function calc_coupled_mortar_flux!(surface_flux_values, mesh::P4estMeshView{2},
                                  have_nonconservative_terms,
                                  equations, mortar_l2::LobattoLegendreMortarL2,
                                  surface_integral, dg::DG, cache,
                                  u_global, semi)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.coupled_mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_primary_upper_threaded, fstar_primary_lower_threaded, u_threaded = cache
    @unpack surface_flux = surface_integral
    index_range = eachnode(dg)

    @threaded for mortar in eachcoupledmortar(dg, cache)
        # Fill remote element data from global solution
        fill_coupled_mortar_from_global!(cache.coupled_mortars, mortar, u_global,
                                        semi, mesh, equations, dg, mortar_l2)

        # Choose thread-specific pre-allocated containers
        fstar_primary_upper = fstar_primary_upper_threaded[Threads.threadid()]
        fstar_primary_lower = fstar_primary_lower_threaded[Threads.threadid()]
        u_buffer = u_threaded[Threads.threadid()]

        # Get indices
        small_indices = node_indices[1, mortar]
        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        # Compute fluxes at each small element position
        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            for i in eachnode(dg)
                # Get solution values on both sides using get_surface_node_vars
                # which handles the [side, v, ...] indexing correctly
                u_small, u_large = get_surface_node_vars(cache.coupled_mortars.u,
                                                         equations, dg,
                                                         position, i, mortar)

                # Get normal direction
                normal_direction = get_normal_direction(cache.coupled_mortars.normal_directions,
                                                       i, position, mortar)

                # Compute flux
                flux = surface_flux(u_small, u_large, normal_direction, equations)

                # Store flux
                if position == 1
                    set_node_vars!(fstar_primary_lower, flux, equations, dg, i)
                else
                    set_node_vars!(fstar_primary_upper, flux, equations, dg, i)
                end

                i_small += i_small_step
                j_small += j_small_step
            end
        end

        # Distribute fluxes to elements
        coupled_mortar_fluxes_to_elements!(surface_flux_values,
                                           mesh, equations, mortar_l2, dg, cache,
                                           mortar, fstar_primary_lower, fstar_primary_upper,
                                           u_buffer)
    end

    return nothing
end

"""
    get_normal_direction(normal_directions, i, position, mortar)

Extract normal direction for a specific node and position at a mortar.
"""
@inline function get_normal_direction(normal_directions, i, position, mortar)
    SVector(normal_directions[1, i, position, mortar],
            normal_directions[2, i, position, mortar])
end

"""
    coupled_mortar_fluxes_to_elements!(surface_flux_values, mesh, equations, mortar_l2,
                                       dg, cache, mortar, fstar_lower, fstar_upper,
                                       u_buffer)

Distribute mortar fluxes to participating elements for coupled mortars.
This is adapted from the regular mortar flux distribution but handles the coupled case
where not all elements are local.
"""
function coupled_mortar_fluxes_to_elements!(surface_flux_values, mesh, equations,
                                            mortar_l2, dg, cache, mortar,
                                            fstar_lower, fstar_upper,
                                            u_buffer)
    local_neighbor_ids = cache.coupled_mortars.local_neighbor_ids[mortar]
    local_neighbor_positions = cache.coupled_mortars.local_neighbor_positions[mortar]
    node_indices = cache.coupled_mortars.node_indices[:, mortar]

    # Get directions for flux storage
    small_indices = node_indices[1]
    large_indices = node_indices[2]
    small_direction = indices2direction(small_indices)
    large_direction = indices2direction(large_indices)

    for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
        if position == 3 # Large element
            # Project small fluxes to large element using reverse operators
            multiply_dimensionwise!(u_buffer,
                                   mortar_l2.reverse_upper, fstar_upper,
                                   mortar_l2.reverse_lower, fstar_lower)

            # The flux is calculated in the outward direction of the small elements,
            # so the sign must be switched to get the flux in outward direction
            # of the large element.
            # The contravariant vectors of the large element are twice as large as
            # those of the small elements, so scale by 2.
            u_buffer .*= -2

            # Copy interpolated flux to large element face in correct orientation
            if :i_backward in large_indices
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, end + 1 - i, large_direction, element] = u_buffer[v,
                                                                                                 i]
                    end
                end
            else
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i, large_direction, element] = u_buffer[v, i]
                    end
                end
            end
        else # position in (1, 2) - Small element
            # Select appropriate flux
            fstar = position == 1 ? fstar_lower : fstar_upper

            # Copy flux directly to small element face
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    surface_flux_values[v, i, small_direction, element] = fstar[v, i]
                end
            end
        end
    end

    return nothing
end

"""
    zero_coupled_mortar_surface_flux!(surface_flux_values, mesh, equations, dg, cache)

Zero out the surface flux values for coupled mortar faces.
This must be called BEFORE the regular rhs! to prevent stale flux values
from being applied by the regular surface integral.
"""
function zero_coupled_mortar_surface_flux!(surface_flux_values, mesh::P4estMeshView{2},
                                           equations, dg, cache)
    for mortar in eachcoupledmortar(dg, cache)
        local_neighbor_ids = cache.coupled_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.coupled_mortars.local_neighbor_positions[mortar]
        node_indices = cache.coupled_mortars.node_indices[:, mortar]

        small_indices = node_indices[1]
        large_indices = node_indices[2]
        small_direction = indices2direction(small_indices)
        large_direction = indices2direction(large_indices)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 3 # Large element
                direction = large_direction
            else # Small element
                direction = small_direction
            end

            # Zero out surface flux for this element's coupled mortar face
            for i in eachnode(dg)
                for v in eachvariable(equations)
                    surface_flux_values[v, i, direction, element] = zero(eltype(surface_flux_values))
                end
            end
        end
    end

    return nothing
end

"""
    calc_coupled_mortar_surface_integral!(du, u, mesh, equations,
                                          surface_integral, dg, cache)

Apply the surface integral contribution from coupled mortars to `du`.
This is called after `calc_coupled_mortar_flux!` since the regular surface
integral has already been computed before coupled mortar fluxes were available.
"""
function calc_coupled_mortar_surface_integral!(du, u,
                                               mesh::P4estMeshView{2},
                                               equations,
                                               surface_integral::SurfaceIntegralWeakForm,
                                               dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, inverse_jacobian = cache.elements

    # Use the same factor as the regular calc_surface_integral! (inverse_weights[1]).
    # For LGL nodes, inverse_weights[1] == inverse_weights[nnodes] due to symmetry.
    factor_1 = inverse_weights[1]
    factor_2 = inverse_weights[nnodes(dg)]

    # Only apply surface integral for elements that participate in coupled mortars
    for mortar in eachcoupledmortar(dg, cache)
        local_neighbor_ids = cache.coupled_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.coupled_mortars.local_neighbor_positions[mortar]
        node_indices = cache.coupled_mortars.node_indices[:, mortar]

        small_indices = node_indices[1]
        large_indices = node_indices[2]
        small_direction = indices2direction(small_indices)
        large_direction = indices2direction(large_indices)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 3 # Large element
                direction = large_direction
            else # Small element
                direction = small_direction
            end

            # Apply surface integral for this element's coupled mortar face
            # Note: The regular rhs! already applied apply_jacobian! to du, so we need
            # to also scale our contribution by -inverse_jacobian to match.
            for l in eachnode(dg)
                for v in eachvariable(equations)
                    # surface at -x
                    if direction == 1
                        # At -x face, nodes are at (1, l)
                        jacobian_factor = -inverse_jacobian[1, l, element]
                        du[v, 1, l, element] = (du[v, 1, l, element] +
                                               surface_flux_values[v, l, direction, element] *
                                               factor_1 * jacobian_factor)
                    # surface at +x
                    elseif direction == 2
                        # At +x face, nodes are at (nnodes, l)
                        jacobian_factor = -inverse_jacobian[nnodes(dg), l, element]
                        du[v, nnodes(dg), l, element] = (du[v, nnodes(dg), l, element] +
                                                        surface_flux_values[v, l, direction, element] *
                                                        factor_2 * jacobian_factor)
                    # surface at -y
                    elseif direction == 3
                        # At -y face, nodes are at (l, 1)
                        jacobian_factor = -inverse_jacobian[l, 1, element]
                        du[v, l, 1, element] = (du[v, l, 1, element] +
                                               surface_flux_values[v, l, direction, element] *
                                               factor_1 * jacobian_factor)
                    # surface at +y
                    else # direction == 4
                        # At +y face, nodes are at (l, nnodes)
                        jacobian_factor = -inverse_jacobian[l, nnodes(dg), element]
                        du[v, l, nnodes(dg), element] = (du[v, l, nnodes(dg), element] +
                                                        surface_flux_values[v, l, direction, element] *
                                                        factor_2 * jacobian_factor)
                    end
                end
            end
        end
    end

    return nothing
end

end # @muladd
