# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function perform_idp_correction!(u, dt,
                                 mesh::Union{TreeMesh{2}, StructuredMesh{2},
                                             P4estMesh{2}},
                                 equations, dg, cache)
    @unpack inverse_weights = dg.basis # Plays role of inverse DG-subcell sizes
    @unpack antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R = cache.antidiffusive_fluxes
    @unpack alpha = dg.volume_integral.limiter.cache.subcell_limiter_coefficients

    # The following code implements the IDP correction in flux-differencing form:
    # u[v, i, j, element] += dt * -inverse_jacobian[i, j, element] *
    #    (inverse_weights[i] *
    #       ((1 - alpha_1_ip1) * antidiffusive_flux1_ip1[v] - (1 - alpha_1) * antidiffusive_flux1[v]) +
    #     inverse_weights[j] *
    #       ((1 - alpha_2_jp1) * antidiffusive_flux2_jp1[v] - (1 - alpha_2) * antidiffusive_flux2[v]))
    # with
    # alpha_1 = max(alpha[i - 1, j, element], alpha[i, j, element]),
    # alpha_1_ip1 = max(alpha[i, j, element], alpha[i + 1, j, element])
    # and equivalently for alpha_2 and alpha_2_jp1.

    # For LGL nodes, the high-order and low-order fluxes at element interfaces are equal
    # and therefore, the antidiffusive fluxes are zero there.
    # To avoid adding zeros and speed up the simulation, we directly loop over the subcell
    # interfaces.

    @threaded for element in eachelement(dg, cache)
        # Perform correction in 1st/x-direction
        for j in eachnode(dg), i in 2:nnodes(dg)
            # Subcell interface between nodes (i - 1, j) and (i, j)
            alpha1 = max(alpha[i - 1, j, element], alpha[i, j, element])

            # Apply to right node (i, j)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, element)
            flux1 = get_node_vars(antidiffusive_flux1_R, equations, dg,
                                  i, j, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[i] * (1 - alpha1)
            multiply_add_to_node_vars!(u, dg_factor, flux1,
                                       equations, dg, i, j, element)

            # Apply to left node (i - 1, j)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i - 1, j, element)
            flux1_ip1 = get_node_vars(antidiffusive_flux1_L, equations, dg,
                                      i, j, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[i - 1] * (1 - alpha1)
            multiply_add_to_node_vars!(u, dg_factor, flux1_ip1,
                                       equations, dg, i - 1, j, element)
        end

        # Perform correction in 2nd/y-direction
        for j in 2:nnodes(dg), i in eachnode(dg)
            # Subcell interface between nodes (i, j - 1) and (i, j)
            alpha2 = max(alpha[i, j - 1, element], alpha[i, j, element])

            # Apply to right node (i, j)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, element)
            flux2 = get_node_vars(antidiffusive_flux2_R, equations, dg,
                                  i, j, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[j] * (1 - alpha2)
            multiply_add_to_node_vars!(u, dg_factor, flux2,
                                       equations, dg, i, j, element)

            # Apply to left node (i, j - 1)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j - 1, element)
            flux2_jp1 = get_node_vars(antidiffusive_flux2_L, equations, dg,
                                      i, j, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[j - 1] * (1 - alpha2)
            multiply_add_to_node_vars!(u, dg_factor, flux2_jp1,
                                       equations, dg, i, j - 1, element)
        end
    end

    return nothing
end

function perform_idp_mortar_correction(u, dt, mesh::TreeMesh{2}, equations, dg, cache)
    (; orientations, limiting_factor) = cache.mortars

    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes

    (; inverse_weights) = dg.basis
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    for mortar in eachmortar(dg, cache)
        if isapprox(limiting_factor[mortar], one(eltype(limiting_factor)))
            continue
        end

        if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
            if orientations[mortar] == 1
                direction_small = 1
                direction_large = 2
            else
                direction_small = 3
                direction_large = 4
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_small = factor
            factor_large = -factor
        else # large_sides[mortar] == 2 -> small elements on left side
            if orientations[mortar] == 1
                direction_small = 2
                direction_large = 1
            else
                direction_small = 4
                direction_large = 3
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            factor_large = factor
            factor_small = -factor
        end

        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                end
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                end
            end

            # small elements
            for small_element_index in 1:2
                small_element = cache.mortars.neighbor_ids[small_element_index, mortar]
                inverse_jacobian_small = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                              mesh, indices_small...,
                                                              small_element)

                flux_small_high_order = get_node_vars(surface_flux_values_high_order,
                                                      equations, dg,
                                                      i, direction_small, small_element)
                flux_small_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                     i, direction_small, small_element)
                flux_difference_small = factor_small *
                                        (flux_small_high_order .- flux_small_low_order)

                multiply_add_to_node_vars!(u,
                                           dt * inverse_jacobian_small *
                                           (1 - limiting_factor[mortar]),
                                           flux_difference_small, equations, dg,
                                           indices_small..., small_element)
            end

            # large element
            large_element = cache.mortars.neighbor_ids[3, mortar]
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_large...,
                                                          large_element)

            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, direction_large,
                                                  large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg, i,
                                                 direction_large, large_element)
            flux_difference_large = factor_large *
                                    (flux_large_high_order .- flux_large_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_large *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_large, equations, dg,
                                       indices_large..., large_element)
        end
    end

    return nothing
end

function perform_idp_mortar_correction(u, dt, mesh::P4estMesh{2}, equations, dg, cache)
    (; neighbor_ids, node_indices, limiting_factor) = cache.mortars

    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis
    index_range = eachnode(dg)

    # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
    # This sign switch is directly applied to the boundary interpolation factors here.
    factor = -inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1

    for mortar in eachmortar(dg, cache)
        if isapprox(limiting_factor[mortar], one(eltype(limiting_factor)))
            continue
        end
        large_element = neighbor_ids[3, mortar]
        upper_element = neighbor_ids[2, mortar]
        lower_element = neighbor_ids[1, mortar]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]
        small_direction = indices2direction(small_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        large_indices = node_indices[2, mortar]
        large_direction = indices2direction(large_indices)

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_small = i_small_start
        j_small = j_small_start
        i_large = i_large_start
        j_large = j_large_start
        for i in eachnode(dg)
            inverse_jacobian_upper = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_small, j_small,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_small, j_small,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, i_large, j_large,
                                                          large_element)

            # lower element
            flux_lower_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, small_direction, lower_element)
            flux_lower_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, small_direction, lower_element)
            flux_difference_lower = factor *
                                    (flux_lower_high_order .- flux_lower_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_lower *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_lower, equations, dg,
                                       i_small, j_small, lower_element)

            flux_upper_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, small_direction, upper_element)
            flux_upper_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, small_direction, upper_element)
            flux_difference_upper = factor *
                                    (flux_upper_high_order .- flux_upper_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_upper *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_upper, equations, dg,
                                       i_small, j_small, upper_element)

            flux_large_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg,
                                                  i, large_direction, large_element)
            flux_large_low_order = get_node_vars(surface_flux_values, equations, dg,
                                                 i, large_direction, large_element)
            flux_difference_large = factor *
                                    (flux_large_high_order .- flux_large_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_large *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_large, equations, dg,
                                       i_large, j_large, large_element)

            i_small += i_small_step
            j_small += j_small_step
            i_large += i_large_step
            j_large += j_large_step
        end
    end

    return nothing
end
end # @muladd
