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

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, element)

            # Note: For LGL nodes, the high-order and low-order fluxes at element interfaces are equal.
            # Therefore, the antidiffusive fluxes are zero.
            # To avoid accessing zero entries, we directly use zero vectors instead.
            if i > 1 # Not at "left" boundary node
                alpha1 = max(alpha[i - 1, j, element], alpha[i, j, element])
                alpha_flux1 = (1 - alpha1) *
                              get_node_vars(antidiffusive_flux1_R, equations, dg,
                                            i, j, element)
            else # At "left" boundary node
                alpha_flux1 = zero(SVector{nvariables(equations), eltype(u)})
            end
            if i < nnodes(dg) # Not at "right" boundary node
                alpha1_ip1 = max(alpha[i, j, element], alpha[i + 1, j, element])
                alpha_flux1_ip1 = (1 - alpha1_ip1) *
                                  get_node_vars(antidiffusive_flux1_L, equations, dg,
                                                i + 1, j, element)
            else # At "right" boundary node
                alpha_flux1_ip1 = zero(SVector{nvariables(equations), eltype(u)})
            end
            if j > 1 # Not at "bottom" boundary node
                alpha2 = max(alpha[i, j - 1, element], alpha[i, j, element])
                alpha_flux2 = (1 - alpha2) *
                              get_node_vars(antidiffusive_flux2_R, equations, dg,
                                            i, j, element)
            else # At "bottom" boundary node
                alpha_flux2 = zero(SVector{nvariables(equations), eltype(u)})
            end
            if j < nnodes(dg) # Not at "top" boundary node
                alpha2_jp1 = max(alpha[i, j, element], alpha[i, j + 1, element])
                alpha_flux2_jp1 = (1 - alpha2_jp1) *
                                  get_node_vars(antidiffusive_flux2_L, equations, dg,
                                                i, j + 1, element)
            else # At "top" boundary node
                alpha_flux2_jp1 = zero(SVector{nvariables(equations), eltype(u)})
            end

            for v in eachvariable(equations)
                u[v, i, j, element] += dt * inverse_jacobian *
                                       (inverse_weights[i] *
                                        (alpha_flux1_ip1[v] - alpha_flux1[v]) +
                                        inverse_weights[j] *
                                        (alpha_flux2_jp1[v] - alpha_flux2[v]))
            end
        end
    end

    return nothing
end

function perform_idp_mortar_correction(u, dt, mesh::TreeMesh{2}, equations, dg, cache)
    (; orientations, limiting_factor) = cache.mortars

    (; surface_flux_values) = cache.elements
    (; surface_flux_values_high_order) = cache.antidiffusive_fluxes
    (; boundary_interpolation) = dg.basis

    for mortar in eachmortar(dg, cache)
        if isapprox(limiting_factor[mortar], one(eltype(limiting_factor)))
            continue
        end
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        for i in eachnode(dg)
            if cache.mortars.large_sides[mortar] == 1 # -> small elements on right side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (1, i)
                    indices_large = (nnodes(dg), i)
                    direction_small = 1
                    direction_large = 2
                else
                    # L2 mortars in y-direction
                    indices_small = (i, 1)
                    indices_large = (i, nnodes(dg))
                    direction_small = 3
                    direction_large = 4
                end
                factor_small = boundary_interpolation[1, 1]
                factor_large = -boundary_interpolation[nnodes(dg), 2]
            else # large_sides[mortar] == 2 -> small elements on left side
                if orientations[mortar] == 1
                    # L2 mortars in x-direction
                    indices_small = (nnodes(dg), i)
                    indices_large = (1, i)
                    direction_small = 2
                    direction_large = 1
                else
                    # L2 mortars in y-direction
                    indices_small = (i, nnodes(dg))
                    indices_large = (i, 1)
                    direction_small = 4
                    direction_large = 3
                end
                factor_large = boundary_interpolation[1, 1]
                factor_small = -boundary_interpolation[nnodes(dg), 2]
            end
            # In `apply_jacobian`, `du` is multiplied with inverse jacobian and a negative sign.
            # This sign switch is directly applied to the boundary interpolation factors here.
            inverse_jacobian_upper = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_small...,
                                                          upper_element)
            inverse_jacobian_lower = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_small...,
                                                          lower_element)
            inverse_jacobian_large = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                          mesh, indices_large...,
                                                          large_element)

            # lower element
            flux_lower_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, direction_small,
                                                  lower_element)
            flux_lower_low_order = get_node_vars(surface_flux_values, equations, dg, i,
                                                 direction_small, lower_element)
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order .- flux_lower_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_lower *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_lower, equations, dg,
                                       indices_small..., lower_element)

            flux_upper_high_order = get_node_vars(surface_flux_values_high_order,
                                                  equations, dg, i, direction_small,
                                                  upper_element)
            flux_upper_low_order = get_node_vars(surface_flux_values, equations, dg, i,
                                                 direction_small, upper_element)
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order .- flux_upper_low_order)

            multiply_add_to_node_vars!(u,
                                       dt * inverse_jacobian_upper *
                                       (1 - limiting_factor[mortar]),
                                       flux_difference_upper, equations, dg,
                                       indices_small..., upper_element)

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
end # @muladd
