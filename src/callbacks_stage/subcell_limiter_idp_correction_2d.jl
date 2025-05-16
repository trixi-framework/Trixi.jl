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
    @unpack inverse_weights = dg.basis
    @unpack antidiffusive_flux1_L, antidiffusive_flux2_L, antidiffusive_flux1_R, antidiffusive_flux2_R = cache.antidiffusive_fluxes
    @unpack alpha1, alpha2 = dg.volume_integral.limiter.cache.subcell_limiter_coefficients

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, element)

            # Note: antidiffusive_flux1[v, i, xi, element] = antidiffusive_flux2[v, xi, i, element] = 0 for all i in 1:nnodes and xi in {1, nnodes+1}
            alpha_flux1 = (1 - alpha1[i, j, element]) *
                          get_node_vars(antidiffusive_flux1_R, equations, dg,
                                        i, j, element)
            alpha_flux1_ip1 = (1 - alpha1[i + 1, j, element]) *
                              get_node_vars(antidiffusive_flux1_L, equations, dg,
                                            i + 1, j, element)
            alpha_flux2 = (1 - alpha2[i, j, element]) *
                          get_node_vars(antidiffusive_flux2_R, equations, dg,
                                        i, j, element)
            alpha_flux2_jp1 = (1 - alpha2[i, j + 1, element]) *
                              get_node_vars(antidiffusive_flux2_L, equations, dg,
                                            i, j + 1, element)

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

@inline function blend_mortar_flux!(u, semi, equations, dg, t, dt)
    (; mesh, cache) = semi
    (; orientations) = cache.mortars

    (; surface_flux_values, surface_flux_values_high_order) = cache.elements
    (; boundary_interpolation) = dg.basis

    ############################
    # TODO: Calculate blending factor for mortar fluxes
    (; limiting_factor) = cache.mortars
    limiting_factor .= zero(eltype(limiting_factor))
    # limiting_factor = 1 => full DG
    # limiting_factor = 0 => full FV
    #######################

    for mortar in eachmortar(dg, cache)
        if isapprox(limiting_factor[mortar], zero(eltype(limiting_factor)))
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
            flux_local_high_order = view(surface_flux_values_high_order, :, i,
                                         direction_small, lower_element)
            flux_local_low_order = view(surface_flux_values, :, i, direction_small,
                                        lower_element)

            for v in eachvariable(equations)
                u[v, indices_small..., lower_element] += dt * inverse_jacobian_lower *
                                       (factor_small * limiting_factor[mortar] *
                                        (flux_local_high_order[v] - flux_local_low_order[v]))
            end

            flux_local_high_order = view(surface_flux_values_high_order, :, i,
                                         direction_small, upper_element)
            flux_local_low_order = view(surface_flux_values, :, i, direction_small,
                                        upper_element)
            for v in eachvariable(equations)
                u[v, indices_small..., upper_element] += dt * inverse_jacobian_upper *
                                       (factor_small * limiting_factor[mortar] *
                                        (flux_local_high_order[v] - flux_local_low_order[v]))
            end

            flux_local_high_order = view(surface_flux_values_high_order, :, i,
                                         direction_large, large_element)
            flux_local_low_order = view(surface_flux_values, :, i, direction_large,
                                        large_element)
            for v in eachvariable(equations)
                u[v, indices_large..., large_element] += dt * inverse_jacobian_large *
                                       (factor_large * limiting_factor[mortar] *
                                        (flux_local_high_order[v] - flux_local_low_order[v]))
            end
        end
    end

    return nothing
end
end # @muladd
