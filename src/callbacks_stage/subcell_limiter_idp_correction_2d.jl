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

@inline function calc_limiting_factor!(u, semi, t, dt)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    (; limiting_factor, orientations) = cache.mortars
    (; surface_flux_values, surface_flux_values_high_order) = cache.elements
    (; boundary_interpolation) = dg.basis

    limiting_factor .= zeros(eltype(limiting_factor))

    (; positivity_correction_factor) = dg.volume_integral.limiter

    index_rho = 1 # TODO

    for mortar in eachmortar(dg, cache)
        large_element = cache.mortars.neighbor_ids[3, mortar]
        upper_element = cache.mortars.neighbor_ids[2, mortar]
        lower_element = cache.mortars.neighbor_ids[1, mortar]

        # Calc minimal low-order solution
        var_min_upper = typemax(eltype(surface_flux_values))
        var_min_lower = typemax(eltype(surface_flux_values))
        var_min_large = typemax(eltype(surface_flux_values))
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
            var_upper = u[index_rho, indices_small..., upper_element]
            var_lower = u[index_rho, indices_small..., lower_element]
            var_large = u[index_rho, indices_large..., large_element]
            var_min_upper = min(var_min_upper, var_upper)
            var_min_lower = min(var_min_lower, var_lower)
            var_min_large = min(var_min_large, var_large)
        end
        var_min_upper = positivity_correction_factor * var_min_upper
        var_min_lower = positivity_correction_factor * var_min_lower
        var_min_large = positivity_correction_factor * var_min_large

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

            var_upper = u[index_rho, indices_small..., upper_element]
            var_lower = u[index_rho, indices_small..., lower_element]
            var_large = u[index_rho, indices_large..., large_element]

            if min(var_upper, var_lower, var_large) < 0
                error("Safe low-order method produces negative value for conservative variable rho. Try a smaller time step.")
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

            # Calculate Pm
            flux_lower_high_order = surface_flux_values_high_order[index_rho, i,
                                                                   direction_small,
                                                                   lower_element]
            flux_lower_low_order = surface_flux_values[index_rho, i, direction_small,
                                                       lower_element]
            flux_difference_lower = factor_small *
                                    (flux_lower_high_order - flux_lower_low_order)

            flux_upper_high_order = surface_flux_values_high_order[index_rho, i,
                                                                   direction_small,
                                                                   upper_element]
            flux_upper_low_order = surface_flux_values[index_rho, i, direction_small,
                                                       upper_element]
            flux_difference_upper = factor_small *
                                    (flux_upper_high_order - flux_upper_low_order)

            flux_large_high_order = surface_flux_values_high_order[index_rho, i,
                                                                   direction_large,
                                                                   large_element]
            flux_large_low_order = surface_flux_values_high_order[index_rho, i,
                                                                  direction_large,
                                                                  large_element]
            flux_difference_large = factor_large *
                                    (flux_large_high_order - flux_large_low_order)

            Qm_upper = min(0, var_min_upper - var_upper)
            Qm_lower = min(0, var_min_lower - var_lower)
            Qm_large = min(0, var_min_large - var_large)

            Pm_upper = min(0, flux_difference_upper)
            Pm_lower = min(0, flux_difference_lower)
            Pm_large = min(0, flux_difference_large)

            Pm_upper = dt * inverse_jacobian_upper * Pm_upper
            Pm_lower = dt * inverse_jacobian_lower * Pm_lower
            Pm_large = dt * inverse_jacobian_large * Pm_large

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm_upper = abs(Qm_upper) / (abs(Pm_upper) + eps(typeof(Qm_upper)) * 100)
            Qm_lower = abs(Qm_lower) / (abs(Pm_lower) + eps(typeof(Qm_lower)) * 100)
            Qm_large = abs(Qm_large) / (abs(Pm_large) + eps(typeof(Qm_large)) * 100)

            limiting_factor[mortar] = max(limiting_factor[mortar], 1 - Qm_upper,
                                          1 - Qm_lower, 1 - Qm_large)
        end
    end

    return nothing
end

@inline function blend_mortar_flux!(u, semi, equations, dg, t, dt)
    (; mesh, cache) = semi
    (; orientations, limiting_factor) = cache.mortars

    (; surface_flux_values, surface_flux_values_high_order) = cache.elements
    (; boundary_interpolation) = dg.basis

    if semi.solver.mortar.pure_low_order
        limiting_factor .= one(eltype(limiting_factor))
        ############################
        # limiting_factor = 1 => full FV
        # limiting_factor = 0 => full DG
        #######################
    end

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
