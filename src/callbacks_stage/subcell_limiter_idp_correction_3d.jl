# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function perform_idp_correction!(u, dt,
                                 mesh::P4estMesh{3},
                                 equations, dg, cache)
    @unpack inverse_weights = dg.basis # Plays role of inverse DG-subcell sizes
    @unpack antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes
    @unpack alpha = dg.volume_integral.limiter.cache.subcell_limiter_coefficients

    # The following code implements the IDP correction in flux-differencing form:
    # u[v, i, j, k, element] += dt * -inverse_jacobian[i, j, k, element] *
    #    (inverse_weights[i] *
    #       ((1 - alpha_1_ip1) * antidiffusive_flux1_ip1[v] - (1 - alpha_1) * antidiffusive_flux1[v]) +
    #     inverse_weights[j] *
    #       ((1 - alpha_2_jp1) * antidiffusive_flux2_jp1[v] - (1 - alpha_2) * antidiffusive_flux2[v]) +
    #     inverse_weights[k] *
    #       ((1 - alpha_3_kp1) * antidiffusive_flux3_kp1[v] - (1 - alpha_3) * antidiffusive_flux3[v]))
    # with
    # alpha_1 = max(alpha[i - 1, j, k, element], alpha[i, j, k, element]),
    # alpha_1_ip1 = max(alpha[i, j, k, element], alpha[i + 1, j, k, element])
    # and equivalently for alpha_2, alpha_2_jp1, alpha_3, alpha_3_kp1.

    # For LGL nodes, the high-order and low-order fluxes at element interfaces are equal
    # and therefore, the antidiffusive fluxes are zero there.
    # To avoid adding zeros and speed up the simulation, we directly loop over the subcell
    # interfaces.

    @threaded for element in eachelement(dg, cache)
        # Perform correction in 1st/x-direction
        for k in eachnode(dg), j in eachnode(dg), i in 2:nnodes(dg)
            # Subcell interface between nodes (i - 1, j, k) and (i, j, k)
            alpha1 = max(alpha[i - 1, j, k, element], alpha[i, j, k, element])

            # Apply to right node (i, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)
            flux1 = get_node_vars(antidiffusive_flux1_R, equations, dg,
                                  i, j, k, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[i] * (1 - alpha1)
            multiply_add_to_node_vars!(u, dg_factor, flux1,
                                       equations, dg, i, j, k, element)

            # Apply to left node (i - 1, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i - 1, j, k, element)
            flux1_ip1 = get_node_vars(antidiffusive_flux1_L, equations, dg,
                                      i, j, k, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[i - 1] * (1 - alpha1)
            multiply_add_to_node_vars!(u, dg_factor, flux1_ip1,
                                       equations, dg, i - 1, j, k, element)
        end

        # Perform correction in 2nd/y-direction
        for k in eachnode(dg), j in 2:nnodes(dg), i in eachnode(dg)
            # Subcell interface between nodes (i, j - 1, k) and (i, j, k)
            alpha2 = max(alpha[i, j - 1, k, element], alpha[i, j, k, element])

            # Apply to right node (i, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)
            flux2 = get_node_vars(antidiffusive_flux2_R, equations, dg,
                                  i, j, k, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[j] * (1 - alpha2)
            multiply_add_to_node_vars!(u, dg_factor, flux2,
                                       equations, dg, i, j, k, element)

            # Apply to left node (i, j - 1, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j - 1, k, element)
            flux2_jp1 = get_node_vars(antidiffusive_flux2_L, equations, dg,
                                      i, j, k, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[j - 1] * (1 - alpha2)
            multiply_add_to_node_vars!(u, dg_factor, flux2_jp1,
                                       equations, dg, i, j - 1, k, element)
        end

        # Perform correction in 3rd/z-direction
        for k in 2:nnodes(dg), j in eachnode(dg), i in eachnode(dg)
            # Subcell interface between nodes (i, j, k - 1) and (i, j, k)
            alpha3 = max(alpha[i, j, k - 1, element], alpha[i, j, k, element])

            # Apply to right node (i, j, k)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)
            flux3 = get_node_vars(antidiffusive_flux3_R, equations, dg,
                                  i, j, k, element)
            dg_factor = -dt * inverse_jacobian * inverse_weights[k] * (1 - alpha3)
            multiply_add_to_node_vars!(u, dg_factor, flux3,
                                       equations, dg, i, j, k, element)

            # Apply to left node (i, j, k - 1)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k - 1, element)
            flux3_kp1 = get_node_vars(antidiffusive_flux3_L, equations, dg,
                                      i, j, k, element)
            dg_factor = dt * inverse_jacobian * inverse_weights[k - 1] * (1 - alpha3)
            multiply_add_to_node_vars!(u, dg_factor, flux3_kp1,
                                       equations, dg, i, j, k - 1, element)
        end
    end

    return nothing
end
end # @muladd
