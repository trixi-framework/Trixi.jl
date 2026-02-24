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
end # @muladd
