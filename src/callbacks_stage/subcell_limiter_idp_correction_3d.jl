# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function perform_idp_correction!(u, dt,
                                 mesh::P4estMesh{3},
                                 equations, dg, cache)
    @unpack inverse_weights = dg.basis
    @unpack antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R = cache.antidiffusive_fluxes
    @unpack alpha1, alpha2, alpha3 = dg.volume_integral.limiter.cache.subcell_limiter_coefficients

    @threaded for element in eachelement(dg, cache)
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            # Sign switch as in apply_jacobian!
            inverse_jacobian = -get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                     mesh, i, j, k, element)

            # Note: antidiffusive_flux1[v, i, xi, eta, element] = antidiffusive_flux2[v, xi, i, eta, element] = antidiffusive_flux3[v, xi, eta, i, element] = 0 for all i in 1:nnodes and xi, eta in {1, nnodes+1}
            alpha_flux1 = (1 - alpha1[i, j, k, element]) *
                          get_node_vars(antidiffusive_flux1_R, equations, dg,
                                        i, j, k, element)
            alpha_flux1_ip1 = (1 - alpha1[i + 1, j, k, element]) *
                              get_node_vars(antidiffusive_flux1_L, equations, dg,
                                            i + 1, j, k, element)
            alpha_flux2 = (1 - alpha2[i, j, k, element]) *
                          get_node_vars(antidiffusive_flux2_R, equations, dg,
                                        i, j, k, element)
            alpha_flux2_jp1 = (1 - alpha2[i, j + 1, k, element]) *
                              get_node_vars(antidiffusive_flux2_L, equations, dg,
                                            i, j + 1, k, element)
            alpha_flux3 = (1 - alpha3[i, j, k, element]) *
                          get_node_vars(antidiffusive_flux3_R, equations, dg,
                                        i, j, k, element)
            alpha_flux3_jp1 = (1 - alpha3[i, j, k + 1, element]) *
                              get_node_vars(antidiffusive_flux3_L, equations, dg,
                                            i, j, k + 1, element)

            for v in eachvariable(equations)
                u[v, i, j, k, element] += dt * inverse_jacobian *
                                          (inverse_weights[i] *
                                           (alpha_flux1_ip1[v] - alpha_flux1[v]) +
                                           inverse_weights[j] *
                                           (alpha_flux2_jp1[v] - alpha_flux2[v]) +
                                           inverse_weights[k] *
                                           (alpha_flux3_jp1[v] - alpha_flux3[v]))
            end
        end
    end

    return nothing
end
end # @muladd
