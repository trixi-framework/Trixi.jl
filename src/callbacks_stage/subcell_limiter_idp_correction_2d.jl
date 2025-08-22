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
end # @muladd
