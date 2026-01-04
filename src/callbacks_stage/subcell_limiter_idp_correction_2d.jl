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
end # @muladd
