# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function save_min_exp_entropy!(limiter::EntropyBoundedLimiter, mesh::AbstractMesh{3},
                               equations, dg, cache, u)
    @threaded for element in eachelement(dg, cache)
        s_min = typemax(eltype(u))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            s_min = min(s_min, entropy_thermodynamic(u_node, equations))
        end
        limiter.min_entropy_exp[element] = exp(s_min)
    end

    return nothing
end

function limiter_entropy_bounded!(u, min_entropy_exp,
                                  mesh::AbstractMesh{3}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        s_min_exp = min_entropy_exp[element]

        # Determine minimum value for entropy difference
        # Can use zero here since tau is defined as min{0, min_x entropy_difference(x)}
        tau_min = zero(eltype(u))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            tau = entropy_difference(pressure(u_node, equations), s_min_exp,
                                     density(u_node, equations), equations.gamma)
            tau_min = min(tau_min, tau)
        end

        # Detect if limiting is necessary. Avoid divison by ("near") zero
        tau_min < -1e-13 || continue

        # Compute mean value
        u_mean = zero(get_node_vars(u, equations, dg, 1, 1, 1, element))
        total_volume = zero(eltype(u))
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian, mesh,
                                                           i, j, k, element)))
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            u_mean += u_node * weights[i] * weights[j] * weights[k] * volume_jacobian
            total_volume += weights[i] * weights[j] * weights[k] * volume_jacobian
        end
        # normalize with the total volume
        u_mean = u_mean / total_volume

        rho_mean = density(u_mean, equations)
        entropy_difference_mean = entropy_difference(pressure(u_mean, equations),
                                                     s_min_exp, rho_mean,
                                                     equations.gamma)

        epsilon = tau_min / (tau_min - entropy_difference_mean)

        # In the derivation of the limiter it is assumed that 
        # entropy_difference_mean >= 0 which would imply epsilon <= 1 (maximum limiting).
        # However, this might not always be the case in a simulation, 
        # thus we clip epsilon at 1.
        if epsilon > 1
            epsilon = 1
        end

        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, k, element)
            set_node_vars!(u, (1 - epsilon) * u_node + epsilon * u_mean,
                           equations, dg, i, j, k, element)
        end
    end

    return nothing
end
end # @muladd
