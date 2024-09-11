# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function prepare_limiter!(limiter::EntropyBoundedLimiter, mesh::AbstractMesh{1}, equations, dg, cache, u)
    @threaded for element in eachelement(dg, cache)
        s_min = typemax(eltype(u))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            s_min = min(s_min, entropy_thermodynamic(u_node, equations))
        end
        limiter.min_entropy_exp[element] = exp(s_min)
    end

    return nothing
end

function limiter_entropy_bounded!(u, density_threshold, min_entropy_exp,
                                  mesh::AbstractMesh{1}, equations, dg::DGSEM, cache)

    # Call Zhang-Shu limiter to enforce positivity of density
    # TODO: Maybe pressure as well?
    limiter_zhang_shu!(u, density_threshold, Trixi.density,
                       mesh, equations, dg, cache)

    @unpack weights = dg.basis

    @threaded for element in eachelement(dg, cache)
        # determine minimum value for entropy difference
        tau_min = typemax(eltype(u))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            tau = entropy_difference(pressure(u_node, equations), min_entropy_exp[element], density(u_node, equations), equations.gamma)
            tau_min = min(tau_min, tau)
        end
        tau_min = min(0, tau_min)

        # detect if limiting is necessary
        tau_min != 0 || continue

        # compute mean value
        u_mean = zero(get_node_vars(u, equations, dg, 1, element))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            u_mean += u_node * weights[i]
        end
        # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
        u_mean = u_mean / 2^ndims(mesh)

        entropy_difference_mean = entropy_difference(pressure(u_mean, equations), min_entropy_exp[element], density(u_mean, equations), equations.gamma)
        @assert entropy_difference_mean > 0 "Entropy difference mean is not positive!"
        @assert Trixi.density(u_mean, equations) > 0 "Density mean is not positive!"
        
        epsilon = tau_min / (tau_min - entropy_difference_mean)

        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            set_node_vars!(u, (1 - epsilon) * u_node + epsilon * u_mean,
                          equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
