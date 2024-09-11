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
    #limiter_zhang_shu!(u, 5.0e-6, density, mesh, equations, dg, cache)
    @unpack weights = dg.basis
    
    @threaded for element in eachelement(dg, cache)
        s_min_exp = min_entropy_exp[element]
        # determine minimum value for entropy difference
        tau_min = zero(eltype(u))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            tau = entropy_difference(pressure(u_node, equations), s_min_exp, density(u_node, equations), equations.gamma)
            tau_min = min(tau_min, tau)
        end

        # detect if limiting is necessary. Avoid divison by ("near") zero
        tau_min < -1e-13 || continue

        # compute mean value
        u_mean = zero(get_node_vars(u, equations, dg, 1, element))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            u_mean += u_node * weights[i]
        end
        # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
        u_mean = u_mean / 2^ndims(mesh)

        rho_mean = density(u_mean, equations)
        entropy_difference_mean = entropy_difference(pressure(u_mean, equations), s_min_exp, rho_mean, equations.gamma)

        epsilon = tau_min / (tau_min - entropy_difference_mean)

        #@assert entropy_difference_mean >= 0 "Entropy difference of mean value is negative!"
        #@assert rho_mean > 0 "Density mean is not positive!"
        
        #@assert epsilon >= 0 && epsilon <= 1 "Epsilon is not in [0,1]!"
        # In the derivation of the limiter it is assumed that 
        # entropy_difference_mean >= 0 which would imply epsilon <= 1 (maximum limiting).
        # This might however not always be the case in a simulation, thus 
        # we clip epsilon at 1.
        if epsilon > 1
            epsilon = 1
        #=
        elseif epsilon < 0
            epsilon = 0
        =#
        end

        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            set_node_vars!(u, (1 - epsilon) * u_node + epsilon * u_mean,
                          equations, dg, i, element)
        end
    end
    
    return nothing
end
end # @muladd
