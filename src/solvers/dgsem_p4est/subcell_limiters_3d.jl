# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

###############################################################################
# IDP Limiting
###############################################################################

###############################################################################
# Global positivity limiting of conservative variables

@inline function idp_positivity_conservative!(alpha, limiter,
                                              u::AbstractArray{<:Real, 5}, dt, semi,
                                              elements, variable)
    mesh, _, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis # Plays role of DG subcell sizes
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in elements
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)
            var = u[variable, i, j, k, element]
            if var < 0
                error("Safe low-order method produces negative value for conservative variable $variable. Try a smaller time step.")
            end

            # Compute bound
            var_min[i, j, k, element] = positivity_correction_factor * var

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each interface, not each node
            Qm = min(0, (var_min[i, j, k, element] - var) / dt)

            # Calculate Pm
            # Note: Boundaries of antidiffusive_flux1/2/3 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1_R[variable, i, j, k, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1_L[variable, i + 1, j, k, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2_R[variable, i, j, k, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2_L[variable, i, j + 1, k, element]
            val_flux3_local = inverse_weights[k] *
                              antidiffusive_flux3_R[variable, i, j, k, element]
            val_flux3_local_jp1 = -inverse_weights[k] *
                                  antidiffusive_flux3_L[variable, i, j, k + 1, element]

            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1) +
                 min(0, val_flux3_local) + min(0, val_flux3_local_jp1)
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

            # Calculate alpha
            alpha[i, j, k, element] = max(alpha[i, j, k, element], 1 - Qm)
        end
    end

    return nothing
end

###############################################################################
# Global positivity limiting of nonlinear variables

@inline function idp_positivity_nonlinear!(alpha, limiter,
                                           u::AbstractArray{<:Real, 5},
                                           dt, semi, elements, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol(string(variable), "_min")]

    @threaded for element in elements
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            inverse_jacobian = get_inverse_jacobian(cache.elements.inverse_jacobian,
                                                    mesh, i, j, k, element)

            # Compute bound
            u_local = get_node_vars(u, equations, dg, i, j, k, element)
            var = variable(u_local, equations)
            if var < 0
                error("Safe low-order method produces negative value for variable $variable. Try a smaller time step.")
            end
            var_min[i, j, k, element] = positivity_correction_factor * var

            # Perform Newton's bisection method to find new alpha
            newton_loops_alpha!(alpha, var_min[i, j, k, element],
                                u_local, i, j, k, element,
                                variable, min,
                                initial_check_nonnegative_newton_idp,
                                final_check_nonnegative_newton_idp,
                                inverse_jacobian, dt, equations, dg, cache, limiter)
        end
    end

    return nothing
end

###############################################################################
# Newton-bisection method

@inline function newton_loops_alpha!(alpha, bound, u, i, j, k, element,
                                     variable, min_or_max,
                                     initial_check, final_check,
                                     inverse_jacobian, dt,
                                     equations, dg, cache, limiter)
    (; inverse_weights) = dg.basis # Plays role of inverse DG-subcell sizes
    (; antidiffusive_flux1_L, antidiffusive_flux1_R, antidiffusive_flux2_L, antidiffusive_flux2_R, antidiffusive_flux3_L, antidiffusive_flux3_R) = cache.antidiffusive_fluxes

    (; gamma_constant_newton) = limiter

    indices = (i, j, k, element)

    # negative xi direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive xi direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[i] *
                         get_node_vars(antidiffusive_flux1_L, equations, dg,
                                       i + 1, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # negative eta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive eta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[j] *
                         get_node_vars(antidiffusive_flux2_L, equations, dg,
                                       i, j + 1, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # negative zeta direction
    antidiffusive_flux = gamma_constant_newton * inverse_jacobian *
                         inverse_weights[k] *
                         get_node_vars(antidiffusive_flux3_R, equations, dg,
                                       i, j, k, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    # positive zeta direction
    antidiffusive_flux = -gamma_constant_newton * inverse_jacobian *
                         inverse_weights[k] *
                         get_node_vars(antidiffusive_flux3_L, equations, dg,
                                       i, j, k + 1, element)
    newton_loop!(alpha, bound, u, indices, variable, min_or_max,
                 initial_check, final_check, equations, dt, limiter, antidiffusive_flux)

    return nothing
end
end # @muladd
