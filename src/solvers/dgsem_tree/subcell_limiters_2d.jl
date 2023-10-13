# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# this method is used when the limiter is constructed as for shock-capturing volume integrals
function create_cache(limiter::Type{SubcellLimiterIDP}, equations::AbstractEquations{2},
                      basis::LobattoLegendreBasis, bound_keys)
    subcell_limiter_coefficients = Trixi.ContainerSubcellLimiterIDP2D{real(basis)
                                                                      }(0,
                                                                        nnodes(basis),
                                                                        bound_keys)

    return (; subcell_limiter_coefficients)
end

function (limiter::SubcellLimiterIDP)(u::AbstractArray{<:Any, 4}, semi, dg::DGSEM, t,
                                      dt;
                                      kwargs...)
    @unpack alpha = limiter.cache.subcell_limiter_coefficients
    alpha .= zero(eltype(alpha))

    if limiter.positivity
        @trixi_timeit timer() "positivity" idp_positivity!(alpha, limiter, u, dt, semi)
    end

    # Calculate alpha1 and alpha2
    @unpack alpha1, alpha2 = limiter.cache.subcell_limiter_coefficients
    @threaded for element in eachelement(dg, semi.cache)
        for j in eachnode(dg), i in 2:nnodes(dg)
            alpha1[i, j, element] = max(alpha[i - 1, j, element], alpha[i, j, element])
        end
        for j in 2:nnodes(dg), i in eachnode(dg)
            alpha2[i, j, element] = max(alpha[i, j - 1, element], alpha[i, j, element])
        end
        alpha1[1, :, element] .= zero(eltype(alpha1))
        alpha1[nnodes(dg) + 1, :, element] .= zero(eltype(alpha1))
        alpha2[:, 1, element] .= zero(eltype(alpha2))
        alpha2[:, nnodes(dg) + 1, element] .= zero(eltype(alpha2))
    end

    return nothing
end

@inline function idp_positivity!(alpha, limiter, u, dt, semi)
    # Conservative variables
    for variable in limiter.positivity_variables_cons
        idp_positivity!(alpha, limiter, u, dt, semi, variable)
    end

    return nothing
end

@inline function idp_positivity!(alpha, limiter, u, dt, semi, variable)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; antidiffusive_flux1, antidiffusive_flux2) = cache.antidiffusive_fluxes
    (; inverse_weights) = dg.basis
    (; positivity_correction_factor) = limiter

    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    var_min = variable_bounds[Symbol("$(variable)_min")]

    @threaded for element in eachelement(dg, semi.cache)
        inverse_jacobian = cache.elements.inverse_jacobian[element]
        for j in eachnode(dg), i in eachnode(dg)
            var = u[variable, i, j, element]
            if var < 0
                error("Safe $variable is not safe. element=$element, node: $i $j, value=$var")
            end

            # Compute bound
            var_min[i, j, element] = positivity_correction_factor * var

            # Real one-sided Zalesak-type limiter
            # * Zalesak (1979). "Fully multidimensional flux-corrected transport algorithms for fluids"
            # * Kuzmin et al. (2010). "Failsafe flux limiting and constrained data projections for equations of gas dynamics"
            # Note: The Zalesak limiter has to be computed, even if the state is valid, because the correction is
            #       for each interface, not each node
            Qm = min(0, (var_min[i, j, element] - var) / dt)

            # Calculate Pm
            # Note: Boundaries of antidiffusive_flux1/2 are constant 0, so they make no difference here.
            val_flux1_local = inverse_weights[i] *
                              antidiffusive_flux1[variable, i, j, element]
            val_flux1_local_ip1 = -inverse_weights[i] *
                                  antidiffusive_flux1[variable, i + 1, j, element]
            val_flux2_local = inverse_weights[j] *
                              antidiffusive_flux2[variable, i, j, element]
            val_flux2_local_jp1 = -inverse_weights[j] *
                                  antidiffusive_flux2[variable, i, j + 1, element]

            Pm = min(0, val_flux1_local) + min(0, val_flux1_local_ip1) +
                 min(0, val_flux2_local) + min(0, val_flux2_local_jp1)
            Pm = inverse_jacobian * Pm

            # Compute blending coefficient avoiding division by zero
            # (as in paper of [Guermond, Nazarov, Popov, Thomas] (4.8))
            Qm = abs(Qm) / (abs(Pm) + eps(typeof(Qm)) * 100)

            # Calculate alpha
            alpha[i, j, element] = max(alpha[i, j, element], 1 - Qm)
        end
    end

    return nothing
end
end # @muladd
