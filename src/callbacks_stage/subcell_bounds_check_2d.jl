# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u::AbstractArray{<:Any, 4},
                              equations, solver, cache,
                              limiter::SubcellLimiterIDP)
    (; local_twosided, positivity, local_onesided) = solver.volume_integral.limiter
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; idp_bounds_delta_local, idp_bounds_delta_global) = limiter.cache

    # Note: In order to get the maximum deviation from the target bounds, this bounds check
    # requires a reduction in every RK stage and for every enabled limiting option. To make
    # this Thread-parallel we are using Polyester.jl's (at least v0.7.10) `@batch reduction`
    # functionality.
    # Although `@threaded` and `@batch` are currently used equivalently in Trixi.jl, we use
    # `@batch` here to allow a possible redefinition of `@threaded` without creating errors here.
    # See also https://github.com/trixi-framework/Trixi.jl/pull/1888#discussion_r1537785293.

    if local_twosided
        for v in limiter.local_twosided_variables_cons
            v_string = string(v)
            key_min = Symbol(v_string, "_min")
            key_max = Symbol(v_string, "_max")
            deviation_min = idp_bounds_delta_local[key_min]
            deviation_max = idp_bounds_delta_local[key_max]
            @batch reduction=((max, deviation_min), (max, deviation_max)) for element in eachelement(solver,
                                                                                                     cache)
                for j in eachnode(solver), i in eachnode(solver)
                    var = u[v, i, j, element]
                    # Note: We always save the absolute deviations >= 0 and therefore use the
                    # `max` operator for the lower and upper bound. The different directions of
                    # upper and lower bound are considered in their calculations with a
                    # different sign.
                    deviation_min = max(deviation_min,
                                        variable_bounds[key_min][i, j, element] - var)
                    deviation_max = max(deviation_max,
                                        var - variable_bounds[key_max][i, j, element])
                end
            end
            idp_bounds_delta_local[key_min] = deviation_min
            idp_bounds_delta_local[key_max] = deviation_max
        end
    end
    if local_onesided
        for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
            key = Symbol(string(variable), "_", string(min_or_max))
            deviation = idp_bounds_delta_local[key]
            sign_ = min_or_max(1.0, -1.0)
            @batch reduction=(max, deviation) for element in eachelement(solver, cache)
                for j in eachnode(solver), i in eachnode(solver)
                    v = variable(get_node_vars(u, equations, solver, i, j, element),
                                 equations)
                    # Note: We always save the absolute deviations >= 0 and therefore use the
                    # `max` operator for lower and upper bounds. The different directions of
                    # upper and lower bounds are considered with `sign_`.
                    deviation = max(deviation,
                                    sign_ * (v - variable_bounds[key][i, j, element]))
                end
            end
            idp_bounds_delta_local[key] = deviation
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_twosided_variables_cons
                continue
            end
            key = Symbol(string(v), "_min")
            deviation = idp_bounds_delta_local[key]
            @batch reduction=(max, deviation) for element in eachelement(solver, cache)
                for j in eachnode(solver), i in eachnode(solver)
                    var = u[v, i, j, element]
                    deviation = max(deviation,
                                    variable_bounds[key][i, j, element] - var)
                end
            end
            idp_bounds_delta_local[key] = deviation
        end
        for variable in limiter.positivity_variables_nonlinear
            key = Symbol(string(variable), "_min")
            deviation = idp_bounds_delta_local[key]
            @batch reduction=(max, deviation) for element in eachelement(solver, cache)
                for j in eachnode(solver), i in eachnode(solver)
                    var = variable(get_node_vars(u, equations, solver, i, j, element),
                                   equations)
                    deviation = max(deviation,
                                    variable_bounds[key][i, j, element] - var)
                end
            end
            idp_bounds_delta_local[key] = deviation
        end
    end

    for (key, _) in idp_bounds_delta_local
        # Update global maximum deviations
        idp_bounds_delta_global[key] = max(idp_bounds_delta_global[key],
                                           idp_bounds_delta_local[key])
    end

    return nothing
end

@inline function save_bounds_check_errors(output_directory, time, iter, equations,
                                          limiter::SubcellLimiterIDP)
    (; local_twosided, positivity, local_onesided) = limiter
    (; idp_bounds_delta_local) = limiter.cache

    # Print to output file
    open(joinpath(output_directory, "deviations.txt"), "a") do f
        print(f, iter, ", ", time)
        if local_twosided
            for v in limiter.local_twosided_variables_cons
                v_string = string(v)
                print(f, ", ", idp_bounds_delta_local[Symbol(v_string, "_min")],
                      ", ", idp_bounds_delta_local[Symbol(v_string, "_max")])
            end
        end
        if local_onesided
            for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
                key = Symbol(string(variable), "_", string(min_or_max))
                print(f, ", ", idp_bounds_delta_local[key])
            end
        end
        if positivity
            for v in limiter.positivity_variables_cons
                if v in limiter.local_twosided_variables_cons
                    continue
                end
                print(f, ", ", idp_bounds_delta_local[Symbol(string(v), "_min")])
            end
            for variable in limiter.positivity_variables_nonlinear
                print(f, ", ", idp_bounds_delta_local[Symbol(string(variable), "_min")])
            end
        end
        println(f)
    end
    # Reset local maximum deviations
    for (key, _) in idp_bounds_delta_local
        idp_bounds_delta_local[key] = zero(eltype(idp_bounds_delta_local[key]))
    end

    return nothing
end

@inline function check_bounds(u::AbstractArray{<:Any, 4},
                              equations, solver, cache,
                              limiter::SubcellLimiterMCL)
    (; var_min, var_max) = limiter.cache.subcell_limiter_coefficients
    (; bar_states1, bar_states2, lambda1, lambda2) = limiter.cache.container_bar_states
    (; mcl_bounds_delta_local, mcl_bounds_delta_global) = limiter.cache
    (; antidiffusive_flux1_L, antidiffusive_flux2_L) = cache.antidiffusive_fluxes

    n_vars = nvariables(equations)

    if limiter.density_limiter
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # New solution u^{n+1}
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   var_min[1, i, j, element] -
                                                   u[1, i, j, element])
                mcl_bounds_delta_local[2, 1] = max(mcl_bounds_delta_local[2, 1],
                                                   u[1, i, j, element] -
                                                   var_max[1, i, j, element])

                # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
                # Checking the bounds for...
                # - density (rho):
                #   \bar{rho}^{min} <= \bar{rho}^{Lim} <= \bar{rho}^{max}

                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1_L[1, i, j, element] /
                              lambda1[i, j, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   var_min[1, i, j, element] -
                                                   rho_limited)
                mcl_bounds_delta_local[2, 1] = max(mcl_bounds_delta_local[2, 1],
                                                   rho_limited -
                                                   var_max[1, i, j, element])
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1_L[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   var_min[1, i, j, element] -
                                                   rho_limited)
                mcl_bounds_delta_local[2, 1] = max(mcl_bounds_delta_local[2, 1],
                                                   rho_limited -
                                                   var_max[1, i, j, element])
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2_L[1, i, j, element] /
                              lambda2[i, j, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   var_min[1, i, j, element] -
                                                   rho_limited)
                mcl_bounds_delta_local[2, 1] = max(mcl_bounds_delta_local[2, 1],
                                                   rho_limited -
                                                   var_max[1, i, j, element])
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2_L[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   var_min[1, i, j, element] -
                                                   rho_limited)
                mcl_bounds_delta_local[2, 1] = max(mcl_bounds_delta_local[2, 1],
                                                   rho_limited -
                                                   var_max[1, i, j, element])
            end
        end
    end # limiter.density_limiter

    if limiter.sequential_limiter
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # New solution u^{n+1}
                for v in 2:n_vars
                    var_limited = u[v, i, j, element] / u[1, i, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited -
                                                       var_max[v, i, j, element])
                end
                if limiter.positivity_limiter_pressure
                    error_pressure = 0.5 *
                                     (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                     u[1, i, j, element] * u[4, i, j, element]
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                end

                # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
                # Checking the bounds for...
                # - velocities and total energy (phi):
                #   \bar{phi}^{min} <= \bar{phi}^{Lim} / \bar{rho}^{Lim} <= \bar{phi}^{max}
                # - pressure (p):
                #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
                var_limited = zero(eltype(mcl_bounds_delta_local))
                error_pressure = zero(eltype(mcl_bounds_delta_local))
                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1_L[1, i, j, element] /
                              lambda1[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i, j, element] -
                                  antidiffusive_flux1_L[v, i, j, element] /
                                  lambda1[i, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited / rho_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited / rho_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1_L[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                for v in 2:n_vars
                    var_limited = bar_states1[v, i + 1, j, element] +
                                  antidiffusive_flux1_L[v, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited / rho_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited / rho_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2_L[1, i, j, element] /
                              lambda2[i, j, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j, element] -
                                  antidiffusive_flux2_L[v, i, j, element] /
                                  lambda2[i, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited / rho_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited / rho_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2_L[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j + 1, element] +
                                  antidiffusive_flux2_L[v, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited / rho_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited / rho_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
            end
        end
    elseif limiter.conservative_limiter
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # New solution u^{n+1}
                for v in 2:n_vars
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       u[v, i, j, element])
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       u[v, i, j, element] -
                                                       var_max[v, i, j, element])
                end
                if limiter.positivity_limiter_pressure
                    error_pressure = 0.5 *
                                     (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                     u[1, i, j, element] * u[4, i, j, element]
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                end

                # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
                # Checking the bounds for...
                # - conservative variables (noted as rho*phi):
                #   \bar{rho*phi}^{min} <= \bar{rho*phi}^{Lim} <= \bar{rho*phi}^{max}
                # - pressure (p):
                #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2
                var_limited = zero(eltype(mcl_bounds_delta_local))
                error_pressure = zero(eltype(mcl_bounds_delta_local))
                # -x
                for v in 2:n_vars
                    var_limited = bar_states1[v, i, j, element] -
                                  antidiffusive_flux1_L[v, i, j, element] /
                                  lambda1[i, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    rho_limited = bar_states1[1, i, j, element] -
                                  antidiffusive_flux1_L[1, i, j, element] /
                                  lambda1[i, j, element]
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
                # +x
                for v in 2:n_vars
                    var_limited = bar_states1[v, i + 1, j, element] +
                                  antidiffusive_flux1_L[v, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    rho_limited = bar_states1[1, i + 1, j, element] +
                                  antidiffusive_flux1_L[1, i + 1, j, element] /
                                  lambda1[i + 1, j, element]
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
                # -y
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j, element] -
                                  antidiffusive_flux2_L[v, i, j, element] /
                                  lambda2[i, j, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    rho_limited = bar_states2[1, i, j, element] -
                                  antidiffusive_flux2_L[1, i, j, element] /
                                  lambda2[i, j, element]
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
                # +y
                for v in 2:n_vars
                    var_limited = bar_states2[v, i, j + 1, element] +
                                  antidiffusive_flux2_L[v, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    mcl_bounds_delta_local[1, v] = max(mcl_bounds_delta_local[1, v],
                                                       var_min[v, i, j, element] -
                                                       var_limited)
                    mcl_bounds_delta_local[2, v] = max(mcl_bounds_delta_local[2, v],
                                                       var_limited -
                                                       var_max[v, i, j, element])
                    if limiter.positivity_limiter_pressure && (v == 2 || v == 3)
                        error_pressure += 0.5 * var_limited^2
                    end
                end
                if limiter.positivity_limiter_pressure
                    rho_limited = bar_states2[1, i, j + 1, element] +
                                  antidiffusive_flux2_L[1, i, j + 1, element] /
                                  lambda2[i, j + 1, element]
                    error_pressure -= var_limited * rho_limited
                    mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                       n_vars + 1],
                                                                error_pressure)
                    error_pressure = zero(eltype(mcl_bounds_delta_local))
                end
            end
        end
    elseif limiter.positivity_limiter_pressure
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # New solution u^{n+1}
                error_pressure = 0.5 * (u[2, i, j, element]^2 + u[3, i, j, element]^2) -
                                 u[1, i, j, element] * u[4, i, j, element]
                mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                   n_vars + 1],
                                                            error_pressure)

                # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
                # Checking the bounds for...
                # - pressure (p):
                #   \bar{rho}^{Lim} \bar{rho * E}^{Lim} >= |\bar{rho * v}^{Lim}|^2 / 2

                # -x
                rho_limited = bar_states1[1, i, j, element] -
                              antidiffusive_flux1_L[1, i, j, element] /
                              lambda1[i, j, element]
                error_pressure = 0.5 *
                                 (bar_states1[2, i, j, element] -
                                  antidiffusive_flux1_L[2, i, j, element] /
                                  lambda1[i, j, element])^2 +
                                 0.5 *
                                 (bar_states1[3, i, j, element] -
                                  antidiffusive_flux1_L[3, i, j, element] /
                                  lambda1[i, j, element])^2 -
                                 (bar_states1[4, i, j, element] -
                                  antidiffusive_flux1_L[4, i, j, element] /
                                  lambda1[i, j, element]) * rho_limited
                mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                   n_vars + 1],
                                                            error_pressure)
                # +x
                rho_limited = bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1_L[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                error_pressure = 0.5 *
                                 (bar_states1[2, i + 1, j, element] +
                                  antidiffusive_flux1_L[2, i + 1, j, element] /
                                  lambda1[i + 1, j, element])^2 +
                                 0.5 *
                                 (bar_states1[3, i + 1, j, element] +
                                  antidiffusive_flux1_L[3, i + 1, j, element] /
                                  lambda1[i + 1, j, element])^2 -
                                 (bar_states1[4, i + 1, j, element] +
                                  antidiffusive_flux1_L[4, i + 1, j, element] /
                                  lambda1[i + 1, j, element]) * rho_limited
                mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                   n_vars + 1],
                                                            error_pressure)
                # -y
                rho_limited = bar_states2[1, i, j, element] -
                              antidiffusive_flux2_L[1, i, j, element] /
                              lambda2[i, j, element]
                error_pressure = 0.5 *
                                 (bar_states2[2, i, j, element] -
                                  antidiffusive_flux2_L[2, i, j, element] /
                                  lambda2[i, j, element])^2 +
                                 0.5 *
                                 (bar_states2[3, i, j, element] -
                                  antidiffusive_flux2_L[3, i, j, element] /
                                  lambda2[i, j, element])^2 -
                                 (bar_states2[4, i, j, element] -
                                  antidiffusive_flux2_L[4, i, j, element] /
                                  lambda2[i, j, element]) * rho_limited
                mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                   n_vars + 1],
                                                            error_pressure)
                # +y
                rho_limited = bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2_L[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                error_pressure = 0.5 *
                                 (bar_states2[2, i, j + 1, element] +
                                  antidiffusive_flux2_L[2, i, j + 1, element] /
                                  lambda2[i, j + 1, element])^2 +
                                 0.5 *
                                 (bar_states2[3, i, j + 1, element] +
                                  antidiffusive_flux2_L[3, i, j + 1, element] /
                                  lambda2[i, j + 1, element])^2 -
                                 (bar_states2[4, i, j + 1, element] +
                                  antidiffusive_flux2_L[4, i, j + 1, element] /
                                  lambda2[i, j + 1, element]) * rho_limited
                mcl_bounds_delta_local[1, n_vars + 1] = max(mcl_bounds_delta_local[1,
                                                                                   n_vars + 1],
                                                            error_pressure)
            end
        end
    end # limiter.positivity_limiter_pressure

    if limiter.positivity_limiter_density
        beta = limiter.positivity_limiter_correction_factor
        for element in eachelement(solver, cache)
            for j in eachnode(solver), i in eachnode(solver)
                # New solution u^{n+1}
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   -u[1, i, j, element])

                # Limited bar states \bar{u}^{Lim} = \bar{u} + Δf^{Lim} / λ
                # Checking the bounds for...
                # - density (rho):
                #   beta * \bar{rho} <= \bar{rho}^{Lim}

                # -x
                rho_limited = (1 - beta) * bar_states1[1, i, j, element] -
                              antidiffusive_flux1_L[1, i, j, element] /
                              lambda1[i, j, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   -rho_limited)
                # +x
                rho_limited = (1 - beta) * bar_states1[1, i + 1, j, element] +
                              antidiffusive_flux1_L[1, i + 1, j, element] /
                              lambda1[i + 1, j, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   -rho_limited)
                # -y
                rho_limited = (1 - beta) * bar_states2[1, i, j, element] -
                              antidiffusive_flux2_L[1, i, j, element] /
                              lambda2[i, j, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   -rho_limited)
                # +y
                rho_limited = (1 - beta) * bar_states2[1, i, j + 1, element] +
                              antidiffusive_flux2_L[1, i, j + 1, element] /
                              lambda2[i, j + 1, element]
                mcl_bounds_delta_local[1, 1] = max(mcl_bounds_delta_local[1, 1],
                                                   -rho_limited)
            end
        end
    end # limiter.positivity_limiter_density

    for v in eachvariable(equations)
        mcl_bounds_delta_global[1, v] = max(mcl_bounds_delta_global[1, v],
                                            mcl_bounds_delta_local[1, v])
        mcl_bounds_delta_global[2, v] = max(mcl_bounds_delta_global[2, v],
                                            mcl_bounds_delta_local[2, v])
    end
    if limiter.positivity_limiter_pressure
        mcl_bounds_delta_global[1, n_vars + 1] = max(mcl_bounds_delta_global[1,
                                                                             n_vars + 1],
                                                     mcl_bounds_delta_local[1,
                                                                            n_vars + 1])
    end

    return nothing
end

@inline function save_bounds_check_errors(output_directory, time, iter, equations,
                                          limiter::SubcellLimiterMCL)
    (; mcl_bounds_delta_local) = limiter.cache

    n_vars = nvariables(equations)

    # Print errors to output file
    open(joinpath(output_directory, "deviations.txt"), "a") do f
        print(f, iter, ", ", time)
        for v in eachvariable(equations)
            print(f, ", ", mcl_bounds_delta_local[1, v], ", ",
                  mcl_bounds_delta_local[2, v])
        end
        if limiter.positivity_limiter_pressure
            print(f, ", ", mcl_bounds_delta_local[1, n_vars + 1])
        end
        println(f)
    end

    # Reset mcl_bounds_delta_local
    for v in eachvariable(equations)
        mcl_bounds_delta_local[1, v] = zero(eltype(mcl_bounds_delta_local))
        mcl_bounds_delta_local[2, v] = zero(eltype(mcl_bounds_delta_local))
    end
    if limiter.positivity_limiter_pressure
        mcl_bounds_delta_local[1, n_vars + 1] = zero(eltype(mcl_bounds_delta_local))
    end

    return nothing
end
end # @muladd
