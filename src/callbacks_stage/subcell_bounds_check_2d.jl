# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u, equations::AbstractEquations{2}, # only works for 2D
                              solver, cache, limiter::SubcellLimiterIDP)
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
end # @muladd
