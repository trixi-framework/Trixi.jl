# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              limiter::SubcellLimiterIDP,
                              time, iter, output_directory, save_errors)
    (; local_minmax, positivity) = solver.volume_integral.limiter
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; idp_bounds_delta) = limiter.cache

    if local_minmax
        for v in limiter.local_minmax_variables_cons
            v_string = string(v)
            key_min = Symbol(v_string, "_min")
            key_max = Symbol(v_string, "_max")
            deviation_min = idp_bounds_delta[key_min]
            deviation_max = idp_bounds_delta[key_max]
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = u[v, i, j, element]
                deviation_min[1] = max(deviation_min[1],
                                       variable_bounds[key_min][i, j, element] - var)
                deviation_max[1] = max(deviation_max[1],
                                       var - variable_bounds[key_max][i, j, element])
            end
            deviation_min[2] = max(deviation_min[2], deviation_min[1])
            deviation_max[2] = max(deviation_max[2], deviation_max[1])
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_minmax_variables_cons
                continue
            end
            key = Symbol(string(v), "_min")
            deviation = idp_bounds_delta[key]
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = u[v, i, j, element]
                deviation[1] = max(deviation[1],
                                   variable_bounds[key][i, j, element] - var)
            end
            deviation[2] = max(deviation[2], deviation[1])
        end
    end
    if save_errors
        # Print to output file
        open("$output_directory/deviations.txt", "a") do f
            print(f, iter, ", ", time)
            if local_minmax
                for v in limiter.local_minmax_variables_cons
                    v_string = string(v)
                    print(f, ", ", idp_bounds_delta[Symbol(v_string, "_min")][1], ", ",
                          idp_bounds_delta[Symbol(v_string, "_max")][1])
                end
            end
            if positivity
                for v in limiter.positivity_variables_cons
                    if v in limiter.local_minmax_variables_cons
                        continue
                    end
                    print(f, ", ", idp_bounds_delta[Symbol(string(v), "_min")][1])
                end
            end
            println(f)
        end
        # Reset first entries of idp_bounds_delta
        for (key, _) in idp_bounds_delta
            idp_bounds_delta[key][1] = zero(eltype(idp_bounds_delta[key][1]))
        end
    end

    return nothing
end
end # @muladd
