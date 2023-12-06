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
    (; idp_bounds_delta_local, idp_bounds_delta_global) = limiter.cache

    if local_minmax
        for v in limiter.local_minmax_variables_cons
            v_string = string(v)
            key_min = Symbol(v_string, "_min")
            key_max = Symbol(v_string, "_max")
            deviation_min_threaded = idp_bounds_delta_local[key_min]
            deviation_max_threaded = idp_bounds_delta_local[key_max]
            @threaded for element in eachelement(solver, cache)
                deviation_min = deviation_min_threaded[Threads.threadid()]
                deviation_max = deviation_max_threaded[Threads.threadid()]
                for j in eachnode(solver), i in eachnode(solver)
                    var = u[v, i, j, element]
                    deviation_min = max(deviation_min,
                                        variable_bounds[key_min][i, j, element] - var)
                    deviation_max = max(deviation_max,
                                        var - variable_bounds[key_max][i, j, element])
                end
                deviation_min_threaded[Threads.threadid()] = deviation_min
                deviation_max_threaded[Threads.threadid()] = deviation_max
            end
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_minmax_variables_cons
                continue
            end
            key = Symbol(string(v), "_min")
            deviation_threaded = idp_bounds_delta_local[key]
            @threaded for element in eachelement(solver, cache)
                deviation = deviation_threaded[Threads.threadid()]
                for j in eachnode(solver), i in eachnode(solver)
                    var = u[v, i, j, element]
                    deviation = max(deviation,
                                    variable_bounds[key][i, j, element] - var)
                end
                deviation_threaded[Threads.threadid()] = deviation
            end
        end
    end

    for (key, _) in idp_bounds_delta_local
        # Reduce threaded local maximum deviations. Save in first entry.
        idp_bounds_delta_local[key][1] = reduce(max, idp_bounds_delta_local[key])
        # Update global maximum deviations
        idp_bounds_delta_global[key] = max(idp_bounds_delta_global[key],
                                           idp_bounds_delta_local[key][1])
    end

    if save_errors
        # Print to output file
        open("$output_directory/deviations.txt", "a") do f
            print(f, iter, ", ", time)
            if local_minmax
                for v in limiter.local_minmax_variables_cons
                    v_string = string(v)
                    print(f, ", ",
                          idp_bounds_delta_local[Symbol(v_string, "_min")][1], ", ",
                          idp_bounds_delta_local[Symbol(v_string, "_max")][1])
                end
            end
            if positivity
                for v in limiter.positivity_variables_cons
                    if v in limiter.local_minmax_variables_cons
                        continue
                    end
                    print(f, ", ", idp_bounds_delta_local[Symbol(string(v), "_min")][1])
                end
            end
            println(f)
        end
        # Reset local maximum deviations
        for (key, _) in idp_bounds_delta_local
            idp_bounds_delta_local[key] .= zero(eltype(idp_bounds_delta_local[key][1]))
        end
    end

    return nothing
end
end # @muladd
