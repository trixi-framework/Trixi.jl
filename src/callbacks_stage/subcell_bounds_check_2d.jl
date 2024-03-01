# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              limiter::SubcellLimiterIDP,
                              time, iter, output_directory, save_errors)
    (; local_twosided, positivity, local_onesided) = solver.volume_integral.limiter
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; idp_bounds_delta_local, idp_bounds_delta_global) = limiter.cache

    # Note: Accessing the threaded memory vector `idp_bounds_delta_local` with
    # `deviation = idp_bounds_delta_local[key][Threads.threadid()]` causes critical performance
    # issues due to False Sharing.
    # Initializing a vector with n times the length and using every n-th entry fixes this
    # problem and allows proper scaling:
    # `deviation = idp_bounds_delta_local[key][n * Threads.threadid()]`
    # Since there are no processors with caches over 128B, we use `n = 128B / size(uEltype)`
    stride_size = div(128, sizeof(eltype(u))) # = n

    if local_twosided
        for v in limiter.local_twosided_variables_cons
            v_string = string(v)
            key_min = Symbol(v_string, "_min")
            key_max = Symbol(v_string, "_max")
            deviation_min_threaded = idp_bounds_delta_local[key_min]
            deviation_max_threaded = idp_bounds_delta_local[key_max]
            @threaded for element in eachelement(solver, cache)
                deviation_min = deviation_min_threaded[stride_size * Threads.threadid()]
                deviation_max = deviation_max_threaded[stride_size * Threads.threadid()]
                for j in eachnode(solver), i in eachnode(solver)
                    var = u[v, i, j, element]
                    deviation_min = max(deviation_min,
                                        variable_bounds[key_min][i, j, element] - var)
                    deviation_max = max(deviation_max,
                                        var - variable_bounds[key_max][i, j, element])
                end
                deviation_min_threaded[stride_size * Threads.threadid()] = deviation_min
                deviation_max_threaded[stride_size * Threads.threadid()] = deviation_max
            end
        end
    end
    if local_onesided
        foreach(limiter.local_onesided_variables_nonlinear) do (variable, min_or_max)
            key = Symbol(string(variable), "_", string(min_or_max))
            deviation_threaded = idp_bounds_delta_local[key]
            @threaded for element in eachelement(solver, cache)
                deviation = deviation_threaded[stride_size * Threads.threadid()]
                for j in eachnode(solver), i in eachnode(solver)
                    v = variable(get_node_vars(u, equations, solver, i, j, element),
                                 equations)
                    if min_or_max === max
                        deviation = max(deviation,
                                        v - variable_bounds[key][i, j, element])
                    else # min_or_max === min
                        deviation = max(deviation,
                                        variable_bounds[key][i, j, element] - v)
                    end
                end
                deviation_threaded[stride_size * Threads.threadid()] = deviation
            end
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_twosided_variables_cons
                continue
            end
            key = Symbol(string(v), "_min")
            deviation_threaded = idp_bounds_delta_local[key]
            @threaded for element in eachelement(solver, cache)
                deviation = deviation_threaded[stride_size * Threads.threadid()]
                for j in eachnode(solver), i in eachnode(solver)
                    var = u[v, i, j, element]
                    deviation = max(deviation,
                                    variable_bounds[key][i, j, element] - var)
                end
                deviation_threaded[stride_size * Threads.threadid()] = deviation
            end
        end
        for variable in limiter.positivity_variables_nonlinear
            key = Symbol(string(variable), "_min")
            deviation_threaded = idp_bounds_delta_local[key]
            @threaded for element in eachelement(solver, cache)
                deviation = deviation_threaded[stride_size * Threads.threadid()]
                for j in eachnode(solver), i in eachnode(solver)
                    var = variable(get_node_vars(u, equations, solver, i, j, element),
                                   equations)
                    deviation = max(deviation,
                                    variable_bounds[key][i, j, element] - var)
                end
                deviation_threaded[stride_size * Threads.threadid()] = deviation
            end
        end
    end

    for (key, _) in idp_bounds_delta_local
        # Calculate maximum deviations of all threads
        idp_bounds_delta_local[key][stride_size] = maximum(idp_bounds_delta_local[key][stride_size * i]
                                                           for i in 1:Threads.nthreads())
        # Update global maximum deviations
        idp_bounds_delta_global[key] = max(idp_bounds_delta_global[key],
                                           idp_bounds_delta_local[key][stride_size])
    end

    if save_errors
        # Print to output file
        open("$output_directory/deviations.txt", "a") do f
            print(f, iter, ", ", time)
            if local_twosided
                for v in limiter.local_twosided_variables_cons
                    v_string = string(v)
                    print(f, ", ",
                          idp_bounds_delta_local[Symbol(v_string, "_min")][stride_size],
                          ", ",
                          idp_bounds_delta_local[Symbol(v_string, "_max")][stride_size])
                end
            end
            if local_onesided
                for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
                    print(f, ", ",
                          idp_bounds_delta_local[Symbol(string(variable), "_",
                                                        string(min_or_max))][stride_size])
                end
            end
            if positivity
                for v in limiter.positivity_variables_cons
                    if v in limiter.local_twosided_variables_cons
                        continue
                    end
                    print(f, ", ",
                          idp_bounds_delta_local[Symbol(string(v), "_min")][stride_size])
                end
                for variable in limiter.positivity_variables_nonlinear
                    print(f, ", ",
                          idp_bounds_delta_local[Symbol(string(variable), "_min")][stride_size])
                end
            end
            println(f)
        end
        # Reset local maximum deviations
        for (key, _) in idp_bounds_delta_local
            for i in 1:Threads.nthreads()
                idp_bounds_delta_local[key][stride_size * i] = zero(eltype(idp_bounds_delta_local[key][stride_size]))
            end
        end
    end

    return nothing
end
end # @muladd
