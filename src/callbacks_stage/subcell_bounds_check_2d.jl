# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function check_bounds(u, mesh::AbstractMesh{2}, equations, solver, cache,
                              limiter::SubcellLimiterIDP,
                              time, iter, output_directory, save_errors)
    (; positivity) = solver.volume_integral.limiter
    (; variable_bounds) = limiter.cache.subcell_limiter_coefficients
    (; idp_bounds_delta) = limiter.cache

    if save_errors
        open("$output_directory/deviations.txt", "a") do f
            print(f, iter, ", ", time)
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            key = Symbol("$(v)_min")
            deviation_min = zero(eltype(u))
            for element in eachelement(solver, cache), j in eachnode(solver),
                i in eachnode(solver)

                var = u[v, i, j, element]
                deviation_min = max(deviation_min,
                                    variable_bounds[key][i, j, element] - var)
            end
            idp_bounds_delta[key] = max(idp_bounds_delta[key], deviation_min)
            if save_errors
                deviation_min_ = deviation_min
                open("$output_directory/deviations.txt", "a") do f
                    print(f, ", ", deviation_min_)
                end
            end
        end
    end
    if save_errors
        open("$output_directory/deviations.txt", "a") do f
            println(f)
        end
    end

    return nothing
end
end # @muladd
