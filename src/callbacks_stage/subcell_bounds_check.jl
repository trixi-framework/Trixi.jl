# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    BoundsCheckCallback(; output_directory="out", save_errors=false, interval=1)

Subcell limiting techniques with [`SubcellLimiterIDP`](@ref) are constructed to adhere certain
local or global bounds. To make sure that these bounds are actually met, this callback calculates
the maximum deviation from the bounds. The maximum deviation per applied bound is printed to
the screen at the end of the simulation.
For more insights, when setting `save_errors=true` the occurring errors are exported every
`interval` time steps during the simulation. Then, the maximum deviations since the last
export are saved in "`output_directory`/deviations.txt".
The `BoundsCheckCallback` has to be applied as a stage callback for the SSPRK time integration scheme.

!!! note
    For `SubcellLimiterIDP`, the solution is corrected in the a posteriori correction stage
    [`SubcellLimiterIDPCorrection`](@ref). So, to check the final solution, this bounds check
    callback must be called after the correction stage.
"""
struct BoundsCheckCallback
    output_directory::String
    save_errors::Bool
    interval::Int
end

function BoundsCheckCallback(; output_directory = "out", save_errors = false,
                             interval = 1)
    BoundsCheckCallback(output_directory, save_errors, interval)
end

function (callback::BoundsCheckCallback)(u_ode, integrator, stage)
    mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
    (; t, iter, alg) = integrator
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    @trixi_timeit timer() "check_bounds" check_bounds(u, equations, solver, cache,
                                                      solver.volume_integral)

    save_errors = callback.save_errors && (callback.interval > 0) &&
                  (stage == length(alg.c)) &&
                  ((iter + 1) % callback.interval == 0 ||   # Every `interval` time steps
                   integrator.finalstep ||                  # Planned last time step
                   (iter + 1) >= integrator.opts.maxiters)  # Maximum iterations reached
    if save_errors
        @trixi_timeit timer() "save_errors" save_bounds_check_errors(callback.output_directory,
                                                                     t, iter + 1,
                                                                     equations,
                                                                     solver.volume_integral)
    end
end

@inline function check_bounds(u, equations, solver, cache,
                              volume_integral::VolumeIntegralSubcellLimiting)
    check_bounds(u, equations, solver, cache, volume_integral.limiter)
end

@inline function save_bounds_check_errors(output_directory, t, iter, equations,
                                          volume_integral::VolumeIntegralSubcellLimiting)
    save_bounds_check_errors(output_directory, t, iter, equations,
                             volume_integral.limiter)
end

function init_callback(callback::BoundsCheckCallback, semi)
    init_callback(callback, semi, semi.solver.volume_integral)
end

function init_callback(callback::BoundsCheckCallback, semi,
                       volume_integral::VolumeIntegralSubcellLimiting)
    init_callback(callback, semi, volume_integral.limiter)
end

function init_callback(callback::BoundsCheckCallback, semi, limiter::SubcellLimiterIDP)
    if !callback.save_errors || (callback.interval == 0)
        return nothing
    end

    (; local_twosided, positivity, local_onesided) = limiter
    (; output_directory) = callback
    variables = varnames(cons2cons, semi.equations)

    mkpath(output_directory)
    open("$output_directory/deviations.txt", "a") do f
        print(f, "# iter, simu_time")
        if local_twosided
            for v in limiter.local_twosided_variables_cons
                variable_string = string(variables[v])
                print(f, ", " * variable_string * "_min, " * variable_string * "_max")
            end
        end
        if local_onesided
            for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
                print(f, ", " * string(variable) * "_" * string(min_or_max))
            end
        end
        if positivity
            for v in limiter.positivity_variables_cons
                if v in limiter.local_twosided_variables_cons
                    continue
                end
                print(f, ", " * string(variables[v]) * "_min")
            end
            for variable in limiter.positivity_variables_nonlinear
                print(f, ", " * string(variable) * "_min")
            end
        end
        println(f)
    end

    return nothing
end

function finalize_callback(callback::BoundsCheckCallback, semi)
    finalize_callback(callback, semi, semi.solver.volume_integral)
end

function finalize_callback(callback::BoundsCheckCallback, semi,
                           volume_integral::VolumeIntegralSubcellLimiting)
    finalize_callback(callback, semi, volume_integral.limiter)
end

@inline function finalize_callback(callback::BoundsCheckCallback, semi,
                                   limiter::SubcellLimiterIDP)
    (; local_twosided, positivity, local_onesided) = limiter
    (; idp_bounds_delta_global) = limiter.cache
    variables = varnames(cons2cons, semi.equations)

    println("─"^100)
    println("Maximum deviation from bounds:")
    println("─"^100)
    if local_twosided
        for v in limiter.local_twosided_variables_cons
            v_string = string(v)
            println("$(variables[v]):")
            println("- lower bound: ",
                    idp_bounds_delta_global[Symbol(v_string, "_min")])
            println("- upper bound: ",
                    idp_bounds_delta_global[Symbol(v_string, "_max")])
        end
    end
    if local_onesided
        for (variable, min_or_max) in limiter.local_onesided_variables_nonlinear
            variable_string = string(variable)
            minmax_string = string(min_or_max)
            println("$variable_string:")
            println("- $minmax_string bound: ",
                    idp_bounds_delta_global[Symbol(variable_string, "_",
                                                   minmax_string)])
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_twosided_variables_cons
                continue
            end
            println(string(variables[v]) * ":\n- positivity: ",
                    idp_bounds_delta_global[Symbol(string(v), "_min")])
        end
        for variable in limiter.positivity_variables_nonlinear
            variable_string = string(variable)
            println(variable_string * ":\n- positivity: ",
                    idp_bounds_delta_global[Symbol(variable_string, "_min")])
        end
    end
    println("─"^100 * "\n")

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

include("subcell_bounds_check_2d.jl")
end # @muladd
