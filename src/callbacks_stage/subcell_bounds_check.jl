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

    save_errors = callback.save_errors && (callback.interval > 0) &&
                  (stage == length(alg.c)) &&
                  (iter % callback.interval == 0 || integrator.finalstep)
    @trixi_timeit timer() "check_bounds" check_bounds(u, mesh, equations, solver, cache,
                                                      solver.volume_integral)

    if save_errors
        @trixi_timeit timer() "save_errors" save_bounds_check_errors(callback.output_directory,
                                                                     u, t, iter + 1,
                                                                     equations,
                                                                     solver.volume_integral)
    end
end

@inline function check_bounds(u, mesh, equations, solver, cache,
                              volume_integral::AbstractVolumeIntegral)
    return nothing
end

@inline function check_bounds(u, mesh, equations, solver, cache,
                              volume_integral::VolumeIntegralSubcellLimiting)
    check_bounds(u, mesh, equations, solver, cache, volume_integral.limiter)
end

@inline function save_bounds_check_errors(output_directory, u, t, iter, equations,
                                          volume_integral::AbstractVolumeIntegral)
    return nothing
end

@inline function save_bounds_check_errors(output_directory, u, t, iter, equations,
                                          volume_integral::VolumeIntegralSubcellLimiting)
    save_bounds_check_errors(output_directory, u, t, iter, equations,
                             volume_integral.limiter)
end

function init_callback(callback::BoundsCheckCallback, semi)
    init_callback(callback, semi, semi.solver.volume_integral)
end

init_callback(callback::BoundsCheckCallback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function init_callback(callback::BoundsCheckCallback, semi,
                       volume_integral::VolumeIntegralSubcellLimiting)
    init_callback(callback, semi, volume_integral.limiter)
end

function init_callback(callback::BoundsCheckCallback, semi, limiter::SubcellLimiterIDP)
    if !callback.save_errors || (callback.interval == 0)
        return nothing
    end

    (; local_minmax, positivity) = limiter
    (; output_directory) = callback
    variables = varnames(cons2cons, semi.equations)

    mkpath(output_directory)
    open("$output_directory/deviations.txt", "a") do f
        print(f, "# iter, simu_time")
        if local_minmax
            for v in limiter.local_minmax_variables_cons
                variable_string = string(variables[v])
                print(f, ", " * variable_string * "_min, " * variable_string * "_max")
            end
        end
        if positivity
            for v in limiter.positivity_variables_cons
                if v in limiter.local_minmax_variables_cons
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

finalize_callback(callback::BoundsCheckCallback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function finalize_callback(callback::BoundsCheckCallback, semi,
                           volume_integral::VolumeIntegralSubcellLimiting)
    finalize_callback(callback, semi, volume_integral.limiter)
end

@inline function finalize_callback(callback::BoundsCheckCallback, semi,
                                   limiter::SubcellLimiterIDP)
    (; local_minmax, positivity) = limiter
    (; idp_bounds_delta_global) = limiter.cache
    variables = varnames(cons2cons, semi.equations)

    println("─"^100)
    println("Maximum deviation from bounds:")
    println("─"^100)
    if local_minmax
        for v in limiter.local_minmax_variables_cons
            v_string = string(v)
            println("$(variables[v]):")
            println("- lower bound: ",
                    idp_bounds_delta_global[Symbol(v_string, "_min")])
            println("- upper bound: ",
                    idp_bounds_delta_global[Symbol(v_string, "_max")])
        end
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_minmax_variables_cons
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

include("subcell_bounds_check_2d.jl")
end # @muladd
