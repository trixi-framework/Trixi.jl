# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    BoundsCheckCallback(; output_directory="out", save_errors=false, interval=1)

Bounds checking routine for [`SubcellLimiterIDP`](@ref). Applied as a stage callback for SSPRK
methods. If `save_errors` is `true`, the resulting deviations are saved in
`output_directory/deviations.txt` for every `interval` time steps.
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

    save_errors_ = callback.save_errors && (callback.interval > 0) &&
                   (iter % callback.interval == 0) && (stage == length(alg.c))
    @trixi_timeit timer() "check_bounds" check_bounds(u, mesh, equations, solver, cache,
                                                      t, iter + 1,
                                                      callback.output_directory,
                                                      save_errors_)
end

function check_bounds(u, mesh, equations, solver, cache, t, iter, output_directory,
                      save_errors)
    check_bounds(u, mesh, equations, solver, cache, solver.volume_integral, t, iter,
                 output_directory, save_errors)
end

function check_bounds(u, mesh, equations, solver, cache,
                      volume_integral::AbstractVolumeIntegral, t, iter,
                      output_directory, save_errors)
    return nothing
end

function check_bounds(u, mesh, equations, solver, cache,
                      volume_integral::VolumeIntegralSubcellLimiting, t, iter,
                      output_directory, save_errors)
    check_bounds(u, mesh, equations, solver, cache, volume_integral.limiter, t, iter,
                 output_directory, save_errors)
end

function init_callback(callback, semi)
    init_callback(callback, semi, semi.solver.volume_integral)
end

init_callback(callback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function init_callback(callback, semi, volume_integral::VolumeIntegralSubcellLimiting)
    init_callback(callback, semi, volume_integral.limiter)
end

function init_callback(callback::BoundsCheckCallback, semi, limiter::SubcellLimiterIDP)
    if !callback.save_errors || (callback.interval == 0)
        return nothing
    end

    (; positivity) = limiter
    (; output_directory) = callback
    variables = varnames(cons2cons, semi.equations)

    mkpath(output_directory)
    open("$output_directory/deviations.txt", "a") do f
        print(f, "# iter, simu_time")
        if positivity
            for index in limiter.positivity_variables_cons
                print(f, ", $(variables[index])_min")
            end
        end
        println(f)
    end

    return nothing
end

function finalize_callback(callback, semi)
    finalize_callback(callback, semi, semi.solver.volume_integral)
end

finalize_callback(callback, semi, volume_integral::AbstractVolumeIntegral) = nothing

function finalize_callback(callback, semi,
                           volume_integral::VolumeIntegralSubcellLimiting)
    finalize_callback(callback, semi, volume_integral.limiter)
end

@inline function finalize_callback(callback::BoundsCheckCallback, semi,
                                   limiter::SubcellLimiterIDP)
    (; positivity) = limiter
    (; idp_bounds_delta) = limiter.cache
    variables = varnames(cons2cons, semi.equations)

    println("─"^100)
    println("Maximum deviation from bounds:")
    println("─"^100)
    if positivity
        for v in limiter.positivity_variables_cons
            println("$(variables[v]):\n- positivity: ",
                    idp_bounds_delta[Symbol("$(v)_min")])
        end
    end
    println("─"^100 * "\n")

    return nothing
end

include("subcell_bounds_check_2d.jl")
end # @muladd
