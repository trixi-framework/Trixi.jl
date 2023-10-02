# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    BoundsCheckCallback(; output_directory="out", save_errors=false, interval=1)

Bounds checking routine for [`SubcellLimiterIDP`](@ref) and [`SubcellLimiterMCL`](@ref). Applied
as a stage callback for SSPRK methods. If `save_errors` is `true`, the resulting deviations are
saved in `output_directory/deviations.txt` for every `interval` time steps.
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
    @unpack t, iter, alg = integrator
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    save_errors_ = callback.save_errors && (callback.interval > 0) &&
                   (stage == length(alg.c))
    @trixi_timeit timer() "check_bounds" check_bounds(u, mesh, equations, solver, cache,
                                                      t, iter + 1,
                                                      callback.output_directory,
                                                      save_errors_, callback.interval)
end

function check_bounds(u, mesh, equations, solver, cache, t, iter, output_directory,
                      save_errors, interval)
    check_bounds(u, mesh, equations, solver, cache, solver.volume_integral, t, iter,
                 output_directory, save_errors, interval)
end

function check_bounds(u, mesh, equations, solver, cache,
                      volume_integral::AbstractVolumeIntegral,
                      t, iter, output_directory, save_errors, interval)
    return nothing
end

function check_bounds(u, mesh, equations, solver, cache,
                      volume_integral::VolumeIntegralSubcellLimiting,
                      t, iter, output_directory, save_errors, interval)
    check_bounds(u, mesh, equations, solver, cache, volume_integral.limiter, t, iter,
                 output_directory, save_errors, interval)
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

    @unpack local_minmax, positivity, spec_entropy, math_entropy = limiter
    @unpack output_directory = callback
    variables = varnames(cons2cons, semi.equations)

    mkpath(output_directory)
    open("$output_directory/deviations.txt", "a") do f
        print(f, "# iter, simu_time")
        if local_minmax
            for index in limiter.local_minmax_variables_cons
                print(f, ", $(variables[index])_min, $(variables[index])_max")
            end
        end
        if spec_entropy
            print(f, ", specEntr_min")
        end
        if math_entropy
            print(f, ", mathEntr_max")
        end
        if positivity
            for index in limiter.positivity_variables_cons
                if index in limiter.local_minmax_variables_cons
                    continue
                end
                print(f, ", $(variables[index])_min")
            end
            for variable in limiter.positivity_variables_nonlinear
                print(f, ", $(variable)_min")
            end
        end
        println(f)
    end

    return nothing
end

function init_callback(callback::BoundsCheckCallback, semi, limiter::SubcellLimiterMCL)
    if !callback.save_errors || (callback.interval == 0)
        return nothing
    end

    @unpack output_directory = callback
    mkpath(output_directory)
    open("$output_directory/deviations.txt", "a") do f
        print(f, "# iter, simu_time",
              join(", $(v)_min, $(v)_max" for v in varnames(cons2cons, semi.equations)))
        if limiter.PressurePositivityLimiterKuzmin
            print(f, ", pressure_min")
        end
        # No check for entropy limiting rn
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
    @unpack local_minmax, positivity, spec_entropy, math_entropy = limiter
    @unpack idp_bounds_delta = limiter.cache
    variables = varnames(cons2cons, semi.equations)

    println("─"^100)
    println("Maximum deviation from bounds:")
    println("─"^100)
    if local_minmax
        for v in limiter.local_minmax_variables_cons
            println("$(variables[v]):")
            println("-lower bound: ", idp_bounds_delta[Symbol("$(v)_min")])
            println("-upper bound: ", idp_bounds_delta[Symbol("$(v)_max")])
        end
    end
    if spec_entropy
        println("spec. entropy:\n- lower bound: ", idp_bounds_delta[:spec_entropy_min])
    end
    if math_entropy
        println("math. entropy:\n- upper bound: ", idp_bounds_delta[:math_entropy_max])
    end
    if positivity
        for v in limiter.positivity_variables_cons
            if v in limiter.local_minmax_variables_cons
                continue
            end
            println("$(variables[v]):\n- positivity: ",
                    idp_bounds_delta[Symbol("$(v)_min")])
        end
        for variable in limiter.positivity_variables_nonlinear
            println("$(variable):\n- positivity: ",
                    idp_bounds_delta[Symbol("$(variable)_min")])
        end
    end
    println("─"^100 * "\n")

    return nothing
end

@inline function finalize_callback(callback::BoundsCheckCallback, semi,
                                   limiter::SubcellLimiterMCL)
    @unpack idp_bounds_delta = limiter.cache

    println("─"^100)
    println("Maximum deviation from bounds:")
    println("─"^100)
    variables = varnames(cons2cons, semi.equations)
    for v in eachvariable(semi.equations)
        println(variables[v], ":\n- lower bound: ", idp_bounds_delta[1, v],
                "\n- upper bound: ", idp_bounds_delta[2, v])
    end
    if limiter.PressurePositivityLimiterKuzmin
        println("pressure:\n- lower bound: ",
                idp_bounds_delta[1, nvariables(semi.equations) + 1])
    end
    println("─"^100 * "\n")

    return nothing
end

include("bounds_check_2d.jl")
end # @muladd
