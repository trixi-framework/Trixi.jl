# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    LimitingAnalysisCallback(; output_directory="out", interval=1)

Analyze the subcell blending coefficient of IDP limiting ([`SubcellLimiterIDP`](@ref)) and
monolithic convex limiting (MCL) ([`SubcellLimiterMCL`](@ref)) in the last RK stage of every
`interval` time steps. This contains a volume-weighted average of the node coefficients. For MCL,
the node coefficients are calculated using either the minimum or the mean of the adjacent subcell
interfaces. The results are saved in `alphas.txt` (for IDP limiting), `alpha_min.txt` and
`alphas_mean.txt` (for MCL) in `output_directory`.
"""
struct LimitingAnalysisCallback
    output_directory::String
    interval::Int
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:LimitingAnalysisCallback})
    return nothing
end

function LimitingAnalysisCallback(; output_directory = "out", interval = 1)
    condition = (u, t, integrator) -> interval > 0 &&
        ((integrator.stats.naccept % interval == 0 &&
          !(integrator.stats.naccept == 0 && integrator.iter > 0)) ||
         isfinished(integrator))

    limiting_analysis_callback = LimitingAnalysisCallback(output_directory, interval)

    DiscreteCallback(condition, limiting_analysis_callback,
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t,
                     integrator) where {Condition, Affect! <: LimitingAnalysisCallback}
    if cb.affect!.interval == 0
        return nothing
    end

    initialize!(cb, u_ode, t, integrator, integrator.p.solver.volume_integral)

    return nothing
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t, integrator,
                     volume_integral::AbstractVolumeIntegral) where {Condition,
                                                                     Affect! <:
                                                                     LimitingAnalysisCallback
                                                                     }
    return nothing
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t, integrator,
                     volume_integral::VolumeIntegralSubcellLimiting) where {Condition,
                                                                            Affect! <:
                                                                            LimitingAnalysisCallback
                                                                            }
    initialize!(cb, u_ode, t, integrator, volume_integral.limiter,
                cb.affect!.output_directory)

    return nothing
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t, integrator,
                     limiter::SubcellLimiterIDP,
                     output_directory) where {Condition,
                                              Affect! <: LimitingAnalysisCallback}
    mkpath(output_directory)
    open("$output_directory/alphas.txt", "a") do f
        println(f, "# iter, simu_time, alpha_max, alpha_avg")
    end

    return nothing
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u_ode, t, integrator,
                     limiter::SubcellLimiterMCL,
                     output_directory) where {Condition,
                                              Affect! <: LimitingAnalysisCallback}
    vars = varnames(cons2cons, integrator.p.equations)

    mkpath(output_directory)
    for file in ["alphas_min.txt", "alphas_mean.txt"]
        open("$output_directory/$file", "a") do f
            print(f, "# iter, simu_time",
                  join(", alpha_min_$v, alpha_avg_$v" for v in vars))
            if limiter.positivity_limiter_pressure
                print(f, ", alpha_min_pressure, alpha_avg_pressure")
            end
            if limiter.entropy_limiter_semidiscrete
                print(f, ", alpha_min_entropy, alpha_avg_entropy")
            end
            println(f)
        end
    end

    return nothing
end

@inline function (limiting_analysis_callback::LimitingAnalysisCallback)(integrator)
    mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
    @unpack t = integrator
    iter = integrator.stats.naccept

    limiting_analysis_callback(mesh, equations, solver, cache, solver.volume_integral,
                               t, iter)
end

@inline function (limiting_analysis_callback::LimitingAnalysisCallback)(mesh, equations,
                                                                        solver, cache,
                                                                        volume_integral::AbstractVolumeIntegral,
                                                                        t, iter)
    return nothing
end

@inline function (limiting_analysis_callback::LimitingAnalysisCallback)(mesh, equations,
                                                                        solver, cache,
                                                                        volume_integral::VolumeIntegralSubcellLimiting,
                                                                        t, iter)
    if limiting_analysis_callback.interval == 0 ||
       (iter % limiting_analysis_callback.interval != 0)
        return nothing
    end

    @trixi_timeit timer() "limiting_analysis_callback" limiting_analysis_callback(mesh,
                                                                                  equations,
                                                                                  solver,
                                                                                  cache,
                                                                                  volume_integral.limiter,
                                                                                  t,
                                                                                  iter)
end

@inline function (limiting_analysis_callback::LimitingAnalysisCallback)(mesh, equations,
                                                                        dg, cache,
                                                                        limiter::SubcellLimiterIDP,
                                                                        time, iter)
    @unpack output_directory = limiting_analysis_callback
    @unpack alpha = limiter.cache.subcell_limiter_coefficients

    alpha_avg = analyze_coefficient(mesh, equations, dg, cache, limiter)

    open("$output_directory/alphas.txt", "a") do f
        println(f, iter, ", ", time, ", ", maximum(alpha), ", ", alpha_avg)
    end
end

@inline function (limiting_analysis_callback::LimitingAnalysisCallback)(mesh, equations,
                                                                        dg, cache,
                                                                        limiter::SubcellLimiterMCL,
                                                                        time, iter)
    @assert limiter.Plotting "Parameter `Plotting` needs to be activated for analysis of limiting factor with `LimitingAnalysisCallback`"

    @unpack output_directory = limiting_analysis_callback
    @unpack weights = dg.basis
    @unpack alpha, alpha_pressure, alpha_entropy,
    alpha_mean, alpha_mean_pressure, alpha_mean_entropy = limiter.cache.subcell_limiter_coefficients

    n_vars = nvariables(equations)

    alpha_min_avg, alpha_mean_avg = analyze_coefficient(mesh, equations, dg, cache,
                                                        limiter)

    open("$output_directory/alphas_min.txt", "a") do f
        print(f, iter, ", ", time)
        for v in eachvariable(equations)
            print(f, ", ", minimum(view(alpha, v, ntuple(_ -> :, n_vars - 1)...)))
            print(f, ", ", alpha_min_avg[v])
        end
        if limiter.positivity_limiter_pressure
            print(f, ", ", minimum(alpha_pressure), ", ", alpha_min_avg[n_vars + 1])
        end
        if limiter.entropy_limiter_semidiscrete
            k = n_vars + limiter.positivity_limiter_pressure + 1
            print(f, ", ", minimum(alpha_entropy), ", ", alpha_min_avg[k])
        end
        println(f)
    end
    open("$output_directory/alphas_mean.txt", "a") do f
        print(f, iter, ", ", time)
        for v in eachvariable(equations)
            print(f, ", ", minimum(view(alpha_mean, v, ntuple(_ -> :, n_vars - 1)...)))
            print(f, ", ", alpha_mean_avg[v])
        end
        if limiter.positivity_limiter_pressure
            print(f, ", ", minimum(alpha_mean_pressure), ", ",
                  alpha_mean_avg[n_vars + 1])
        end
        if limiter.entropy_limiter_semidiscrete
            k = n_vars + limiter.positivity_limiter_pressure + 1
            print(f, ", ", minimum(alpha_mean_entropy), ", ", alpha_mean_avg[k])
        end
        println(f)
    end

    return nothing
end
end # @muladd

include("limiting_analysis_2d.jl")
