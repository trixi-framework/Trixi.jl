# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    LimitingAnalysisCallback(; output_directory="out", interval=1)

Analyze the subcell blending coefficient of IDP limiting ([`SubcellLimiterIDP`](@ref)) in the last
RK stage of every `interval` time steps. This contains a volume-weighted average of the node
coefficients. The results are saved in `alphas.txt` in `output_directory`.

When [`MortarIDP`](@ref) is used, the maximum and the size-weighted average mortar limiting factor
are additionally saved in `mortar_limiting_factor.txt` in `output_directory`.
"""
struct LimitingAnalysisCallback
    output_directory::String
    interval::Int
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:LimitingAnalysisCallback})
    # TODO: Do we want any return value here?
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

    _, _, solver, _ = mesh_equations_solver_cache(integrator.p)
    if solver.mortar isa LobattoLegendreMortarIDP
        open(joinpath(output_directory, "mortar_limiting_factor.txt"), "a") do f
            println(f,
                    "# iter, simu_time, limiting_factor_max, limiting_factor_avg")
        end
    end

    return nothing
end

@inline function (limiting_analysis_callback::LimitingAnalysisCallback)(integrator)
    mesh, equations, solver, cache = mesh_equations_solver_cache(integrator.p)
    @unpack t = integrator
    iter = integrator.stats.naccept

    if (limiting_analysis_callback.interval == 0 ||
        (iter % limiting_analysis_callback.interval != 0)) &&
       !isfinished(integrator)
        return nothing
    end
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

    # Provisional analysis of limiting factor
    if nmortars(cache.mortars) > 0 && dg.mortar isa LobattoLegendreMortarIDP
        (; limiting_factor) = cache.mortars
        limiting_factor_avg = average_mortar_limiting_factor(limiting_factor, mesh,
                                                             dg, cache)

        open(joinpath(output_directory, "mortar_limiting_factor.txt"), "a") do f
            println(f, iter, ", ", time, ", ", maximum(limiting_factor), ", ",
                    limiting_factor_avg)
        end
    end
end
end # @muladd

include("limiting_analysis_2d.jl")
