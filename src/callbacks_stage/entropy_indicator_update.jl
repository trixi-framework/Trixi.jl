# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    EntropyIndicatorUpdateStageCB(; indicator::IndicatorEntropyDecay, scaling=1.0)

Stage callback that adjusts the `target_decay` of the given
[`IndicatorEntropyDecay`](@ref) based on the observed entropy production ``\dot{S}`` during
the current Runge-Kutta stage.
In particular, if the global entropy production is positive, i.e., ``\dot{S} > 0``,
the `target_decay` is set to ``-\dot{S} / N_\mathrm{WF} * scaling``, where ``N_\mathrm{WF}`` is the number of cells
that used the weak form volume integral in this stage.
This way, the indicator becomes more restrictive in the next stage and more cells will use the stabilized
volume integral form.
"""
struct EntropyIndicatorUpdateStageCB{Indicator <: IndicatorEntropyDecay, RealT <: Real}
    indicator::Indicator
    scaling::RealT
end

function EntropyIndicatorUpdateStageCB(; indicator::IndicatorEntropyDecay,
                                       scaling = 1.0)
    return EntropyIndicatorUpdateStageCB{typeof(indicator), typeof(scaling)}(indicator,
                                                                             scaling)
end

init_callback(limiter!::EntropyIndicatorUpdateStageCB, semi) = nothing
finalize_callback(limiter!::EntropyIndicatorUpdateStageCB, semi) = nothing

function (limiter!::EntropyIndicatorUpdateStageCB)(u_ode, integrator, stage)
    @assert :du in fieldnames(typeof(integrator)) "EntropyIndicatorUpdateStageCB requires `du` for efficient computation of previous entropy!"
    semi = integrator.p
    t_stage = integrator.t + integrator.dt * integrator.alg.c[stage]
    return limiter!(u_ode, integrator, semi, t_stage)
end

function (limiter!::EntropyIndicatorUpdateStageCB)(u_ode, integrator,
                                                   semi::AbstractSemidiscretization,
                                                   t)
    @trixi_timeit timer() "IndicatorEntropyDecay correction" begin
        @unpack indicator, scaling = limiter!

        u = wrap_array(u_ode, semi)

        # We could follow the approach from `AnalysiCallback` here:
        # 1) Get temporary storage for the time derivative
        ##du_ode = first(get_tmp_cache(integrator))
        # 2) Re-compute update
        ##@notimeit timer() integrator.f(du_ode, u_ode, semi, t)
        # For a stage callback, however, this is not acceptable - 
        # this doubles the computational cost, however!
        # Thus, we rely on time integrators which provide `du` directly.
        du_ode = integrator.du
        du = wrap_array(du_ode, semi)

        update_target_decay!(indicator, scaling, du, u, t, semi)
    end

    return nothing
end
end # @muladd
