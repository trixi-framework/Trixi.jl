# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    EntropyIndicatorCorrection(indicator::IndicatorEntropyIncrease)

Stage callback that adjusts the `threshold` of the given
[`IndicatorEntropyIncrease`](@ref) based on the observed entropy production ``\dot{S}`` during
the current Runge-Kutta stage.
In particular, if the global entropy production is positive, i.e., ``\dot{S} > 0``,
the `threshold` is set to ``-\dot{S} / N_\mathrm{WF}``, where ``N_\mathrm{WF}`` is the number of cells
that used the weak form volume integral in this stage.
This way, the indicator becomes more restrictive in the next stage and more cells will use the stabilized
volume integral form.
"""
struct EntropyIndicatorCorrection{Indicator <: IndicatorEntropyIncrease}
    indicator::Indicator
end

function EntropyIndicatorCorrection(indicator)
    return EntropyIndicatorCorrection{typeof(indicator)}(indicator)
end

init_callback(limiter!::EntropyIndicatorCorrection, semi) = nothing
finalize_callback(limiter!::EntropyIndicatorCorrection, semi) = nothing

function (limiter!::EntropyIndicatorCorrection)(u_ode, integrator, stage)
    @assert :du in fieldnames(typeof(integrator)) "EntropyIndicatorCorrection requires `du` for efficient computation of previous entropy!"
    semi = integrator.p
    t_stage = integrator.t + integrator.dt * integrator.alg.c[stage]
    return limiter!(u_ode, integrator, semi, t_stage)
end

function (limiter!::EntropyIndicatorCorrection)(u_ode, integrator,
                                                semi::AbstractSemidiscretization,
                                                t)
    @trixi_timeit timer() "IndicatorEntropyIncrease correction" begin
        @unpack indicator = limiter!
        @unpack threshold, n_cells_fluxdiff_threaded = indicator

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

        #println("norm u_ode SC: ", norm(u))
        #println("norm du_ode SC: ", norm(du))

        mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

        dS = analyze(entropy_timederivative, du, u, t,
                     mesh, equations, dg, cache)

        #println("dS: ", dS)

        safety_scaling = 2.0

        if dS > 0
            # Number of cells calculated with flux differencing in this stage
            n_cells_fd = sum(n_cells_fluxdiff_threaded)
            # Number of cells calculated with weak form in this stage
            n_cells_wf = nelements(dg, cache) - n_cells_fd

            if n_cells_wf > 0
                threshold = -dS / n_cells_wf * safety_scaling
            end
        end
    end

    return nothing
end
end # @muladd
