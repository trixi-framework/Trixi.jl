# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    EntropyIndicatorUpdateStepCB(; indicator::IndicatorEntropyDecay, scaling=1.0)

Step callback that adjusts the `target_decay` of the given
[`IndicatorEntropyDecay`](@ref) based on the observed entropy production ``\dot{S}`` during
the current Runge-Kutta step.
In particular, if the global entropy production is positive, i.e., ``\dot{S} > 0``,
the `target_decay` is set to ``-\dot{S} / N_\mathrm{WF} * scaling``, where ``N_\mathrm{WF}`` is the number of cells
that used the weak form volume integral in this Runge-Kutta step.
This way, the indicator becomes more restrictive in the next step and more cells will use the stabilized
volume integral form.
"""
mutable struct EntropyIndicatorUpdateStepCB{Indicator <: IndicatorEntropyDecay,
                                            RealT <: Real}
    indicator::Indicator
    scaling::RealT
end

function EntropyIndicatorUpdateStepCB(; indicator::IndicatorEntropyDecay, scaling = 1.0)
    entropy_ind_up_step_cb = EntropyIndicatorUpdateStepCB(indicator, scaling)

    return DiscreteCallback(entropy_ind_up_step_cb, entropy_ind_up_step_cb, # the first one is the condition, the second the affect!
                            save_positions = (false, false),
                            initialize = initialize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:EntropyIndicatorUpdateStepCB})
    @nospecialize cb # reduce precompilation time
    entropy_ind_up_step_cb = cb.affect!
    @unpack scaling = entropy_ind_up_step_cb

    print(io, "EntropyIndicatorUpdateStepCB(scaling=", scaling, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:EntropyIndicatorUpdateStepCB})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        entropy_ind_up_step_cb = cb.affect!
        setup = [
            "scaling" => first(entropy_ind_up_step_cb.scaling)
        ]
        summary_box(io, "EntropyIndicatorUpdateStepCB", setup)
    end
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition,
                                        Affect! <: EntropyIndicatorUpdateStepCB}
    return nothing
end

# this method is called to determine whether the callback should be activated
function (::EntropyIndicatorUpdateStepCB)(u, t, integrator)
    return true
end

function update_target_decay!(indicator::IndicatorEntropyDecay, scaling,
                              du, u, t, semi::AbstractSemidiscretization)
    @unpack target_decay, n_cells_fluxdiff_threaded = indicator

    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)

    dS = analyze(entropy_timederivative, du, u, t,
                 mesh, equations, dg, cache)

    if dS > 0
        # Number of cells calculated with flux differencing in this stage
        n_cells_fd = sum(n_cells_fluxdiff_threaded)
        # Number of cells calculated with weak form in this stage
        n_cells_wf = nelements(dg, cache) - n_cells_fd

        if n_cells_wf > 0
            target_decay = -dS / n_cells_wf * scaling
        end
    end

    return nothing
end

# this method is called when the callback is activated
function (entropy_ind_up_step_cb::EntropyIndicatorUpdateStepCB)(integrator)
    @trixi_timeit timer() "IndicatorEntropyDecay correction" begin
        @unpack indicator, scaling = entropy_ind_up_step_cb

        semi = integrator.p
        u = wrap_array(integrator.u, semi)

        # Get temporary storage for the time derivative
        du_ode = first(get_tmp_cache(integrator))
        # Re-compute update
        t = integrator.t
        @notimeit timer() integrator.f(du_ode, integrator.u, semi, t)

        du = wrap_array(du_ode, semi)
        update_target_decay!(indicator, scaling, du, u, t, semi)
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end
end # @muladd
