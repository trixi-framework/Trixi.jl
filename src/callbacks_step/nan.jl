# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    NaNCallback(analysis_interval=0, nan_interval=analysis_interval÷10)

Callback checking for NaNs in the solution vector `u` every `nan_interval`.
If `analysis_interval ≂̸ 0`, the output is omitted every
`analysis_interval` time steps.
"""
mutable struct NaNCallback
    nan_interval::Int
    analysis_interval::Int
end

function NaNCallback(; analysis_interval = 0,
                       nan_interval = analysis_interval ÷ 10)
    nan_callback = NaNCallback(nan_interval, analysis_interval)

    DiscreteCallback(nan_callback, nan_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:NaNCallback})
    @nospecialize cb # reduce precompilation time

    nan_callback = cb.affect!
    print(io, "NaNCallback(nan_interval=", nan_callback.nan_interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:NaNCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        nan_callback = cb.affect!

        setup = [
            "interval" => nan_callback.nan_interval,
        ]
        summary_box(io, "NaNCallback", setup)
    end
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: NaNCallback}
    nan_callback = cb.affect!
    return nothing
end

# this method is called to determine whether the callback should be activated
function (nan_callback::NaNCallback)(u, t, integrator)
    @unpack nan_interval, analysis_interval = nan_callback

    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    return nan_interval > 0 && ((integrator.stats.naccept % nan_interval == 0 &&
             !(integrator.stats.naccept == 0 && integrator.iter > 0) &&
             (analysis_interval == 0 ||
              integrator.stats.naccept % analysis_interval != 0)) ||
            isfinished(integrator))
end

# this method is called when the callback is activated
function (nan_callback::NaNCallback)(integrator)
    if any(isnan, integrator.u)
        error("NaN detected in the solution vector `u` at time $(integrator.t)")
    end
    return nothing
end
end # @muladd
