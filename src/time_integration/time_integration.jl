# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Wrapper type for solutions from Trixi.jl's own time integrators, partially mimicking
# SciMLBase.ODESolution
struct TimeIntegratorSolution{tType, uType, P}
    t::tType
    u::uType
    prob::P
end

# Abstract supertype of Trixi.jl's own time integrators for dispatch
abstract type AbstractTimeIntegrator end

# Abstract supertype for the time integration algorithms of Trixi.jl
abstract type AbstractTimeIntegrationAlgorithm end

# get a cache where the RHS can be stored
get_du(integrator::AbstractTimeIntegrator) = integrator.du

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::AbstractTimeIntegrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

# used by adaptive timestepping algorithms in DiffEq
@inline function set_proposed_dt!(integrator::AbstractTimeIntegrator, dt)
    (integrator.dt = dt; integrator.dtcache = dt)

    return nothing
end

# Required e.g. for `glm_speed_callback`
@inline function get_proposed_dt(integrator::AbstractTimeIntegrator)
    return integrator.dt
end

function initialize_callbacks!(callbacks::Union{CallbackSet, Nothing},
                               integrator::AbstractTimeIntegrator)
    # initialize callbacks
    if callbacks isa CallbackSet
        foreach(callbacks.continuous_callbacks) do cb
            throw(ArgumentError("Continuous callbacks are unsupported."))
        end
        foreach(callbacks.discrete_callbacks) do cb
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return nothing
end

function handle_callbacks!(callbacks::Union{CallbackSet, Nothing},
                           integrator::AbstractTimeIntegrator)
    # handle callbacks
    if callbacks isa CallbackSet
        foreach(callbacks.discrete_callbacks) do cb
            if cb.condition(integrator.u, integrator.t, integrator)
                cb.affect!(integrator)
            end
            return nothing
        end
    end

    return nothing
end

@inline function check_max_iter!(integrator::AbstractTimeIntegrator)
    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end

    return nothing
end

"""
    Trixi.solve(ode::ODEProblem, alg::AbstractTimeIntegrationAlgorithm;
                dt, callbacks, kwargs...)

Fakes `solve` from https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
"""
function solve(ode::ODEProblem, alg::AbstractTimeIntegrationAlgorithm;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::AbstractTimeIntegrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end

    finalize_callbacks(integrator)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u), prob)
end

# Interface required by DiffEqCallbacks.jl
function DiffEqBase.get_tstops(integrator::AbstractTimeIntegrator)
    return integrator.opts.tstops
end
function DiffEqBase.get_tstops_array(integrator::AbstractTimeIntegrator)
    return get_tstops(integrator).valtree
end
function DiffEqBase.get_tstops_max(integrator::AbstractTimeIntegrator)
    return maximum(get_tstops_array(integrator))
end

function finalize_callbacks(integrator::AbstractTimeIntegrator)
    callbacks = integrator.opts.callback

    if callbacks isa CallbackSet
        foreach(callbacks.discrete_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
        foreach(callbacks.continuous_callbacks) do cb
            cb.finalize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return nothing
end

include("methods_2N.jl")
include("methods_3Sstar.jl")
include("methods_SSP.jl")
include("paired_explicit_runge_kutta/paired_explicit_runge_kutta.jl")
include("relaxation_methods/relaxation_methods.jl")
end # @muladd
