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
end

# get a cache where the RHS can be stored
get_du(integrator) = integrator.du
get_tmp_cache(integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator, dt)
    integrator.dt = dt
end

# Required e.g. for `glm_speed_callback` 
function get_proposed_dt(integrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

function solve!(integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end

    finalize_callbacks(integrator)

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

include("methods_2N.jl")
include("methods_3Sstar.jl")
include("methods_SSP.jl")
include("paired_explicit_runge_kutta/paired_explicit_runge_kutta.jl")
include("relaxation_methods/relaxation_methods.jl")
end # @muladd
