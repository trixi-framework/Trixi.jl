# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StepsizeCallback(; cfl=1.0, interval = 1)

Set the time step size according to a CFL condition with CFL number `cfl`
if the time integration method isn't adaptive itself.

The supplied keyword argument `cfl` must be either a `Real` number or
a function of time `t` returning a `Real` number.
By default, the timestep will be adjusted at every step.
For different values of `interval`, the timestep will be adjusted every `interval` steps.
"""
mutable struct StepsizeCallback{CflType}
    cfl_number::CflType
    interval::Int
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_number, interval = stepsize_callback
    print(io, "StepsizeCallback(",
          "cfl_number=", cfl_number, ", ",
          "interval=", interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = ["CFL number" => stepsize_callback.cfl_number
                 "Interval" => stepsize_callback.interval]
        summary_box(io, "StepsizeCallback", setup)
    end
end

function StepsizeCallback(; cfl = 1.0, interval = 1)
    stepsize_callback = StepsizeCallback{typeof(cfl)}(cfl, interval)

    DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

# Compatibility constructor
function StepsizeCallback(cfl)
    StepsizeCallback{typeof(cfl)}(cfl, 1)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: StepsizeCallback}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (stepsize_callback::StepsizeCallback)(u, t, integrator)
    @unpack interval = stepsize_callback

    # Although the CFL-based timestep is usually not used with
    # adaptive time integration methods, we still check the accepted steps `naccept` here.
    return interval > 0 && integrator.stats.naccept % interval == 0
end

# This method is called as callback during the time integration.
@inline function (stepsize_callback::StepsizeCallback)(integrator)
    if integrator.opts.adaptive
        throw(ArgumentError("The `StepsizeCallback` has no effect when using an adaptive time integration scheme. Please remove the `StepsizeCallback` or set `adaptive = false` in `solve`."))
    end

    t = integrator.t
    u_ode = integrator.u
    semi = integrator.p
    @unpack cfl_number = stepsize_callback

    # Dispatch based on semidiscretization
    dt = @trixi_timeit timer() "calculate dt" calculate_dt(u_ode, t, cfl_number, semi)

    set_proposed_dt!(integrator, dt)
    integrator.opts.dtmax = dt
    integrator.dtcache = dt

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

# Time integration methods from the DiffEq ecosystem without adaptive time stepping on their own
# such as `CarpenterKennedy2N54` require passing `dt=...` in `solve(ode, ...)`. Since we don't have
# an integrator at this stage but only the ODE, this method will be used there. It's called in
# many examples in `solve(ode, ..., dt=stepsize_callback(ode), ...)`.
function (cb::DiscreteCallback{Condition, Affect!})(ode::ODEProblem) where {Condition,
                                                                            Affect! <:
                                                                            StepsizeCallback
                                                                            }
    stepsize_callback = cb.affect!
    @unpack cfl_number = stepsize_callback
    u_ode = ode.u0
    t = first(ode.tspan)
    semi = ode.p

    dt = calculate_dt(u_ode, t, cfl_number, semi)
end

# General case for a single (i.e., non-coupled) semidiscretization
# Case for constant `cfl_number`.
function calculate_dt(u_ode, t, cfl_number::Real, semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt = cfl_number * max_dt(u, t, mesh,
                have_constant_speed(equations), semi, equations, solver, cache,
                solver.volume_integral)
end
# Case for `cfl_number` as a function of time `t`.
function calculate_dt(u_ode, t, cfl_number, semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt = cfl_number(t) * max_dt(u, t, mesh,
                have_constant_speed(equations), semi, equations, solver, cache,
                solver.volume_integral)
end

function max_dt(u, t, mesh, constant_speed, semi, equations, solver, cache,
                volume_integral::AbstractVolumeIntegral)
    max_dt(u, t, mesh, constant_speed, equations, solver, cache)
end

@inline function max_dt(u, t, mesh,
                        constant_speed, semi, equations, solver, cache,
                        volume_integral::VolumeIntegralSubcellLimiting)
    @unpack limiter = volume_integral
    if limiter isa SubcellLimiterIDP && !limiter.bar_states
        return max_dt(u, t, mesh, constant_speed, equations, solver, cache)
    else
        return max_dt(u, t, mesh, constant_speed, equations, semi, solver, cache,
                      limiter)
    end
end

include("stepsize_dg1d.jl")
include("stepsize_dg2d.jl")
include("stepsize_dg3d.jl")
end # @muladd
