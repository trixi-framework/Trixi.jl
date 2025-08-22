# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StepsizeCallback(; cfl=1.0, cfl_diffusive = 0.0,
                     interval = 1)

Set the time step size according to a CFL condition with CFL number `cfl`
if the time integration method isn't adaptive itself.
The keyword argument `cfl` must be either a `Real` number, corresponding to a constant 
CFL number, or a function of time `t` returning a `Real` number.
The latter approach allows for variable CFL numbers that can be used to realize e.g.
a ramp-up of the timestep.

One can additionally supply a diffusive CFL number `cfl_diffusive` to
limit the admissible timestep also respecting diffusive restrictions.
This is only applicable for semidiscretizations of type [`SemidiscretizationHyperbolicParabolic`](@ref).
In this scenario, a number larger than zero or a function of time needs to be supplied. 
By default, `cfl_diffusive` is set to zero which means that only the convective
CFL number is considered.

By default, the timestep will be adjusted at every step.
For different values of `interval`, the timestep will be adjusted every `interval` steps.
"""
mutable struct StepsizeCallback{CflConvectiveType, CflDiffusiveType}
    cfl_convective::CflConvectiveType
    cfl_diffusive::CflDiffusiveType
    interval::Int
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_convective, cfl_diffusive, interval = stepsize_callback
    print(io, "StepsizeCallback(",
          "cfl_convective=", cfl_convective, ", ",
          "cfl_diffusive=", cfl_diffusive, ", ",
          "interval=", interval, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "CFL Convective" => stepsize_callback.cfl_convective,
            "CFL Diffusive" => stepsize_callback.cfl_diffusive,
            "Interval" => stepsize_callback.interval
        ]
        summary_box(io, "StepsizeCallback", setup)
    end
end

function StepsizeCallback(; cfl::Real = 1.0, cfl_diffusive::Real = 0.0,
                          interval = 1)
    stepsize_callback = StepsizeCallback{typeof(cfl), typeof(cfl_diffusive)}(cfl,
                                                                             cfl_diffusive,
                                                                             interval)

    DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

# Compatibility constructors, used e.g. in `EulerAcousticsCouplingCallback`
function StepsizeCallback(cfl_convective)
    StepsizeCallback(cfl = cfl_convective)
end

function StepsizeCallback(cfl_convective, cfl_diffusive)
    StepsizeCallback(cfl = cfl_convective, cfl_diffusive = cfl_diffusive)
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
    @unpack cfl_convective, cfl_diffusive = stepsize_callback

    # Dispatch based on semidiscretization
    dt = @trixi_timeit timer() "calculate dt" calculate_dt(u_ode, t, cfl_convective,
                                                           cfl_diffusive, semi)

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
    @unpack cfl_convective, cfl_diffusive = stepsize_callback
    u_ode = ode.u0
    t = first(ode.tspan)
    semi = ode.p

    return calculate_dt(u_ode, t, cfl_convective, cfl_diffusive, semi)
end

# General case for an abstract single (i.e., non-coupled) semidiscretization
# Case for constant `cfl_number`.
function calculate_dt(u_ode, t, cfl_convective::Real, cfl_diffusive,
                      semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    # Use only convective cfl for non hyperbolic-parabolic semidiscretization
    return cfl_convective * max_dt(u, t, mesh,
                  have_constant_speed(equations), equations,
                  solver, cache)
end
# Case for `cfl_number` as a function of time `t`.
function calculate_dt(u_ode, t, cfl_convective, cfl_diffusive,
                      semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    return cfl_convective(t) * max_dt(u, t, mesh,
                  have_constant_speed(equations), equations,
                  solver, cache)
end

# Case for a hyperbolic-parabolic semidiscretization
# Case for both constant `cfl_convective`, `cfl_diffusive`.
function calculate_dt(u_ode, t, cfl_convective::Real, cfl_diffusive::Real,
                      semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = cfl_convective * max_dt(u, t, mesh,
                           have_constant_speed(equations), equations,
                           solver, cache)

    if cfl_diffusive > 0.0 # Check if diffusive CFL should be considered
        dt_diffusive = cfl_diffusive * max_dt(u, t, mesh,
                              have_constant_diffusivity(equations_parabolic), equations,
                              equations_parabolic, solver, cache)

        return min(dt_convective, dt_diffusive)
    else
        return dt_convective
    end
end

# Case for variable `cfl_convective`, constant `cfl_diffusive`.
function calculate_dt(u_ode, t, cfl_convective, cfl_diffusive::Real,
                      semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = cfl_convective(t) * max_dt(u, t, mesh,
                           have_constant_speed(equations), equations,
                           solver, cache)

    if cfl_diffusive > 0.0 # Check if diffusive CFL should be considered
        dt_diffusive = cfl_diffusive * max_dt(u, t, mesh,
                              have_constant_diffusivity(equations_parabolic), equations,
                              equations_parabolic, solver, cache)

        return min(dt_convective, dt_diffusive)
    else
        return dt_convective
    end
end

# Case for constant `cfl_convective`, variable `cfl_diffusive`.
function calculate_dt(u_ode, t, cfl_convective::Real, cfl_diffusive,
                      semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = cfl_convective * max_dt(u, t, mesh,
                           have_constant_speed(equations), equations,
                           solver, cache)

    # If `cfl_diffusive` is provided as a function of time `t`, we always evaluate
    dt_diffusive = cfl_diffusive(t) * max_dt(u, t, mesh,
                          have_constant_diffusivity(equations_parabolic), equations,
                          equations_parabolic, solver, cache)

    return min(dt_convective, dt_diffusive)
end

# Case for variable `cfl_convective`, variable `cfl_diffusive`.
function calculate_dt(u_ode, t, cfl_convective, cfl_diffusive,
                      semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = cfl_convective(t) * max_dt(u, t, mesh,
                           have_constant_speed(equations), equations,
                           solver, cache)

    # If `cfl_diffusive` is provided as a function of time `t`, we always evaluate
    dt_diffusive = cfl_diffusive(t) * max_dt(u, t, mesh,
                          have_constant_diffusivity(equations_parabolic), equations,
                          equations_parabolic, solver, cache)

    return min(dt_convective, dt_diffusive)
end

include("stepsize_dg1d.jl")
include("stepsize_dg2d.jl")
include("stepsize_dg3d.jl")
end # @muladd
