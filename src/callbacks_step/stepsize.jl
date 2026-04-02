# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StepsizeCallback(; cfl=1.0, cfl_parabolic = 0.0,
                     interval = 1)

Set the time step size according to a CFL condition with hyperbolic CFL number `cfl`
if the time integration method isn't adaptive itself.
The hyperbolic CFL number `cfl` must be either a `Real` number, corresponding to a constant
CFL number, or a function of time `t` returning a `Real` number.
The latter approach allows for variable CFL numbers that can be used to realize, e.g.,
a ramp-up of the time step.

One can additionally supply a parabolic CFL number `cfl_parabolic` to
limit the admissible timestep also respecting parabolic restrictions.
This is only applicable for semidiscretizations of type [`SemidiscretizationHyperbolicParabolic`](@ref).
To enable checking for parabolic timestep restrictions, provide a value greater than zero for `cfl_parabolic`.
By default, `cfl_parabolic` is set to zero which means that only the hyperbolic CFL number `cfl` is considered.
The keyword argument `cfl_parabolic` must be either a `Real` number, corresponding to a constant
parabolic CFL number, or a function of time `t` returning a `Real` number.

By default, the timestep will be adjusted at every step.
For different values of `interval`, the timestep will be adjusted every `interval` steps.
"""
struct StepsizeCallback{CflHyperbolicType, CflParabolicType}
    cfl_hyperbolic::CflHyperbolicType
    cfl_parabolic::CflParabolicType
    interval::Int
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_hyperbolic, cfl_parabolic, interval = stepsize_callback
    print(io, "StepsizeCallback(",
          "cfl_hyperbolic=", cfl_hyperbolic, ", ",
          "cfl_parabolic=", cfl_parabolic, ", ",
          "interval=", interval, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "CFL Hyperbolic" => stepsize_callback.cfl_hyperbolic,
            "CFL Parabolic" => stepsize_callback.cfl_parabolic,
            "Interval" => stepsize_callback.interval
        ]
        summary_box(io, "StepsizeCallback", setup)
    end
end

function StepsizeCallback(; cfl = 1.0, cfl_parabolic = 0.0,
                          interval = 1)
    # Convert plain real numbers to functions for unified treatment
    cfl_hyp = isa(cfl, Real) ? Returns(cfl) : cfl
    cfl_para = isa(cfl_parabolic, Real) ? Returns(cfl_parabolic) : cfl_parabolic
    stepsize_callback = StepsizeCallback{typeof(cfl_hyp), typeof(cfl_para)}(cfl_hyp,
                                                                            cfl_para,
                                                                            interval)

    return DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                            save_positions = (false, false),
                            initialize = initialize!)
end

# Compatibility constructor used in `EulerAcousticsCouplingCallback`
function StepsizeCallback(cfl_hyperbolic)
    RealT = typeof(cfl_hyperbolic)
    return StepsizeCallback{RealT, RealT}(cfl_hyperbolic, zero(RealT), 1)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: StepsizeCallback}
    return cb.affect!(integrator)
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
    @unpack cfl_hyperbolic, cfl_parabolic = stepsize_callback

    backend = trixi_backend(u_ode)
    # Dispatch based on semidiscretization
    dt = @trixi_timeit_ext backend timer() "calculate dt" calculate_dt(u_ode, t,
                                                                       cfl_hyperbolic,
                                                                       cfl_parabolic,
                                                                       semi)

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
    @unpack cfl_hyperbolic, cfl_parabolic = stepsize_callback
    u_ode = ode.u0
    t = first(ode.tspan)
    semi = ode.p

    return calculate_dt(u_ode, t, cfl_hyperbolic, cfl_parabolic, semi)
end

# General case for an abstract single (i.e., non-coupled) semidiscretization
function calculate_dt(u_ode, t, cfl_hyperbolic, cfl_parabolic,
                      semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    return cfl_hyperbolic(t) * max_dt(u, t, mesh,
                  have_constant_speed(equations), equations,
                  solver, cache)
end

# For Euler-Acoustic simulations with `EulerAcousticsCouplingCallback`
function calculate_dt(u_ode, t, cfl_hyperbolic::Real, cfl_parabolic::Real,
                      semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    return cfl_hyperbolic * max_dt(u, t, mesh,
                  have_constant_speed(equations), equations,
                  solver, cache)
end

# Case for a hyperbolic-parabolic semidiscretization
function calculate_dt(u_ode, t, cfl_hyperbolic, cfl_parabolic,
                      semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_hyperbolic = cfl_hyperbolic(t) * max_dt(u, t, mesh,
                           have_constant_speed(equations), equations,
                           solver, cache)

    cfl_para = cfl_parabolic(t)
    if cfl_para > 0 # Check if parabolic CFL should be considered
        dt_parabolic = cfl_para * max_dt(u, t, mesh,
                              have_constant_diffusivity(equations_parabolic), equations,
                              equations_parabolic, solver, cache)

        return min(dt_hyperbolic, dt_parabolic)
    else
        return dt_hyperbolic
    end
end

function calc_max_scaled_speed(backend::Nothing, u, mesh, constant_speed, equations, dg,
                               cache)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    max_scaled_speed = zero(eltype(u))
    @batch reduction=(max, max_scaled_speed) for element in eachelement(dg, cache)
        max_lambda = max_scaled_speed_per_element(u, typeof(mesh), constant_speed,
                                                  equations, dg,
                                                  contravariant_vectors,
                                                  inverse_jacobian,
                                                  element)
        # Use `Base.max` to prevent silent failures, as `max` from `@fastmath` doesn't propagate
        # `NaN`s properly. See https://github.com/trixi-framework/Trixi.jl/pull/2445#discussion_r2336812323
        max_scaled_speed = Base.max(max_scaled_speed, max_lambda)
    end
    return max_scaled_speed
end

function calc_max_scaled_speed(backend::Backend, u, mesh, constant_speed, equations, dg,
                               cache)
    @unpack contravariant_vectors, inverse_jacobian = cache.elements

    num_elements = nelements(dg, cache)
    max_scaled_speeds = allocate(backend, eltype(u), num_elements)

    kernel! = max_scaled_speed_KAkernel!(backend)
    kernel!(max_scaled_speeds, u, typeof(mesh), constant_speed, equations, dg,
            contravariant_vectors,
            inverse_jacobian;
            ndrange = num_elements)

    return maximum(max_scaled_speeds)
end

@kernel function max_scaled_speed_KAkernel!(max_scaled_speeds, u, MeshT, constant_speed,
                                            equations,
                                            dg, contravariant_vectors, inverse_jacobian)
    element = @index(Global)
    max_scaled_speeds[element] = max_scaled_speed_per_element(u, MeshT, constant_speed,
                                                              equations, dg,
                                                              contravariant_vectors,
                                                              inverse_jacobian,
                                                              element)
end

include("stepsize_dg1d.jl")
include("stepsize_dg2d.jl")
include("stepsize_dg3d.jl")
end # @muladd
