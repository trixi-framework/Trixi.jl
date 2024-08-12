# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StepsizeCallbackHyperbolicParabolic(; cfl_convective=1.0, cfl_diffusive=1.0)

Set the time step size according to a CFL condition with CFL number `cfl`
if the time integration method isn't adaptive itself.
"""
mutable struct StepsizeCallbackHyperbolicParabolic{RealT}
    cfl_convective::RealT
    cfl_diffusive::RealT
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any, <:StepsizeCallbackHyperbolicParabolic})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_convective = stepsize_callback
    @unpack cfl_diffusive = stepsize_callback
    print(io, "StepsizeCallbackHyperbolicParabolic(cfl_convective=", cfl_convective,
          "cfl_diffusive=", cfl_diffusive, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallbackHyperbolicParabolic})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "Convective CFL number" => stepsize_callback.cfl_convective,
            "Diffusive CFL number" => stepsize_callback.cfl_diffusive,
        ]
        summary_box(io, "StepsizeCallbackHyperbolicParabolic", setup)
    end
end

function StepsizeCallbackHyperbolicParabolic(;
                                             cfl_convective::Real = 1.0,
                                             cfl_diffusive::Real = 1.0)
    stepsize_callback = StepsizeCallbackHyperbolicParabolic(cfl_convective,
                                                            cfl_diffusive)

    DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition,
                                        Affect! <: StepsizeCallbackHyperbolicParabolic}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (stepsize_callback::StepsizeCallbackHyperbolicParabolic)(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (stepsize_callback::StepsizeCallbackHyperbolicParabolic)(integrator)
    # TODO: Taal decide, shall we set the time step even if the integrator is adaptive?
    if !integrator.opts.adaptive
        t = integrator.t
        u_ode = integrator.u
        semi = integrator.p
        @unpack cfl_convective, cfl_diffusive = stepsize_callback

        # Dispatch based on semidiscretization
        dt = @trixi_timeit timer() "calculate dt" calculate_dt(u_ode, t,
                                                               cfl_convective,
                                                               cfl_diffusive,
                                                               semi)

        set_proposed_dt!(integrator, dt)
        integrator.opts.dtmax = dt
        integrator.dtcache = dt
    end

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)
    return nothing
end

# Case for a hyperbolic-parabolic semidiscretization
function calculate_dt(u_ode, t, cfl_convective, cfl_diffusive,
                      semi::SemidiscretizationHyperbolicParabolic)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic

    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = cfl_convective * max_dt(u, t, mesh,
                           have_constant_speed(equations),
                           equations, solver, cache)

    dt_diffusive = cfl_diffusive * max_dt(u, t, mesh,
                          have_constant_diffusivity(equations_parabolic),
                          equations_parabolic, solver, cache)

    return min(dt_convective, dt_diffusive)
end

# Time integration methods from the DiffEq ecosystem without adaptive time stepping on their own
# such as `CarpenterKennedy2N54` require passing `dt=...` in `solve(ode, ...)`. Since we don't have
# an integrator at this stage but only the ODE, this method will be used there. It's called in
# many examples in `solve(ode, ..., dt=stepsize_callback(ode), ...)`.
function (cb::DiscreteCallback{Condition, Affect!})(ode::ODEProblem) where {Condition,
                                                                            Affect! <:
                                                                            StepsizeCallbackHyperbolicParabolic
                                                                            }
    stepsize_callback = cb.affect!
    @unpack cfl_convective, cfl_diffusive = stepsize_callback
    u_ode = ode.u0
    t = first(ode.tspan)
    semi = ode.p
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    equations_parabolic = semi.equations_parabolic
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = dt_convective *
                    max_dt(u, t, mesh, have_constant_speed(equations),
                           equations, solver, cache)

    dt_diffusive = dt_diffusive *
                   max_dt(u, t, mesh, have_constant_diffusivity(equations_parabolic),
                          equations_parabolic, solver, cache)

    return min(dt_convective, dt_diffusive)
end
end # @muladd
