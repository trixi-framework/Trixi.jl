# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StepsizeCallback(; cfl=1.0, cfl_diffusive = 0.0)

Set the time step size according to a CFL condition with CFL number `cfl`
if the time integration method isn't adaptive itself.
One can additionally supply a diffusive CFL number `cfl_diffusive` to
limit the admissible timestep also respecting diffusive restrictions.
In that case, a number larger than zero needs to be supplied. 
By default, `cfl_diffusive` is set to zero which means that only the convective
CFL number is considered.
"""
mutable struct StepsizeCallback{RealT}
    cfl_convective::RealT
    cfl_diffusive::RealT
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_convective, cfl_diffusive = stepsize_callback
    if cfl_diffusive == 0.0
        print(io, "StepsizeCallback(cfl_convective=", cfl_convective, ")")
    else
        print(io, "StepsizeCallback(cfl_convective=", cfl_convective,
              "cfl_diffusive=", cfl_diffusive, ")")
    end
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        if stepsize_callback.cfl_diffusive == 0.0
            setup = [
                "CFL number" => stepsize_callback.cfl_convective,
            ]
        else
            setup = [
                "CFL number" => stepsize_callback.cfl_convective,
                "Diffusive CFL number" => stepsize_callback.cfl_diffusive,
            ]
        end
        summary_box(io, "StepsizeCallback", setup)
    end
end

function StepsizeCallback(; cfl::Real = 1.0, cfl_diffusive::Real = 0.0)
    stepsize_callback = StepsizeCallback(cfl, cfl_diffusive)

    DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                     save_positions = (false, false),
                     initialize = initialize!)
end

function initialize!(cb::DiscreteCallback{Condition, Affect!}, u, t,
                     integrator) where {Condition, Affect! <: StepsizeCallback}
    cb.affect!(integrator)
end

# this method is called to determine whether the callback should be activated
function (stepsize_callback::StepsizeCallback)(u, t, integrator)
    return true
end

# This method is called as callback during the time integration.
@inline function (stepsize_callback::StepsizeCallback)(integrator)
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

# General case for a single semidiscretization
function calculate_dt(u_ode, t, cfl_convective, cfl_diffusive,
                      semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt = cfl_convective * max_dt(u, t, mesh,
                have_constant_speed(equations), equations,
                solver, cache)
end

# Case for a hyperbolic-parabolic semidiscretization
function calculate_dt(u_ode, t, cfl_convective, cfl_diffusive,
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
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt_convective = cfl_convective *
                    max_dt(u, t, mesh, have_constant_speed(equations), equations,
                           solver, cache)

    # Check if diffusive CFL should be considered.
    # NOTE: 
    # For non-zero `cfl_diffusive`, `semi` is expected to be a `SemidiscretizationHyperbolicParabolic`.
    if cfl_diffusive > 0.0
        dt_diffusive = cfl_diffusive *
                       max_dt(u, t, mesh,
                              have_constant_diffusivity(semi.equations_parabolic),
                              equations, semi.equations_parabolic, solver, cache)

        return min(dt_convective, dt_diffusive)
    else
        return dt_convective
    end
end

include("stepsize_dg1d.jl")
include("stepsize_dg2d.jl")
include("stepsize_dg3d.jl")
end # @muladd
