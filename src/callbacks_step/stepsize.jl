# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    StepsizeCallback(; cfl=1.0)

Set the time step size according to a CFL condition with CFL number `cfl`
if the time integration method isn't adaptive itself.
"""
mutable struct StepsizeCallback{RealT}
    cfl_number::RealT
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    stepsize_callback = cb.affect!
    @unpack cfl_number = stepsize_callback
    print(io, "StepsizeCallback(cfl_number=", cfl_number, ")")
end

function Base.show(io::IO, ::MIME"text/plain",
                   cb::DiscreteCallback{<:Any, <:StepsizeCallback})
    @nospecialize cb # reduce precompilation time

    if get(io, :compact, false)
        show(io, cb)
    else
        stepsize_callback = cb.affect!

        setup = [
            "CFL number" => stepsize_callback.cfl_number
        ]
        summary_box(io, "StepsizeCallback", setup)
    end
end

function StepsizeCallback(; cfl::Real = 1.0)
    stepsize_callback = StepsizeCallback(cfl)

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
        @unpack cfl_number = stepsize_callback

        # Dispatch based on semidiscretization
        dt = @trixi_timeit timer() "calculate dt" calculate_dt(u_ode, t, cfl_number,
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
function calculate_dt(u_ode, t, cfl_number, semi::AbstractSemidiscretization)
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    dt = cfl_number * max_dt(u, t, mesh,
                have_constant_speed(equations), equations,
                solver, cache)
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
    mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
    u = wrap_array(u_ode, mesh, equations, solver, cache)

    return cfl_number *
           max_dt(u, t, mesh, have_constant_speed(equations), equations, solver, cache)
end

include("stepsize_dg1d.jl")
include("stepsize_dg2d.jl")
include("stepsize_dg3d.jl")
end # @muladd
