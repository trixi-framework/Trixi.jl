
"""
    StepsizeCallback(; cfl)

Set the time step size according to a CFL condition with CFL number `cfl`
if the time integration method isn't adaptive itself. When using a semidiscretization that wraps
multiple solvers, you might need to provide a tuple of CFL numbers.
"""
mutable struct StepsizeCallback{RealT}
  # FIXME: RealT -> CFLNumber
  cfl_number::RealT
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:StepsizeCallback}
  stepsize_callback = cb.affect!
  @unpack cfl_number = stepsize_callback
  print(io, "StepsizeCallback(cfl_number=", cfl_number, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:StepsizeCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    stepsize_callback = cb.affect!

    setup = [
             "CFL number" => stepsize_callback.cfl_number,
            ]
    summary_box(io, "StepsizeCallback", setup)
  end
end


function StepsizeCallback(; cfl)

  stepsize_callback = StepsizeCallback(cfl)

  DiscreteCallback(stepsize_callback, stepsize_callback, # the first one is the condition, the second the affect!
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:StepsizeCallback}
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

    dt = @timeit_debug timer() "calculate dt" max_dt(u_ode, t, cfl_number, semi)

    set_proposed_dt!(integrator, dt)
    integrator.opts.dtmax = dt
    integrator.dtcache = dt
  end

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end


# Time integration methods from the DiffEq ecosystem without adaptive time stepping on their own
# such as `CarpenterKennedy2N54` require passing `dt=...` in `solve(ode, ...)`. Since we don't have
# an integrator at this stage but only the ODE, this method will be used there. It's called in
# many examples in `solve(ode, ..., dt=stepsize_callback(ode), ...)`.
function (cb::DiscreteCallback{Condition,Affect!})(ode::ODEProblem) where {Condition, Affect!<:StepsizeCallback}
  stepsize_callback = cb.affect!
  @unpack cfl_number = stepsize_callback
  u_ode = ode.u0
  t = first(ode.tspan)
  semi = ode.p

  return max_dt(u_ode, t, cfl_number, semi)
end


include("stepsize_dg1d.jl")
include("stepsize_dg2d.jl")
include("stepsize_dg3d.jl")
