
"""
    LBMCollisionCallback()

Apply the LBM collision operator before each time step.
"""
struct LBMCollisionCallback end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:LBMCollisionCallback}
  lbm_collision_callback = cb.affect!
  print(io, "LBMCollisionCallback()")
end


function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:LBMCollisionCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    lbm_collision_callback = cb.affect!

    summary_box(io, "LBMCollisionCallback")
  end
end


function LBMCollisionCallback(; )
  # when is the callback activated
  condition = (u, t, integrator) -> true

  lbm_collision_callback = LBMCollisionCallback(glm_scale, cfl)

  DiscreteCallback(condition, lbm_collision_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:LBMCollisionCallback}
  cb.affect!(integrator)
end


# This method is called as callback after the StepsizeCallback during the time integration.
@inline function (lbm_collision_callback::LBMCollisionCallback)(integrator)

  dt = get_proposed_dt(integrator)
  semi = integrator.p
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u_ode = integrator.u
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  apply_collision!(collision_bgk, u, dt, mesh, equations, solver, cache)

  return nothing
end

include("lbm_collision_dg.jl")
