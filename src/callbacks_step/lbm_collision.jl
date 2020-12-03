
"""
    LBMCollisionCallback()

Apply the LBM collision operator before each time step.
"""
function LBMCollisionCallback()
  DiscreteCallback(lbm_collision_callback, lbm_collision_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end

# Always execute collision step after a time step, but not after the last step
lbm_collision_callback(u, t, integrator) = !isfinished(integrator)


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:typeof(lbm_collision_callback)}
  print(io, "LBMCollisionCallback()")
end


function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:typeof(lbm_collision_callback)}
  if get(io, :compact, false)
    show(io, cb)
  else
    summary_box(io, "LBMCollisionCallback")
  end
end


# Execute collision step once in the very beginning
function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:typeof(lbm_collision_callback)}
  cb.affect!(integrator)
end


# This method is called as callback after the StepsizeCallback during the time integration.
@inline function lbm_collision_callback(integrator)

  dt = get_proposed_dt(integrator)
  semi = integrator.p
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack collision_op = equations

  u_ode = integrator.u
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  apply_collision!(collision_op, u, dt, mesh, equations, solver, cache)

  return nothing
end

include("lbm_collision_dg.jl")
