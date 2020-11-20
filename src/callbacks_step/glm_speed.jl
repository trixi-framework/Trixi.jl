
"""
    GlmSpeedCallback(; glm_scale=0.5)

Update the divergence cleaning wave speed c_h according to the time step
computed in StepsizeCallback for the ideal GLM-MHD equations.
"""
mutable struct GlmSpeedCallback{RealT}
  glm_scale::RealT
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:GlmSpeedCallback}
  glm_speed_callback = cb.affect!
  @unpack glm_scale = glm_speed_callback
  print(io, "GlmSpeedCallback(glm_scale=", glm_scale, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:GlmSpeedCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    glm_speed_callback = cb.affect!

    setup = [
             "GLM wave speed scaling" => glm_speed_callback.glm_scale,
            ]
    summary_box(io, "GlmSpeedCallback", setup)
  end
end


function GlmSpeedCallback(; glm_scale::Real=0.5)
  # when is the callback activated
  condition = (u, t, integrator) -> true

  glm_speed_callback = GlmSpeedCallback(glm_scale)

  DiscreteCallback(condition, glm_speed_callback,
                   save_positions=(false,false),
                   initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u, t, integrator) where {Condition, Affect!<:GlmSpeedCallback}
  cb.affect!(integrator)
end


# This method is called as callback after the StepsizeCallback during the time integration.
@inline function (glm_speed_callback::GlmSpeedCallback)(integrator)

  dt = integrator.dtcache
  semi = integrator.p
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack glm_scale = glm_speed_callback

  # compute time step for ONLY the GLM linear advection equation with c_h = 1
  # must be redone each time due to the AMR possibility
  max_scaled_speed_for_c_h = nextfloat(zero(dt))
  for element in eachelement(solver, cache)
    inv_jacobian = cache.elements.inverse_jacobian[element]
    max_scaled_speed_for_c_h = max(max_scaled_speed_for_c_h, ndims(semi.equations) * inv_jacobian)
  end
  cfl = 1.0 # FIXME: This is a hack beacuse I was not sure how to get access to this information from StepsizeCallback
  c_h_deltat = cfl * 2 / (nnodes(solver) * max_scaled_speed_for_c_h)

  # c_h is proportional to its own time step divided by the complete mhd time step
  equations.c_h = 0.5 * c_h_deltat / dt

  return nothing
end
