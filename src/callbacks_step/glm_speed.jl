
"""
    GlmSpeedCallback(; glm_scale=0.5, cfl=1.0)

Update the divergence cleaning wave speed `c_h` according to the time step
computed in [`StepsizeCallback`](@ref) for the ideal GLM-MHD equations.
"""
struct GlmSpeedCallback{RealT<:Real}
  glm_scale::RealT
  cfl::RealT
end


function Base.show(io::IO, cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:GlmSpeedCallback}
  glm_speed_callback = cb.affect!
  @unpack glm_scale, cfl = glm_speed_callback
  print(io, "GlmSpeedCallback(glm_scale=", glm_scale, ", cfl=", cfl, ")")
end


function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{Condition,Affect!}) where {Condition, Affect!<:GlmSpeedCallback}
  if get(io, :compact, false)
    show(io, cb)
  else
    glm_speed_callback = cb.affect!

    setup = [
             "GLM wave speed scaling" => glm_speed_callback.glm_scale,
             "Expected CFL number" => glm_speed_callback.cfl,
            ]
    summary_box(io, "GlmSpeedCallback", setup)
  end
end


function GlmSpeedCallback(; glm_scale=0.5, cfl=1.0)
  # when is the callback activated
  condition = (u, t, integrator) -> true

  glm_speed_callback = GlmSpeedCallback(glm_scale, cfl)

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
  _, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack glm_scale, cfl = glm_speed_callback

  # compute time step for GLM linear advection equation with c_h=1 (redone due to the possible AMR)
  c_h_deltat = calc_dt_for_cleaning_speed(cfl, equations, solver, cache)

  # c_h is proportional to its own time step divided by the complete MHD time step
  equations.c_h = glm_scale * c_h_deltat / dt

  return nothing
end

include("glm_speed_dg.jl")
