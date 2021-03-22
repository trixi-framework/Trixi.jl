function adapt_to_level!(u_ode::AbstractVector, semi, level)
  amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first), base_level=level)
  amr_callback = AMRCallback(semi, amr_controller, interval=0)

  has_changed = amr_callback.affect!(u_ode, semi, 0.0, 0)
  while has_changed
    has_changed = amr_callback.affect!(u_ode, semi, 0.0, 0)
  end

  return u_ode, semi
end

adapt_to_level!(sol::TrixiODESolution, level; kwargs...) = adapt_to_level!(sol.u[end], sol.prob.p, level; kwargs...)
