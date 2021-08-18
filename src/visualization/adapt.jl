# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    adapt_to_mesh_level!(u_ode, semi, level)
    adapt_to_mesh_level!(sol::Trixi.TrixiODESolution, level)

Like [`adapt_to_mesh_level`](@ref), but modifies the solution and parts of the
semidiscretization (mesh and caches) in place.
"""
function adapt_to_mesh_level!(u_ode, semi, level)
  # Create AMR callback with controller that refines everything towards a single level
  amr_controller = ControllerThreeLevel(semi, IndicatorMax(semi, variable=first), base_level=level)
  amr_callback = AMRCallback(semi, amr_controller, interval=0)

  # Adapt mesh until it does not change anymore
  has_changed = amr_callback.affect!(u_ode, semi, 0.0, 0)
  while has_changed
    has_changed = amr_callback.affect!(u_ode, semi, 0.0, 0)
  end

  return u_ode, semi
end

adapt_to_mesh_level!(sol::TrixiODESolution, level) = adapt_to_mesh_level!(sol.u[end], sol.prob.p, level)


"""
    adapt_to_mesh_level(u_ode, semi, level)
    adapt_to_mesh_level(sol::Trixi.TrixiODESolution, level)

Use the regular adaptive mesh refinement routines to adaptively refine/coarsen the solution `u_ode`
with semidiscretization `semi` towards a uniformly refined grid with refinement level `level`. The
solution and semidiscretization are copied such that the original objects remain *unaltered*.

A convenience method accepts an ODE solution object, from which solution and semidiscretization are
extracted as needed.

See also: [`adapt_to_mesh_level!`](@ref)
"""
function adapt_to_mesh_level(u_ode, semi, level)
  # Create new semidiscretization with copy of the current mesh
  mesh, _, _, _ = mesh_equations_solver_cache(semi)
  new_semi = remake(semi, mesh=deepcopy(mesh))

  return adapt_to_mesh_level!(deepcopy(u_ode), new_semi, level)
end

adapt_to_mesh_level(sol::TrixiODESolution, level) = adapt_to_mesh_level(sol.u[end], sol.prob.p, level)


end # @muladd
