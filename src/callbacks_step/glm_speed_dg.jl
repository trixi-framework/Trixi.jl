
function calc_dt_for_cleaning_speed(cfl::Real, equations::AbstractIdealGlmMhdEquations, dg::DG, cache)
# compute time step for GLM linear advection equation with c_h=1 for the DG discretization on
# Cartesian meshes
  max_scaled_speed_for_c_h = maximum(cache.elements.inverse_jacobian) * ndims(equations)
  return cfl * 2 / (nnodes(dg) * max_scaled_speed_for_c_h)
end
