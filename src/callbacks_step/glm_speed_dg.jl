# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function calc_dt_for_cleaning_speed(cfl::Real, mesh,
                                    equations::Union{AbstractIdealGlmMhdEquations, AbstractIdealGlmMhdMulticomponentEquations}, dg::DG, cache)
# compute time step for GLM linear advection equation with c_h=1 for the DG discretization on
# Cartesian meshes
  max_scaled_speed_for_c_h = maximum(cache.elements.inverse_jacobian) * ndims(equations)
  # OBS! This depends on the implementation details of the StepsizeCallback and needs to be adapted
  #      as well if that callback changes.
  return cfl * 2 / (nnodes(dg) * max_scaled_speed_for_c_h)
end

function calc_dt_for_cleaning_speed(cfl::Real, mesh,
                                    equations::Union{AbstractIdealGlmMhdEquations, AbstractIdealGlmMhdMulticomponentEquations},
                                    dg::DGMulti, cache)
  # compute time step for GLM linear advection equation with c_h=1 for a DGMulti discretization.
  return cfl * estimate_dt(mesh, dg)
end


end # @muladd
