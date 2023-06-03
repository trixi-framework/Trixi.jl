# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


"""
    AntidiffusiveStage()

Perform antidiffusive stage for IDP limiting.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct AntidiffusiveStage end

function (antidiffusive_stage!::AntidiffusiveStage)(u_ode, integrator, stage)

  antidiffusive_stage!(u_ode, integrator.p, integrator.t, integrator.dt, integrator.p.solver.volume_integral)
end

(::AntidiffusiveStage)(u_ode, semi, t, dt, volume_integral::AbstractVolumeIntegral) = nothing

function (antidiffusive_stage!::AntidiffusiveStage)(u_ode, semi, t, dt, volume_integral::VolumeIntegralShockCapturingSubcell)

  @trixi_timeit timer() "antidiffusive_stage!" antidiffusive_stage!(u_ode, semi, t, dt, volume_integral.indicator)
end

function (antidiffusive_stage!::AntidiffusiveStage)(u_ode, semi, t, dt, indicator::IndicatorIDP)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)

  u = wrap_array(u_ode, mesh, equations, solver, cache)

  @trixi_timeit timer() "alpha calculation" semi.solver.volume_integral.indicator(u, semi, solver, t, dt)

  perform_IDP_correction(u, dt, mesh, equations, solver, cache)

  return nothing
end

@inline function perform_IDP_correction(u, dt, mesh::TreeMesh2D, equations, dg, cache)
  @unpack inverse_weights = dg.basis
  @unpack antidiffusive_flux1, antidiffusive_flux2 = cache.ContainerAntidiffusiveFlux2D
  @unpack alpha1, alpha2 = dg.volume_integral.indicator.cache.ContainerShockCapturingIndicator

  # Loop over blended DG-FV elements
  @threaded for element in eachelement(dg, cache)
    inverse_jacobian = -cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      # Note: antidiffusive_flux1[v, i, xi, element] = antidiffusive_flux2[v, xi, i, element] = 0 for all i in 1:nnodes and xi in {1, nnodes+1}
      alpha_flux1     = (1.0 - alpha1[i,   j, element]) * get_node_vars(antidiffusive_flux1, equations, dg, i,   j, element)
      alpha_flux1_ip1 = (1.0 - alpha1[i+1, j, element]) * get_node_vars(antidiffusive_flux1, equations, dg, i+1, j, element)
      alpha_flux2     = (1.0 - alpha2[i,   j, element]) * get_node_vars(antidiffusive_flux2, equations, dg, i,   j, element)
      alpha_flux2_jp1 = (1.0 - alpha2[i, j+1, element]) * get_node_vars(antidiffusive_flux2, equations, dg, i, j+1, element)

      for v in eachvariable(equations)
        u[v, i, j, element] += dt * inverse_jacobian * (inverse_weights[i] * (alpha_flux1_ip1[v] - alpha_flux1[v]) +
                                                        inverse_weights[j] * (alpha_flux2_jp1[v] - alpha_flux2[v]) )
      end
    end
  end

  return nothing
end

init_callback(callback::AntidiffusiveStage, semi) = nothing

finalize_callback(antidiffusive_stage!::AntidiffusiveStage, semi) = nothing


end # @muladd
