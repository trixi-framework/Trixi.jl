# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


  """
  Subcel FV positivity limiter for the DGSEM

  Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations
  - Rueda-Ramirez, Gassner (2021),
    A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations
    [doi: 10.23967/wccm-eccomas.2020.038](https://doi.org/10.23967/wccm-eccomas.2020.038)

  """
  mutable struct PositivityPreservingLimiterRuedaGassner{RealT<:Real, Cache}
    beta::RealT
    stage::Int32
    cache::Cache
    t::RealT
    dt::RealT
  end
  

  function PositivityPreservingLimiterRuedaGassner(semi::AbstractSemidiscretization; beta)
    cache = create_cache(semi)
    initial_stage = 1
    t = 0.0
    dt = 0.0
    PositivityPreservingLimiterRuedaGassner{typeof(beta), typeof(cache)}(beta, initial_stage, cache, t, dt)

  end


  function create_cache(semi::AbstractSemidiscretization)
    @unpack volume_flux_fv = semi.solver.volume_integral 
    @unpack basis = semi.solver 
    @unpack surface_flux = semi.solver.surface_integral
    @unpack mesh, equations, initial_condition = semi 

    volume_integral = VolumeIntegralPureLGLFiniteVolume(volume_flux_fv)
    solver_fv = DGSEM(basis, volume_flux_fv, volume_integral)
    semi_fv = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_fv)

    tolerance = 1e-15 # allowed tolerance
    iterations_newton = 10 # number of Newton Iterations for pressure correction
    alpha_max = 1 # Allow corrections up to 1 

    # Array for safe Solution with same size as u 
    # Size = # conservative variables * # nodes in element * # of elements
    num_vars = 2 + ndims(mesh) 
    u_safe = zeros(num_vars * nnodes(solver_fv)^(ndims(mesh)) * nelements(solver_fv, semi.cache))
   
    u_latest_stage = similar(u_safe)
    tmp_lates_stage = similar(u_safe)

    node_dg = zeros(2 + ndims(mesh))
    node_tmp = similar(node_dg)
    du_dα = similar(node_dg)
    dp_du = similar(node_dg)

    return (; semi_fv, tolerance, iterations_newton, alpha_max, tmp_lates_stage,
               u_safe, u_latest_stage, node_dg, node_tmp, du_dα, dp_du)
  end

  # Get pure FV solution for Carpenter-Kennedy time-integration scheme
  # rhs! allocates a lot of memory!
  function get_usafe!(u_safe, limiter!, integrator)
    @unpack stage = limiter!
    @unpack semi_fv, u_latest_stage, tmp_lates_stage  = limiter!.cache
    @unpack t, dt, uprev, f = integrator
    @unpack k, tmp = integrator.cache
    @unpack A2end, B1, B2end, c2end = integrator.cache.tab

    if stage == 1
      rhs!(k, uprev, semi_fv, t)
      for i in 1:length(uprev)
        u_safe[i] = uprev[i] + B1 * dt * k[i]
      end
    else
      if stage == 5 
        dt =limiter!.dt
        t = limiter!.t
      end
      rhs!(k, u_latest_stage, semi_fv, t + c2end[stage-1] * dt)
      # tmp_lates_stage += dt * k
      for d in 1:length(u_safe)
        tmp_lates_stage[d] = tmp_lates_stage[d] + dt * k[d]
        u_safe[d] = u_latest_stage[d] + B2end[stage-1] * tmp_lates_stage[d]
      end
    end
  end

  function (limiter!::PositivityPreservingLimiterRuedaGassner)(
    u_ode, integrator, semi::AbstractSemidiscretization, t)
    @unpack alpha = semi.solver.volume_integral.indicator.cache
    @unpack mesh = semi

    u = wrap_array(u_ode, semi)   

    limiter_rueda_gassner!(u, alpha, mesh, integrator, semi, limiter!)

    # save solution o_ode and tmp in integrator for next stage
    for idx in eachindex(u_ode)
      limiter!.cache.u_latest_stage[idx] = u_ode[idx]
      limiter!.cache.tmp_lates_stage[idx] = integrator.cache.tmp[idx]        
    end

    # Used in Stage 5
    limiter!.dt = integrator.dt
    limiter!.t = integrator.t

    # increase stage for Limiter
    limiter!.stage == 5 ? limiter!.stage = 1 : limiter!.stage+= 1
    return nothing
  end

  include("positivity_alpha_rueda_gassner_dg1d.jl")
  include("positivity_alpha_rueda_gassner_dg2d.jl")

end # @muladd
  