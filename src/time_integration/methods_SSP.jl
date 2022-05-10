# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Abstract base type for time integration schemes of explicit strong stabilitypreserving (SSP)
# Runge-Kutta (RK) methods. They are high-order time discretizations that guarantee the TVD property.
abstract type SimpleAlgorithmSSP end


"""
    SimpleSSPRK33()

The third-order SSP Runge-Kutta method of
    Shu, Osher (1988) Efficient Implementation of Essentially Non-oscillatory Shock-Capturing Schemes, eq. 2.18.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct SimpleSSPRK33 <: SimpleAlgorithmSSP
  a::SVector{3, Float64}
  b::SVector{3, Float64}
  c::SVector{3, Float64}

  function SimpleSSPRK33()
    a = SVector(1.0, 1/4, 2/3)
    b = SVector(1.0, 1/4, 2/3)
    c = SVector(0.0, 1.0, 1/2)

    # Butcher tableau
    #   c |       a
    #   0 |
    #   1 |   1
    # 1/2 | 1/4  1/4
    # --------------------
    #   b | 1/6  1/6  2/3

    new(a, b, c)
  end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleIntegratorSSPOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function SimpleIntegratorSSPOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  SimpleIntegratorSSPOptions{typeof(callback)}(
    callback, false, Inf, maxiters, [last(tspan)])
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct SimpleIntegratorSSP{RealT<:Real, uType, Params, Sol, Alg, SimpleIntegratorSSPOptions}
  u::uType
  du::uType
  u_safe::uType
  u_old::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  alg::Alg
  opts::SimpleIntegratorSSPOptions
  finalstep::Bool # added for convenience
end

# Forward integrator.destats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleIntegratorSSP, field::Symbol)
  if field === :destats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

"""
    solve_IDP(ode, semi; dt, callbacks, kwargs...)

The following structures and methods provide a implementation of the third-order SSP Runge-Kutta
method [`SimpleSSPRK33`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function solve_IDP(ode::ODEProblem, semi; alg=SimpleSSPRK33(), dt, callback=nothing, kwargs...)
  u = copy(ode.u0)
  du = similar(u)
  u_safe = similar(u)
  u_old = similar(u)
  t = first(ode.tspan)
  iter = 0
  integrator = SimpleIntegratorSSP(u, du, u_safe, u_old, t, dt, zero(dt), iter, ode.p,
                  (prob=ode,), alg,
                  SimpleIntegratorSSPOptions(callback, ode.tspan; kwargs...), false)

  # Resize antidiffusive fluxes
  resize!(integrator.p.cache.ContainerFCT2D, nelements(integrator.p.solver, integrator.p.cache))
  # Resize alpha
  resize!(integrator.p.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator,
              nelements(integrator.p.solver, integrator.p.cache))

  if callback isa CallbackSet
    for cb in callback.continuous_callbacks
      error("unsupported")
    end
    for cb in callback.discrete_callbacks
      cb.initialize(cb, integrator.u, integrator.t, integrator)
    end
  elseif !isnothing(callback)
    error("unsupported")
  end

  solve!(integrator)
end

function solve!(integrator::SimpleIntegratorSSP)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  integrator.finalstep = false
  @trixi_timeit timer() "main loop" while !integrator.finalstep
    if isnan(integrator.dt)
      error("time step size `dt` is NaN")
    end

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end || isapprox(integrator.t + integrator.dt, t_end)
      integrator.dt = t_end - integrator.t
      terminate!(integrator)
    end

    @. integrator.u_safe = integrator.u
    for i in 1:length(alg.a)
      @trixi_timeit timer() "RK stage" begin
        prob.f(integrator.du, integrator.u_safe, integrator.p, integrator.t + alg.c[i] * integrator.dt)
        @. integrator.u_old = (1.0 - alg.a[i]) * integrator.u + alg.a[i] * integrator.u_safe
        @. integrator.u_safe = integrator.u_old + alg.b[i] * integrator.dt * integrator.du
      end
      @trixi_timeit timer() "antidiffusive stage" antidiffusive_stage!(integrator.u_safe, integrator.u_old, alg.b[i] * integrator.dt, integrator.p)
    end
    @. integrator.u = integrator.u_safe

    # Note:
    # @. integrator.u_old = (1.0 - alg.a[i]) * integrator.u + alg.a[i] * integrator.u_safe
    # The combination of the macro muladd with the operator @. changes the order of operations knowingly, which
    # results in changed solutions.
    # Moreover, unrolling the for-loop changes the order unexpectedly. Using a cache variable like
    # @. u_tmp = (1.0 - alg.a[i]) * integrator.u
    # @. integrator.u_old = u_tmp + alg.a[i] * integrator.u_safe
    # solves the differences between the (not-)unrolled for-loop versions.

    if integrator.iter == length(integrator.p.solver.volume_integral.indicator.cache.alpha_max_per_timestep)
      new_length = length(integrator.p.solver.volume_integral.indicator.cache.alpha_max_per_timestep) + 200
      resize!(integrator.p.solver.volume_integral.indicator.cache.alpha_max_per_timestep,  new_length)
      resize!(integrator.p.solver.volume_integral.indicator.cache.alpha_mean_per_timestep, new_length)
    end

    integrator.p.solver.volume_integral.indicator.cache.alpha_max_per_timestep[integrator.iter+1] =
        maximum(integrator.p.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator.alpha)
    integrator.p.solver.volume_integral.indicator.cache.alpha_mean_per_timestep[integrator.iter+1] =
        (1/(nnodes(integrator.p.solver)^ndims(integrator.p.equations) * nelements(integrator.p.solver, integrator.p.cache))) *
             sum(integrator.p.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator.alpha)

    integrator.iter += 1
    integrator.t += integrator.dt

    # handle callbacks
    if callbacks isa CallbackSet
      for cb in callbacks.discrete_callbacks
        if cb.condition(integrator.u, integrator.t, integrator)
          cb.affect!(integrator)
        end
      end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
      @warn "Interrupted. Larger maxiters is needed."
      terminate!(integrator)
    end
  end

  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                              (prob.u0, integrator.u),
                              integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegratorSSP) = integrator.du
get_tmp_cache(integrator::SimpleIntegratorSSP) = (integrator.u_safe, integrator.u_old)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::SimpleIntegratorSSP, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::SimpleIntegratorSSP, dt)
  integrator.dt = dt
end

# stop the time integration
function terminate!(integrator::SimpleIntegratorSSP)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)

  resize!(integrator.p.solver.volume_integral.indicator.cache.alpha_max_per_timestep,  integrator.iter+1)
  resize!(integrator.p.solver.volume_integral.indicator.cache.alpha_mean_per_timestep, integrator.iter+1)
end

# used for AMR
function Base.resize!(integrator::SimpleIntegratorSSP, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)
  resize!(integrator.u_old, new_size)

  # Resize antidiffusive fluxes
  resize!(integrator.p.cache.ContainerFCT2D, nelements(integrator.p.solver, integrator.p.cache))
  # Resize alpha
  resize!(integrator.p.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator,
              nelements(integrator.p.solver, integrator.p.cache))

end


end # @muladd
