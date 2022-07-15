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
    solve(ode; dt, callbacks, kwargs...)

The following structures and methods provide a implementation of the third-order SSP Runge-Kutta
method [`SimpleSSPRK33`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function solve(ode::ODEProblem; alg=SimpleSSPRK33()::SimpleAlgorithmSSP,
               dt, callback=nothing, kwargs...)
  u = copy(ode.u0)
  du = similar(u)
  u_safe = similar(u)
  u_old = similar(u)
  t = first(ode.tspan)
  iter = 0
  integrator = SimpleIntegratorSSP(u, du, u_safe, u_old, t, dt, zero(dt), iter, ode.p,
                  (prob=ode,), alg,
                  SimpleIntegratorSSPOptions(callback, ode.tspan; kwargs...), false)

  # Resize container
  resize!(integrator.p, nelements(integrator.p.solver, integrator.p.cache))

  # initialize callbacks
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
  @unpack indicator = integrator.p.solver.volume_integral
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
    for stage in eachindex(alg.c)
      t_stage = integrator.t + integrator.dt * alg.c[stage]
      prob.f(integrator.du, integrator.u_safe, integrator.p, t_stage)

      @trixi_timeit timer() "Runge-Kutta stage" begin
        @. integrator.u_old = (1.0 - alg.a[stage]) * integrator.u + alg.a[stage] * integrator.u_safe
        @. integrator.u_safe = integrator.u_old + alg.b[stage] * integrator.dt * integrator.du
      end
      @trixi_timeit timer() "Antidiffusive stage" antidiffusive_stage!(integrator.u_safe, integrator.u_old, alg.b[stage] * integrator.dt, integrator.p, indicator)

      # Check that we are within bounds
      if indicator.IDPCheckBounds
        @trixi_timeit timer() "IDP_checkBounds" IDP_checkBounds(integrator.u_safe, integrator.p)
      end
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

    if integrator.iter == length(indicator.cache.alpha_max_per_timestep)
      new_length = length(indicator.cache.alpha_max_per_timestep) + 200
      resize!(indicator.cache.alpha_max_per_timestep,  new_length)
      resize!(indicator.cache.alpha_mean_per_timestep, new_length)
    end

    if indicator isa IndicatorIDP
      indicator.cache.alpha_max_per_timestep[integrator.iter+1] =
          maximum(indicator.cache.ContainerShockCapturingIndicator.alpha)
      indicator.cache.alpha_mean_per_timestep[integrator.iter+1] =
          (1/(nnodes(integrator.p.solver)^ndims(integrator.p.equations) * nelements(integrator.p.solver, integrator.p.cache))) *
              sum(indicator.cache.ContainerShockCapturingIndicator.alpha)
    end

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

  # Check that we are within bounds
  if indicator.IDPCheckBounds
    summary_check_bounds(indicator, integrator.p.equations)
  end

  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u), prob)
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
  resize!(integrator.u_safe, new_size)
  resize!(integrator.u_old, new_size)

  # Resize container
  resize!(integrator.p, new_size)
end

function Base.resize!(semi::AbstractSemidiscretization, new_size)
  # Resize ContainerAntidiffusiveFlux2D
  resize!(semi.cache.ContainerAntidiffusiveFlux2D, new_size)

  # Resize ContainerShockCapturingIndicator
  resize!(semi.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator, new_size)
end

# check deviation from boundaries of IDP indicator
@inline function summary_check_bounds(indicator::IndicatorIDP, equations::CompressibleEulerEquations2D)
  @unpack IDPDensityTVD, IDPPressureTVD, IDPPositivity, IDPSpecEntropy, IDPMathEntropy = indicator
  @unpack idp_bounds_delta_threaded = indicator.cache

  err_bounds = idp_bounds_delta_threaded[1]

  for i in 2:Threads.nthreads()
    for index in 1:length(err_bounds)
      err_bounds[index] = max(err_bounds[index], idp_bounds_delta_threaded[i][index])
    end
  end

  println("─"^100)
  println("Maximum deviation from bounds:")
  println("─"^100)
  counter = 0
  if IDPDensityTVD
    counter += 2
    println("rho:\n- lower bound: ", err_bounds[counter-1], "\n- upper bound: ", err_bounds[counter])
  end
  if IDPPressureTVD
    counter += 2
    println("pressure:\n- lower bound: ", err_bounds[counter-1], "\n- upper bound: ", err_bounds[counter])
  end
  if IDPPositivity && !IDPDensityTVD
    counter += 1
    println("rho:\n- positivity: ", err_bounds[counter])
  end
  if IDPPositivity && !IDPPressureTVD
    counter += 1
    println("pressure:\n- positivity: ", err_bounds[counter])
  end
  if IDPSpecEntropy
    counter += 1
    println("spec. entropy:\n- lower bound: ", err_bounds[counter])
  end
  if IDPMathEntropy
    counter += 1
    println("math. entropy:\n- upper bound: ", err_bounds[counter])
  end
  println("─"^100 * "\n")

  return nothing
end

# check deviation from boundaries of IndicatorMCL
@inline function summary_check_bounds(indicator::IndicatorMCL, equations::CompressibleEulerEquations2D)
  @unpack idp_bounds_delta_threaded = indicator.cache

  err_bounds = idp_bounds_delta_threaded[1]

  for i in 2:Threads.nthreads()
    for index in 1:length(err_bounds)
      err_bounds[index] = max(err_bounds[index], idp_bounds_delta_threaded[i][index])
    end
  end

  println("─"^100)
  println("Maximum deviation from bounds:")
  println("─"^100)
  variables = varnames(cons2cons, equations)
  for v in eachvariable(equations)
    println(variables[v], ":\n- lower bound: ", err_bounds[2*v-1], "\n- upper bound: ", err_bounds[2*v])
  end
  if indicator.IDPPressureTVD
    println("pressure:\n- lower bound: ", err_bounds[2*nvariables(equations)+1])
  end
  println("─"^100 * "\n")

  return nothing
end

end # @muladd
