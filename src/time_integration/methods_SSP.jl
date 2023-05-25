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
struct SimpleSSPRK33{StageCallback} <: SimpleAlgorithmSSP
  a::SVector{3, Float64}
  b::SVector{3, Float64}
  c::SVector{3, Float64}
  stage_callback::StageCallback

  function SimpleSSPRK33(; stage_callback=nothing)
    a = SVector(0.0, 3/4, 1/3)
    b = SVector(1.0, 1/4, 2/3)
    c = SVector(0.0, 1.0, 1/2)

    # Butcher tableau
    #   c |       a
    #   0 |
    #   1 |   1
    # 1/2 | 1/4  1/4
    # --------------------
    #   b | 1/6  1/6  2/3

    new{typeof(stage_callback)}(a, b, c, stage_callback)
  end
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct SimpleIntegratorSSPOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal number of time steps
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
mutable struct SimpleIntegratorSSP{RealT<:Real, uType, Params, Sol, F, Alg, SimpleIntegratorSSPOptions}
  u::uType
  du::uType
  r0::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F
  alg::Alg
  opts::SimpleIntegratorSSPOptions
  finalstep::Bool # added for convenience
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::SimpleIntegratorSSP, field::Symbol)
  if field === :stats
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
  r0 = similar(u)
  t = first(ode.tspan)
  iter = 0
  integrator = SimpleIntegratorSSP(u, du, r0, t, dt, zero(dt), iter, ode.p,
                  (prob=ode,), ode.f, alg,
                  SimpleIntegratorSSPOptions(callback, ode.tspan; kwargs...), false)

  # resize container
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

  if alg.stage_callback !== nothing
    init_callback(alg.stage_callback, integrator.p)
  end

  solve!(integrator)
end

function solve!(integrator::SimpleIntegratorSSP)
  @unpack indicator = integrator.p.solver.volume_integral
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  # WARNING: Only works if the last callback got a variable `output_directory`.
  if callbacks.discrete_callbacks[end].condition isa SaveSolutionCallback
    output_directory = callbacks.discrete_callbacks[end].condition.output_directory
  else
    output_directory = "out"
  end

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

    # Reset alphas for MCL
    if indicator isa IndicatorMCL && indicator.Plotting
      @unpack alpha, alpha_pressure, alpha_entropy = indicator.cache.ContainerShockCapturingIndicator
      @threaded for element in eachelement(integrator.p.solver, integrator.p.cache)
        for j in eachnode(integrator.p.solver), i in eachnode(integrator.p.solver)
          alpha[:, i, j, element] .= one(eltype(alpha))
          if indicator.PressurePositivityLimiterKuzmin
            alpha_pressure[i, j, element] = one(eltype(alpha_pressure))
          end
          if indicator.SemiDiscEntropyLimiter
            alpha_entropy[i, j, element] = one(eltype(alpha_entropy))
          end
        end
      end
    end

    @. integrator.r0 = integrator.u
    for stage in eachindex(alg.c)
      @trixi_timeit timer() "Runge-Kutta stage" begin
        t_stage = integrator.t + integrator.dt * alg.c[stage]
        # compute du
        integrator.f(integrator.du, integrator.u, integrator.p, t_stage)

        # perform forward Euler step
        @. integrator.u = integrator.u + integrator.dt * integrator.du
      end
      @trixi_timeit timer() "Antidiffusive stage" antidiffusive_stage!(integrator.u, t_stage, integrator.dt, integrator.p, indicator)

      @trixi_timeit timer() "update_alpha_max_avg!" update_alpha_max_avg!(indicator, integrator.iter+1, length(alg.c), integrator.p, integrator.p.mesh)

      if alg.stage_callback !== nothing
        laststage = (stage == length(alg.c))
        alg.stage_callback(integrator.u, integrator.p, integrator.t, integrator.iter+1, laststage)
      end

      # perform convex combination
      @. integrator.u = alg.a[stage] * integrator.r0 + alg.b[stage] * integrator.u
    end

    @trixi_timeit timer() "save_alpha" save_alpha(indicator, integrator.t, integrator.iter+1, integrator.p, integrator.p.mesh, output_directory)

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

  if alg.stage_callback !== nothing
    finalize_callback(alg.stage_callback, integrator.p)
  end

  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u), prob)
end

# get a cache where the RHS can be stored
get_du(integrator::SimpleIntegratorSSP) = integrator.du
get_tmp_cache(integrator::SimpleIntegratorSSP) = (integrator.r0,)

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
end

# used for AMR
function Base.resize!(integrator::SimpleIntegratorSSP, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.r0, new_size)

  # Resize container
  resize!(integrator.p, new_size)
end

function Base.resize!(semi::AbstractSemidiscretization, new_size)
  resize!(semi, semi.solver.volume_integral, new_size)
end

Base.resize!(semi, volume_integral::AbstractVolumeIntegral, new_size) = nothing

function Base.resize!(semi, volume_integral::VolumeIntegralShockCapturingSubcell, new_size)
  # Resize ContainerAntidiffusiveFlux2D
  resize!(semi.cache.ContainerAntidiffusiveFlux2D, new_size)

  # Resize ContainerShockCapturingIndicator
  resize!(semi.solver.volume_integral.indicator.cache.ContainerShockCapturingIndicator, new_size)
  # Calc subcell normal directions before StepsizeCallback
  @unpack indicator = semi.solver.volume_integral
  if indicator isa IndicatorMCL || (indicator isa IndicatorIDP && indicator.BarStates)
    resize!(semi.solver.volume_integral.indicator.cache.ContainerBarStates, new_size)
    calc_normal_directions!(indicator.cache.ContainerBarStates, mesh_equations_solver_cache(semi)...)
  end
end

function calc_normal_directions!(ContainerBarStates, mesh::TreeMesh, equations, dg, cache)

  return nothing
end

function calc_normal_directions!(ContainerBarStates, mesh::StructuredMesh, equations, dg, cache)
  @unpack weights, derivative_matrix = dg.basis
  @unpack contravariant_vectors = cache.elements

  @unpack normal_direction_xi, normal_direction_eta = ContainerBarStates
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg)
      normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)
      for i in 2:nnodes(dg)
        for m in 1:nnodes(dg)
          normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
        end
        for v in 1:nvariables(equations)-2
          normal_direction_xi[v, i-1, j, element] = normal_direction[v]
        end
      end
    end
    for i in eachnode(dg)
      normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)
      for j in 2:nnodes(dg)
        for m in 1:nnodes(dg)
          normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
        end
        for v in 1:nvariables(equations)-2
          normal_direction_eta[v, i, j-1, element] = normal_direction[v]
        end
      end
    end
  end

  return nothing
end


end # @muladd
