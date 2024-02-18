# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

abstract type PERK end
# Abstract base type for single/standalone P-ERK time integration schemes
abstract type PERKSingle <: PERK end

function ComputeACoeffs(NumStageEvals::Int,
                        SE_Factors::Vector{Float64}, MonCoeffs::Vector{Float64})
  ACoeffs = MonCoeffs

  for stage in 1:NumStageEvals - 2
    ACoeffs[stage] /= SE_Factors[stage]
    for prev_stage in 1:stage-1
      ACoeffs[stage] /= ACoeffs[prev_stage]
    end
  end

  return reverse(ACoeffs)
end

function ComputePERK2_ButcherTableau(NumStages::Int, semi::AbstractSemidiscretization, bS::Float64, cEnd::Float64)

  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(NumStages)
  for k in 2:NumStages
    c[k] = cEnd * (k - 1)/(NumStages - 1)
  end
  println("Timestep-split: "); display(c); println("\n")
  SE_Factors = bS * reverse(c[2:end-1])

  # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
  CoeffsMax = NumStages - 2

  AMatrix = zeros(CoeffsMax, 2)
  AMatrix[:, 1] = c[3:end]

  
  ConsOrder = 2
  filter_thres = 1e-12
  dtMax = 1.0
  dtEps = 1e-9

  J = jacobian_ad_forward(semi)
  EigVals = eigvals(J)

  NumEigVals, EigVals = filter_Eigvals(EigVals, filter_thres)

  MonCoeffs, PWorstCase, dtOpt = Bisection(ConsOrder, NumEigVals, NumStages, dtMax, dtEps, EigVals)
  MonCoeffs = undo_normalization(ConsOrder, NumStages, MonCoeffs)

  NumMonCoeffs = length(MonCoeffs)
  @assert NumMonCoeffs == CoeffsMax
  A = ComputeACoeffs(NumStages, SE_Factors, MonCoeffs)
  
  
  #=
  # TODO: Not sure if I not rather want to read-in values (especially those from Many Stage C++ Optim)
  PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStages) * ".txt"
  NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
  @assert NumMonCoeffs == CoeffsMax
  =#

  AMatrix[:, 1] -= A
  AMatrix[:, 2]  = A
    
  println("A matrix: "); display(AMatrix); println()

  return AMatrix, c
end

function ComputePERK2_ButcherTableau(NumStages::Int, BasePathMonCoeffs::AbstractString, bS::Float64, cEnd::Float64)

  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(NumStages)
  for k in 2:NumStages
    c[k] = cEnd * (k - 1)/(NumStages - 1)
  end
  println("Timestep-split: "); display(c); println("\n")
  SE_Factors = bS * reverse(c[2:end-1])

  # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
  CoeffsMax = NumStages - 2

  AMatrix = zeros(CoeffsMax, 2)
  AMatrix[:, 1] = c[3:end]

  
  PathMonCoeffs = BasePathMonCoeffs * "gamma_" * string(NumStages) * ".txt"
  NumMonCoeffs, MonCoeffs = read_file(PathMonCoeffs, Float64)
  @assert NumMonCoeffs == CoeffsMax
  A = ComputeACoeffs(NumStages, SE_Factors, MonCoeffs)
  
  
  #=
  # TODO: Not sure if I not rather want to read-in values (especially those from Many Stage C++ Optim)
  PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStages) * ".txt"
  NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
  @assert NumMonCoeffs == CoeffsMax
  =#

  AMatrix[:, 1] -= A
  AMatrix[:, 2]  = A
    
  println("A matrix: "); display(AMatrix); println()

  return AMatrix, c
end

"""
    PERK2()

The following structures and methods provide a minimal implementation of
the paired explicit Runge-Kutta method (https://doi.org/10.1016/j.jcp.2019.05.014)
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PERK2 <: PERKSingle
  const NumStages::Int

  AMatrix::Matrix{Float64}
  c::Vector{Float64}
  bS::Float64
  b1::Float64
  cEnd::Float64

  #Constructor that read the coefficients from the file
  function PERK2(NumStages_::Int, BasePathMonCoeffs_::AbstractString, bS_::Float64=1.0, cEnd_::Float64=0.5)

    newPERK2 = new(NumStages_)

    newPERK2.AMatrix, newPERK2.c = 
      ComputePERK2_ButcherTableau(NumStages_, BasePathMonCoeffs_, bS_, cEnd_)

    newPERK2.b1 = one(bS_) - bS_
    newPERK2.bS = bS_
    newPERK2.cEnd = cEnd_
    return newPERK2
  end

  #Constructor that calculate the coefficients with polynomial optimizer
  function PERK2(NumStages_::Int, semi_::AbstractSemidiscretization, bS_::Float64=1.0, cEnd_::Float64=0.5)

    newPERK2 = new(NumStages_)

    newPERK2.AMatrix, newPERK2.c = 
      ComputePERK2_ButcherTableau(NumStages_, semi_, bS_, cEnd_)

    newPERK2.b1 = one(bS_) - bS_
    newPERK2.bS = bS_
    newPERK2.cEnd = cEnd_
    return newPERK2
  end
end # struct PERK2


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PERK_IntegratorOptions{Callback}
  callback::Callback # callbacks; used in Trixi
  adaptive::Bool # whether the algorithm is adaptive; ignored
  dtmax::Float64 # ignored
  maxiters::Int # maximal numer of time steps
  tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function PERK_IntegratorOptions(callback, tspan; maxiters=typemax(Int), kwargs...)
  PERK_IntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

abstract type PERK_Integrator end
abstract type PERKSingle_Integrator <: PERK_Integrator end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK2_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions} <: PERKSingle_Integrator
  u::uType
  du::uType
  u_tmp::uType
  t::RealT
  dt::RealT # current time step
  dtcache::RealT # ignored
  iter::Int # current number of time steps (iteration)
  p::Params # will be the semidiscretization from Trixi
  sol::Sol # faked
  f::F
  alg::Alg # This is our own class written above; Abbreviation for ALGorithm
  opts::PERK_IntegratorOptions
  finalstep::Bool # added for convenience
  # PERK2 stages:
  k1::uType
  k_higher::uType
  t_stage::RealT
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERK_Integrator, field::Symbol)
  if field === :stats
    return (naccept = getfield(integrator, :iter),)
  end
  # general fallback
  return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK2;
               dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = zero(u0) #previously: similar(u0)
  u_tmp = zero(u0)

  # PERK2 stages
  k1       = zero(u0)
  k_higher = zero(u0)

  t0 = first(ode.tspan)
  iter = 0

  integrator = PERK2_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                              (prob=ode,), ode.f, alg,
                              PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                              k1, k_higher, t0)
            
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

function solve!(integrator::PERK2_Integrator)
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

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
      # k1:
      integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
      @threaded for i in eachindex(integrator.du)
        integrator.k1[i] = integrator.du[i] * integrator.dt
      end

      # k2
      integrator.t_stage = integrator.t + alg.c[2] * integrator.dt
    
      # Construct current state
      @threaded for i in eachindex(integrator.du)
        integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
      end

      integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

      @threaded for i in eachindex(integrator.du)
        integrator.k_higher[i] = integrator.du[i] * integrator.dt
      end
      
      # Higher stages
      for stage = 3:alg.NumStages
        integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

        # Construct current state
        @threaded for i in eachindex(integrator.du)
          integrator.u_tmp[i] = integrator.u[i] + alg.AMatrix[stage - 2, 1] * integrator.k1[i] + 
                                                  alg.AMatrix[stage - 2, 2] * integrator.k_higher[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

        @threaded for i in eachindex(integrator.du)
          integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end
      end

      @threaded for i in eachindex(integrator.u)
        #integrator.u[i] += integrator.k_higher[i]
        integrator.u[i] += alg.b1 * integrator.k1[i] + alg.bS * integrator.k_higher[i]
      end
    end # PERK2 step

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
  end # "main loop" timer
  
  return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                (prob.u0, integrator.u),
                                integrator.sol.prob)
end

# get a cache where the RHS can be stored
get_du(integrator::PERK_Integrator) = integrator.du
get_tmp_cache(integrator::PERK_Integrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERK_Integrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERK_Integrator, dt)
  integrator.dt = dt
end

function get_proposed_dt(integrator::PERK_Integrator)
  return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERK_Integrator)
  integrator.finalstep = true
  empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK2_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
end

end # @muladd