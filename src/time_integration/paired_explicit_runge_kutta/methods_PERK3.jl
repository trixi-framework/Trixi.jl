# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using NLsolve
@muladd begin

function F_own(aUnknown, NumStages, NumStageEvals, MonCoeffs)
    cS2 = 1 # cS2 = c_ts(S-2)
    c_ts = c_Own(cS2, NumStages) # ts = timestep

    c_eq = zeros(NumStageEvals - 2) # Add equality constraint that cS2 is equal to 1
    # Both terms should be present
    for i in 1:(NumStageEvals - 4)
        term1 = aUnknown[NumStageEvals - 1]
        term2 = aUnknown[NumStageEvals]
        for j in 1:i
            term1 *= aUnknown[NumStageEvals - 1 - j]
            term2 *= aUnknown[NumStageEvals - j]
        end
        term1 *= c_ts[NumStages - 2 - i] * 1/6
        term2 *= c_ts[NumStages - 1 - i] * 4/6

        c_eq[i] = MonCoeffs[i] - (term1 + term2)
    end

    # Highest coefficient: Only one term present
    i = NumStageEvals - 3
    term2 = aUnknown[NumStageEvals]
    for j in 1:i
        term2 *= aUnknown[NumStageEvals - j]
    end
    term2 *= c_ts[NumStages - 1 - i] * 4/6

    c_eq[i] = MonCoeffs[i] - term2

    c_eq[NumStageEvals - 2] = 1.0 - 4 * aUnknown[NumStageEvals] - aUnknown[NumStageEvals - 1]

    return c_eq
end

function c_Own(cS2, NumStages)
    c_ts = zeros(NumStages);

    # Last timesteps as for SSPRK33
    c_ts[NumStages]     = 0.5;
    c_ts[NumStages - 1] = 1;

    # Linear increasing timestep for remainder
    for i = 2:NumStages-2
        c_ts[i] = cS2 * (i-1)/(NumStages - 3);
    end
    return c_ts
end

function ComputePERK3_ButcherTableau(NumStages, NumStageEvals, semi::AbstractSemidiscretization)

    #Initialize array of c
    c_ts = c_Own(1.0, NumStages)

    #Initialize the array of our solution
    aUnknown = zeros(NumStageEvals)

    #Case of e = 3
    if NumStageEvals == 3
        aUnknown = [0, c_ts[2], 0.25]

    else
        #Calculate MonCoeffs from polynomial Optimizer
        ConsOrder = 3
        dtMax = 1.0
        dtEps = 1e-9
        filter_threshold = 1e-12

        #Get EigVals
        J = jacobian_ad_forward(semi)
        EigVals = eigvals(J)
        NumEigVals, EigVals = filter_Eigvals(EigVals, filter_threshold)

        MonCoeffs, dt, _ = Bisection(ConsOrder, NumEigVals, NumStages, dtMax, dtEps, EigVals)
        MonCoeffs = undo_normalization(ConsOrder, NumStages, MonCoeffs)

        #Define the objective_function
        function objective_function(x)
            return F_own(x, NumStages, NumStageEvals, MonCoeffs)
        end

        #call nlsolver to solve until the answer is not NaN or negative values
        is_sol_valid = false
        while is_sol_valid  == false
            #Initialize initial guess
            x0 = 0.1 .* rand(NumStageEvals)
            x0[1] = 0.0
            x0[2] = c_ts[2]

            sol = nlsolve(objective_function, method = :trust_region, x0, 
                          ftol = 1e-15, iterations = 10^4, xtol = 1e-13)
                          
            aUnknown = sol.zero

            #First check the values of a[i, i-1] which are acquired by nonlinear solver and c[i] - a[i, i-1].
            is_sol_valid  = all(x -> !isnan(x) && x >= 0, aUnknown[3:end]) && all(x -> !isnan(x) && x >= 0 , c_ts[3:end] .- aUnknown[3:end])
        end
    end

    println("aUnknown")
    println(aUnknown[3:end]) #To debug

    AMatrix = zeros(NumStages - 2, 2)
    AMatrix[:, 1] = c_ts[3:end]
    AMatrix[:, 1] -= aUnknown[3:end]
    AMatrix[:, 2]  = aUnknown[3:end]
  
    return AMatrix, c_ts
end

function ComputePERK3_ButcherTableau(NumStages::Int, BasePathMonCoeffs::AbstractString, cS2::Float64)

  # c Vector form Butcher Tableau (defines timestep per stage)
  c = zeros(NumStages)
  for k in 2:NumStages-2
    c[k] = cS2 * (k - 1)/(NumStages - 3) # Equidistant timestep distribution (similar to PERK2)
  end
  
  # Original third order proposed PERK
  #=
  c[NumStages - 1] = 1.0/3.0
  c[NumStages]     = 1.0
  =#
  # Own third order PERK based on SSPRK33
  c[NumStages - 1] = 1.0
  c[NumStages]     = 0.5

  println("Timestep-split: "); display(c); println("\n")
  
  # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
  CoeffsMax = NumStages - 2

  AMatrix = zeros(CoeffsMax, 2)
  AMatrix[:, 1] = c[3:end]

  PathMonCoeffs = BasePathMonCoeffs * "a_" * string(NumStages) * "_" * string(NumStages) * ".txt"
  NumMonCoeffs, A = read_file(PathMonCoeffs, Float64)
  @assert NumMonCoeffs == CoeffsMax

  AMatrix[:, 1] -= A
  AMatrix[:, 2]  = A
    
  println("A matrix: "); display(AMatrix); println()

  return AMatrix, c
end

"""
    PERK3()

The following structures and methods provide a minimal implementation of
the third order paired explicit Runge-Kutta method (https://www.sciencedirect.com/science/article/pii/S0021999122005320)
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).

This is using the same interface as OrdinaryDiffEq.jl, copied from file "methods_2N.jl" for the
CarpenterKennedy2N{54, 43} methods.
"""

mutable struct PERK3 <: PERKSingle
  const NumStages::Int

  AMatrix::Matrix{Float64}
  c::Vector{Float64}

  # Constructor for previously computed A Coeffs
  function PERK3(NumStages_::Int, BasePathMonCoeffs_::AbstractString, cS2_::Float64=1.0)

    newPERK3 = new(NumStages_)

    newPERK3.AMatrix, newPERK3.c = 
      ComputePERK3_ButcherTableau(NumStages_, BasePathMonCoeffs_, cS2_)
    return newPERK3
  end

  #Constructor that compute A coefficients from semidiscretization
  function PERK3(NumStages_::Int, NumStageEvals_ ::Int, semi_::AbstractSemidiscretization)

    newPERK3 = new(NumStages_)

    newPERK3.AMatrix, newPERK3.c = 
      ComputePERK3_ButcherTableau(NumStages_, NumStageEvals_, semi_)

    return newPERK3
  end

end # struct PERK3


# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK3_Integrator{RealT<:Real, uType, Params, Sol, F, Alg, PERK_IntegratorOptions}
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
  # PERK stages:
  k1::uType
  k_higher::uType
  k_S1::uType # Required for third order
  t_stage::RealT
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK3;
               dt, callback=nothing, kwargs...)

  u0    = copy(ode.u0)
  du    = zero(u0) #previously: similar(u0) 
  u_tmp = zero(u0)

  # PERK stages
  k1       = zero(u0)
  k_higher = zero(u0)
  k_S1     = zero(u0)

  t0 = first(ode.tspan)
  iter = 0

  integrator = PERK3_Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                (prob=ode,), ode.f, alg,
                                PERK_IntegratorOptions(callback, ode.tspan; kwargs...), false,
                                k1, k_higher, k_S1, t0)
            
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

function solve!(integrator::PERK3_Integrator)
  @unpack prob = integrator.sol
  @unpack alg = integrator
  t_end = last(prob.tspan)
  callbacks = integrator.opts.callback

  integrator.finalstep = false

  #@trixi_timeit timer() "main loop" while !integrator.finalstep
  while !integrator.finalstep
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

      if alg.NumStages == 3
        @threaded for i in eachindex(integrator.du)
          integrator.k_S1[i] = integrator.k_higher[i]
        end
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

        # IDEA: Stop for loop at NumStages -1 to avoid if (maybe more performant?)
        if stage == alg.NumStages - 1
          @threaded for i in eachindex(integrator.du)
            integrator.k_S1[i] = integrator.k_higher[i]
          end
        end
      end

      @threaded for i in eachindex(integrator.u)
        # Original proposed PERK3
        #integrator.u[i] += 0.75 * integrator.k_S1[i] + 0.25 * integrator.k_higher[i]
        # Own PERK based on SSPRK33
        integrator.u[i] += (integrator.k1[i] + integrator.k_S1[i] + 4.0 * integrator.k_higher[i])/6.0
      end
    end # PERK step timer

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

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK3_Integrator, new_size)
  resize!(integrator.u, new_size)
  resize!(integrator.du, new_size)
  resize!(integrator.u_tmp, new_size)

  resize!(integrator.k1, new_size)
  resize!(integrator.k_higher, new_size)
  resize!(integrator.k_S1, new_size)
end

end # @muladd