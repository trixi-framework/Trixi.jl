# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using NLsolve
@muladd begin

function f_own(a_unknown, num_stages, num_stage_evals, mon_coeffs)
    c_s2 = 1 # c_s2 = c_ts(S-2)
    c_ts = c_own(c_s2, num_stages) # ts = timestep

    c_eq = zeros(num_stage_evals - 2) # Add equality constraint that c_s2 is equal to 1
    # Both terms should be present
    for i in 1:(num_stage_evals - 4)
        term1 = a_unknown[num_stage_evals - 1]
        term2 = a_unknown[num_stage_evals]
        for j in 1:i
            term1 *= a_unknown[num_stage_evals - 1 - j]
            term2 *= a_unknown[num_stage_evals - j]
        end
        term1 *= c_ts[num_stages - 2 - i] * 1/6
        term2 *= c_ts[num_stages - 1 - i] * 4/6

        c_eq[i] = mon_coeffs[i] - (term1 + term2)
    end

    # Highest coefficient: Only one term present
    i = num_stage_evals - 3
    term2 = a_unknown[num_stage_evals]
    for j in 1:i
        term2 *= a_unknown[num_stage_evals - j]
    end
    term2 *= c_ts[num_stages - 1 - i] * 4/6

    c_eq[i] = mon_coeffs[i] - term2

    c_eq[num_stage_evals - 2] = 1.0 - 4 * a_unknown[num_stage_evals] - a_unknown[num_stage_evals - 1]

    return c_eq
end

function c_own(c_s2, num_stages)
    c_ts = zeros(num_stages)

    # Last timesteps as for SSPRK33
    c_ts[num_stages]     = 0.5
    c_ts[num_stages - 1] = 1

    # Linear increasing timestep for remainder
    for i in 2:num_stages-2
        c_ts[i] = c_s2 * (i-1)/(num_stages - 3)
    end
    return c_ts
end

function compute_PERK3_butcher_tableau(num_stages, num_stage_evals, semi::AbstractSemidiscretization)

    # Initialize array of c
    c_ts = c_own(1.0, num_stages)

    # Initialize the array of our solution
    a_unknown = zeros(num_stage_evals)

    # Special case of e = 3
    if num_stage_evals == 3
        a_unknown = [0, c_ts[2], 0.25]

    else
        # Calculate coefficients of the stability polynomial in monomial form
        cons_order = 3
        dt_max = 1.0
        dt_eps = 1e-9
        filter_threshold = 1e-12

        # Compute spectrum
        J = jacobian_ad_forward(semi)
        eig_vals = eigvals(J)
        num_eig_vals, eig_vals = filter_eigvals(eig_vals, filter_threshold)

        mon_coeffs, dt, _ = bisection(cons_order, num_eig_vals, num_stages, dt_max, dt_eps, eig_vals)
        mon_coeffs = undo_normalization!(cons_order, num_stages, mon_coeffs)

        # Define the objective_function
        function objective_function(x)
            return f_own(x, num_stages, num_stage_evals, mon_coeffs)
        end

        # Call nlsolver to solve repeatedly until the result is not NaN or negative values
        is_sol_valid = false
        while !is_sol_valid
            # Initialize initial guess
            x0 = 0.1 .* rand(num_stage_evals)
            x0[1] = 0.0
            x0[2] = c_ts[2]

            sol = nlsolve(objective_function, x0, method = :trust_region, ftol = 1e-15, iterations = 10^4, xtol = 1e-13)
                          
            a_unknown = sol.zero

            # Check if the values a[i, i-1] >= 0.0 (which stem from the nonlinear solver) and subsequently c[i] - a[i, i-1] >= 0.0
            is_sol_valid = all(x -> !isnan(x) && x >= 0, a_unknown[3:end]) && all(x -> !isnan(x) && x >= 0 , c_ts[3:end] .- a_unknown[3:end])
        end
    end

    println("a_unknown")
    println(a_unknown[3:end]) # To debug

    a_matrix = zeros(num_stages - 2, 2)
    a_matrix[:, 1] = c_ts[3:end]
    a_matrix[:, 1] -= a_unknown[3:end]
    a_matrix[:, 2]  = a_unknown[3:end]
  
    return a_matrix, c_ts
end

function compute_PERK3_butcher_tableau(num_stages::Int, base_path_mon_coeffs::AbstractString, c_s2::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(num_stages)
    for k in 2:num_stages-2
      c[k] = c_s2 * (k - 1)/(num_stages - 3) # Equidistant timestep distribution (similar to PERK2)
    end
  
    # Original third order proposed PERK
    #=
    c[num_stages - 1] = 1.0/3.0
    c[num_stages]     = 1.0
    =#
    # Own third order PERK based on SSPRK33
    c[num_stages - 1] = 1.0
    c[num_stages]     = 0.5

    println("Timestep-split: "); display(c); println("\n")
  
    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stages - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    path_mon_coeffs = base_path_mon_coeffs * "a_" * string(num_stages) * "_" * string(num_stages) * ".txt"
    num_mon_coeffs, A = read_file(path_mon_coeffs, Float64)
    @assert num_mon_coeffs == coeffs_max

    a_matrix[:, 1] -= A
    a_matrix[:, 2]  = A
    
    println("A matrix: "); display(AMatrix); println()

    return a_matrix, c
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
    const num_stages::Int

    a_matrix::Matrix{Float64}
    c::Vector{Float64}

    # Constructor for previously computed A Coeffs
    function PERK3(num_stages_::Int, base_path_mon_coeffs_::AbstractString, c_s2_::Float64=1.0)
        newPERK3 = new(num_stages_)

        newPERK3.a_matrix, newPERK3.c = compute_PERK3_butcher_tableau(num_stages_, base_path_mon_coeffs_, c_s2_)
      
        return newPERK3
    end

    # Constructor that computes Butcher matrix A coefficients from a semidiscretization
    function PERK3(num_stages_::Int, num_stage_evals_ ::Int, semi_::AbstractSemidiscretization)

        newPERK3 = new(num_stages_)

        newPERK3.a_matrix, newPERK3.c = compute_PERK3_butcher_tableau(num_stages_, num_stage_evals_, semi_)

        display(newPERK3.a_matrix)

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