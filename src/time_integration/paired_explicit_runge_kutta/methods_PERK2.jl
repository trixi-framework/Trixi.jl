# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

abstract type PERK end
# Abstract base type for single/standalone P-ERK time integration schemes
abstract type PERKSingle <: PERK end

function compute_a_coeffs(num_stage_evals::Int,
                        se_factors::Vector{Float64}, mon_coeffs::Vector{Float64})
    a_coeffs = mon_coeffs

    for stage in 1:(num_stage_evals - 2)
        a_coeffs[stage] /= se_factors[stage]
        for prev_stage in 1:(stage - 1)
            a_coeffs[stage] /= a_coeffs[prev_stage]
        end
    end

    return reverse(a_coeffs)
end

function compute_PERK2_butcher_tableau(num_stages::Int, semi::AbstractSemidiscretization,
                                     b_s::Float64, c_end::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(num_stages)
    for k in 2:num_stages
        c[k] = c_end * (k - 1) / (num_stages - 1)
    end
    println("Timestep-split: ")
    display(c)
    println("\n")
    se_factors = b_s * reverse(c[2:(end - 1)])

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stages - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    cons_order = 2
    filter_thres = 1e-12
    dtmax = 1.0
    dt_eps = 1e-9

    eig_vals = eigvals(jacobian_ad_forward(semi))

    num_eig_vals, eig_vals = filter_eigvals(eig_vals, filter_thres)

    mon_coeffs, p_worst_case, dt_opt = bisection(cons_order, num_eig_vals, num_stages, dtmax, dt_eps,
                                             eig_vals)
    mon_coeffs = undo_normalization!(cons_order, num_stages, mon_coeffs)

    num_mon_coeffs = length(mon_coeffs)
    @assert num_mon_coeffs == coeffs_max
    A = compute_a_coeffs(num_stages, se_factors, mon_coeffs)

    #=
    # TODO: Not sure if I not rather want to read-in values (especially those from Many Stage C++ Optim)
    path_mon_coeffs = base_path_mon_coeffs * "a_" * string(num_stages) * ".txt"
    num_mon_coeffs, A = read_file(path_mon_coeffs, Float64)
    @assert num_mon_coeffs == coeffs_max
    =#

    a_matrix[:, 1] -= A
    a_matrix[:, 2] = A

    println("A matrix: ")
    display(a_matrix)
    println()

    return a_matrix, c
end

function compute_PERK2_butcher_tableau(num_stages::Int, base_path_mon_coeffs::AbstractString,
                                     b_s::Float64, c_end::Float64)

    # c Vector form Butcher Tableau (defines timestep per stage)
    c = zeros(num_stages)
    for k in 2:num_stages
        c[k] = c_end * (k - 1) / (num_stages - 1)
    end
    println("Timestep-split: ")
    display(c)
    println("\n")
    se_factors = b_s * reverse(c[2:(end - 1)])

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    coeffs_max = num_stages - 2

    a_matrix = zeros(coeffs_max, 2)
    a_matrix[:, 1] = c[3:end]

    path_mon_coeffs = base_path_mon_coeffs * "gamma_" * string(num_stages) * ".txt"
    num_mon_coeffs, mon_coeffs = read_file(path_mon_coeffs, Float64)
    @assert num_mon_coeffs == coeffs_max
    A = compute_a_coeffs(num_stages, se_factors, mon_coeffs)

    #=
    # TODO: Not sure if I not rather want to read-in values (especially those from Many Stage C++ Optim)
    path_mon_coeffs = base_path_mon_coeffs * "a_" * string(num_stages) * ".txt"
    num_mon_coeffs, A = read_file(path_mon_coeffs, Float64)
    @assert num_mon_coeffs == coeffs_max
    =#

    a_matrix[:, 1] -= A
    a_matrix[:, 2] = A

    println("A matrix: ")
    display(a_matrix)
    println()

    return a_matrix, c
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
    const num_stages::Int

    a_matrix::Matrix{Float64}
    c::Vector{Float64}
    b_s::Float64
    b1::Float64
    c_end::Float64

    #Constructor that read the coefficients from the file
    function PERK2(num_stages_::Int, base_path_mon_coeffs_::AbstractString, b_s_::Float64 = 1.0,
                   c_end_::Float64 = 0.5)
        newPERK2 = new(num_stages_)

        newPERK2.a_matrix, newPERK2.c = compute_PERK2_butcher_tableau(num_stages_,
                                                                   base_path_mon_coeffs_, b_s_,
                                                                   c_end_)

        newPERK2.b1 = one(b_s_) - b_s_
        newPERK2.b_s = b_s_
        newPERK2.c_end = c_end_
        return newPERK2
    end

    #Constructor that calculate the coefficients with polynomial optimizer
    function PERK2(num_stages_::Int, semi_::AbstractSemidiscretization, b_s_::Float64 = 1.0,
                   c_end_::Float64 = 0.5)
        newPERK2 = new(num_stages_)

        newPERK2.a_matrix, newPERK2.c = compute_PERK2_butcher_tableau(num_stages_, semi_, b_s_,
                                                                   c_end_)

        newPERK2.b1 = one(b_s_) - b_s_
        newPERK2.b_s = b_s_
        newPERK2.c_end = c_end_
        return newPERK2
    end
end # struct PERK2

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L1
mutable struct PERKIntegratorOptions{Callback}
    callback::Callback # callbacks; used in Trixi
    adaptive::Bool # whether the algorithm is adaptive; ignored
    dtmax::Float64 # ignored
    maxiters::Int # maximal numer of time steps
    tstops::Vector{Float64} # tstops from https://diffeq.sciml.ai/v6.8/basics/common_solver_opts/#Output-Control-1; ignored
end

function PERKIntegratorOptions(callback, tspan; maxiters = typemax(Int), kwargs...)
    PERKIntegratorOptions{typeof(callback)}(callback, false, Inf, maxiters, [last(tspan)])
end

abstract type PERKIntegrator end
abstract type PERKSingleIntegrator <: PERKIntegrator end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PERK2Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                PERKIntegratorOptions} <: PERKSingleIntegrator
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
    opts::PERKIntegratorOptions
    finalstep::Bool # added for convenience
    # PERK2 stages:
    k1::uType
    k_higher::uType
    t_stage::RealT
end

# Forward integrator.stats.naccept to integrator.iter (see GitHub PR#771)
function Base.getproperty(integrator::PERKIntegrator, field::Symbol)
    if field === :stats
        return (naccept = getfield(integrator, :iter),)
    end
    # general fallback
    return getfield(integrator, field)
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PERK2;
               dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0) #previously: similar(u0)
    u_tmp = zero(u0)

    # PERK2 stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
    iter = 0

    integrator = PERK2Integrator(u0, du, u_tmp, t0, dt, zero(dt), iter, ode.p,
                                  (prob = ode,), ode.f, alg,
                                  PERKIntegratorOptions(callback, ode.tspan; kwargs...),
                                  false,
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

function solve!(integrator::PERK2Integrator)
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
        if integrator.t + integrator.dt > t_end ||
           isapprox(integrator.t + integrator.dt, t_end)
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
            for stage in 3:(alg.num_stages)
                integrator.t_stage = integrator.t + alg.c[stage] * integrator.dt

                # Construct current state
                @threaded for i in eachindex(integrator.du)
                    integrator.u_tmp[i] = integrator.u[i] +
                                          alg.a_matrix[stage - 2, 1] * integrator.k1[i] +
                                          alg.a_matrix[stage - 2, 2] * integrator.k_higher[i]
                end

                integrator.f(integrator.du, integrator.u_tmp, prob.p, integrator.t_stage)

                @threaded for i in eachindex(integrator.du)
                    integrator.k_higher[i] = integrator.du[i] * integrator.dt
                end
            end

            @threaded for i in eachindex(integrator.u)
                #integrator.u[i] += integrator.k_higher[i]
                integrator.u[i] += alg.b1 * integrator.k1[i] +
                                   alg.b_s * integrator.k_higher[i]
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
get_du(integrator::PERKIntegrator) = integrator.du
get_tmp_cache(integrator::PERKIntegrator) = (integrator.u_tmp,)

# some algorithms from DiffEq like FSAL-ones need to be informed when a callback has modified u
u_modified!(integrator::PERKIntegrator, ::Bool) = false

# used by adaptive timestepping algorithms in DiffEq
function set_proposed_dt!(integrator::PERKIntegrator, dt)
    integrator.dt = dt
end

function get_proposed_dt(integrator::PERKIntegrator)
    return integrator.dt
end

# stop the time integration
function terminate!(integrator::PERKIntegrator)
    integrator.finalstep = true
    empty!(integrator.opts.tstops)
end

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PERK2Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end

end