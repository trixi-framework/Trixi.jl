# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function PERK2_compute_c_coeffs(num_stages, cS)
    c = zeros(num_stages)
    for k in 2:num_stages
        c[k] = cS * (k - 1) / (num_stages - 1)
    end

    return c
end

# Compute the coefficients of the A matrix in the Butcher tableau using
# stage scaling factors and monomial coefficients
function compute_a_coeffs(num_stage_evals, stage_scaling_factors, monomial_coeffs)
    a_coeffs = copy(monomial_coeffs)

    for stage in 1:(num_stage_evals - 2)
        a_coeffs[stage] /= stage_scaling_factors[stage]
        for prev_stage in 1:(stage - 1)
            a_coeffs[stage] /= a_coeffs[prev_stage]
        end
    end

    return reverse(a_coeffs)
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 2
# using a list of eigenvalues
function compute_PairedExplicitRK2_butcher_tableau(num_stages, eig_vals, tspan,
                                                   bS, cS; verbose = false)
    c = PERK2_compute_c_coeffs(num_stages, cS)
    stage_scaling_factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since first entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    num_coeffs_max = num_stages - 2

    a_matrix = zeros(2, num_coeffs_max)
    a_matrix[1, :] = c[3:end]

    dtmax = tspan[2] - tspan[1]
    dteps = 1e-9 # Hyperparameter of the optimization, might be too large for systems requiring very small timesteps

    num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

    consistency_order = 2
    monomial_coeffs, dt_opt = bisect_stability_polynomial(consistency_order,
                                                          num_eig_vals, num_stages,
                                                          dtmax, dteps,
                                                          eig_vals; verbose)

    if num_coeffs_max > 0
        num_monomial_coeffs = length(monomial_coeffs)
        @assert num_monomial_coeffs == num_coeffs_max
        A = compute_a_coeffs(num_stages, stage_scaling_factors, monomial_coeffs)
        a_matrix[1, :] -= A
        a_matrix[2, :] = A
    end

    return a_matrix, c, dt_opt
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 2
# using provided monomial coefficients file
function compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                   base_path_monomial_coeffs::AbstractString,
                                                   bS, cS)
    c = PERK2_compute_c_coeffs(num_stages, cS)
    stage_scaling_factors = bS * reverse(c[2:(end - 1)])

    # - 2 Since first entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    num_coeffs_max = num_stages - 2

    a_matrix = zeros(2, num_coeffs_max)
    a_matrix[1, :] = c[3:end]

    if num_coeffs_max > 0
        path_monomial_coeffs = joinpath(base_path_monomial_coeffs,
                                        "gamma_" * string(num_stages) * ".txt")

        @assert isfile(path_monomial_coeffs) "Couldn't find file"
        monomial_coeffs = readdlm(path_monomial_coeffs, Float64)
        num_monomial_coeffs = size(monomial_coeffs, 1)

        @assert num_monomial_coeffs == num_coeffs_max
        A = compute_a_coeffs(num_stages, stage_scaling_factors, monomial_coeffs)

        a_matrix[1, :] -= A
        a_matrix[2, :] = A
    end

    return a_matrix, c
end

@doc raw"""
    PairedExplicitRK2(num_stages, base_path_monomial_coeffs::AbstractString; dt_opt = nothing,
                      bS = 1.0, cS = 0.5)
    PairedExplicitRK2(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, bS = 1.0, cS = 0.5)
    PairedExplicitRK2(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, bS = 1.0, cS = 0.5)

The following structures and methods provide a minimal implementation of
the second-order paired explicit Runge-Kutta (PERK) method
optimized for a certain simulation setup (PDE, IC & BCs, Riemann Solver, DG Solver).
The original paper is

- Brian Vermeire (2019).
  Paired explicit Runge-Kutta schemes for stiff systems of equations
  [DOI: 10.1016/j.jcp.2019.05.014](https://doi.org/10.1016/j.jcp.2019.05.014)

# Arguments
- `num_stages` (`Int`): Number of stages in the PERK method.
- `base_path_monomial_coeffs` (`AbstractString`): Path to a file containing 
    monomial coefficients of the stability polynomial of PERK method.
    The coefficients should be stored in a text file at `joinpath(base_path_monomial_coeffs, "gamma_$(num_stages).txt")` and separated by line breaks.
- `dt_opt` (`Float64`, optional): Optimal time step size for the simulation setup. Can be `nothing` if it is unknown. 
    In this case the optimal CFL number cannot be computed and the [`StepsizeCallback`](@ref) cannot be used.
- `tspan`: Time span of the simulation.
- `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
- `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
    equation has been semidiscretized.
- `verbose` (`Bool`, optional): Verbosity flag, default is false.
- `bS` (`Float64`, optional): Value of $b_S$ in the Butcher tableau, where
    $S$ is the number of stages. Default is `1.0`.
- `cS` (`Float64`, optional): Value of $c_S$ in the Butcher tableau, where
    $S$ is the number of stages. Default is `0.5`.

!!! note
    To use this integrator, the user must import the
    [Convex.jl](https://github.com/jump-dev/Convex.jl) and 
    [ECOS.jl](https://github.com/jump-dev/ECOS.jl) packages
    unless the coefficients are provided in a `gamma_<num_stages>.txt` file.
"""
struct PairedExplicitRK2 <: AbstractPairedExplicitRKSingle
    num_stages::Int

    a_matrix::Matrix{Float64}
    c::Vector{Float64}
    b1::Float64
    bS::Float64
    cS::Float64

    dt_opt::Union{Float64, Nothing}
end

# Constructor that reads the coefficients from a file
function PairedExplicitRK2(num_stages, base_path_monomial_coeffs::AbstractString;
                           dt_opt = nothing,
                           bS = 1.0, cS = 0.5)
    @assert num_stages>=2 "PERK2 requires at least two stages"
    # If the user has the monomial coefficients, they also must have the optimal time step
    a_matrix, c = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                            base_path_monomial_coeffs,
                                                            bS, cS)

    return PairedExplicitRK2(num_stages, a_matrix, c, 1 - bS, bS, cS, dt_opt)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# semidiscretization
function PairedExplicitRK2(num_stages, tspan, semi::AbstractSemidiscretization;
                           verbose = false,
                           bS = 1.0, cS = 0.5)
    @assert num_stages>=2 "PERK2 requires at least two stages"
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return PairedExplicitRK2(num_stages, tspan, eig_vals; verbose, bS, cS)
end

# Constructor that calculates the coefficients with polynomial optimizer from a
# list of eigenvalues
function PairedExplicitRK2(num_stages, tspan, eig_vals::Vector{ComplexF64};
                           verbose = false,
                           bS = 1.0, cS = 0.5)
    @assert num_stages>=2 "PERK2 requires at least two stages"
    a_matrix, c, dt_opt = compute_PairedExplicitRK2_butcher_tableau(num_stages,
                                                                    eig_vals, tspan,
                                                                    bS, cS;
                                                                    verbose)

    return PairedExplicitRK2(num_stages, a_matrix, c, 1 - bS, bS, cS, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.
mutable struct PairedExplicitRK2Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                           PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT # DIRection of time integration, i.e., if one marches forward or backward in time
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F # `rhs!` of the semidiscretization
    alg::Alg # PairedExplicitRK2
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # Additional PERK register
    k1::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK2;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK2Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             PairedExplicitRKOptions(callback,
                                                                     ode.tspan;
                                                                     kwargs...),
                                             false, true, false,
                                             k1)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            throw(ArgumentError("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods."))
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    end

    return integrator
end

function step!(integrator::PairedExplicitRK2Integrator)
    @unpack prob = integrator.sol
    @unpack alg = integrator
    t_end = last(prob.tspan)
    callbacks = integrator.opts.callback

    @assert !integrator.finalstep
    if isnan(integrator.dt)
        error("time step size `dt` is NaN")
    end

    modify_dt_for_tstops!(integrator)

    # if the next iteration would push the simulation beyond the end time, set dt accordingly
    if integrator.t + integrator.dt > t_end ||
       isapprox(integrator.t + integrator.dt, t_end)
        integrator.dt = t_end - integrator.t
        terminate!(integrator)
    end

    @trixi_timeit timer() "Paired Explicit Runge-Kutta ODE integration step" begin
        # First and second stage are identical across all single/standalone PERK methods
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages
        for stage in 3:(alg.num_stages)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        @threaded for i in eachindex(integrator.u)
            integrator.u[i] += integrator.dt *
                               (alg.b1 * integrator.k1[i] +
                                alg.bS * integrator.du[i])
        end
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            foreach(callbacks.discrete_callbacks) do cb
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
                return nothing
            end
        end
    end

    # respect maximum number of iterations
    if integrator.iter >= integrator.opts.maxiters && !integrator.finalstep
        @warn "Interrupted. Larger maxiters is needed."
        terminate!(integrator)
    end
end
end # @muladd
