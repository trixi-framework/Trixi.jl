# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
using DelimitedFiles: readdlm

@muladd begin
#! format: noindent

# Initialize Butcher array abscissae c for PairedExplicitRK3 based on SSPRK33 base method
function compute_c_coeffs(num_stages, cS2)
    c = zeros(num_stages)

    # Last timesteps as for SSPRK33, see motivation in
    # https://doi.org/10.48550/arXiv.2403.05144
    c[num_stages - 1] = 1.0f0
    c[num_stages] = 0.5f0

    # Linear increasing timestep for remainder
    for i in 2:(num_stages - 2)
        c[i] = cS2 * (i - 1) / (num_stages - 3)
    end

    return c
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 3
# using a list of eigenvalues
function compute_PairedExplicitRK3_butcher_tableau(num_stages, tspan,
                                                   eig_vals::Vector{ComplexF64};
                                                   verbose = false, cS2)
    # Initialize array of c
    c = compute_c_coeffs(num_stages, cS2)

    # Initialize the array of our solution
    a_unknown = zeros(num_stages - 2)

    # Special case of e = 3
    if num_stages == 3
        a_unknown = [0.25]
    else
        # Calculate coefficients of the stability polynomial in monomial form
        consistency_order = 3
        dtmax = tspan[2] - tspan[1]
        dteps = 1.0f-9

        num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

        monomial_coeffs, _ = bisect_stability_polynomial(consistency_order,
                                                         num_eig_vals, num_stages,
                                                         dtmax, dteps,
                                                         eig_vals; verbose)
        monomial_coeffs = undo_normalization!(monomial_coeffs, consistency_order,
                                              num_stages)

        # Solve the nonlinear system of equations from monomial coefficient and
        # Butcher array abscissae c to find Butcher matrix A
        # This function is extended in TrixiNLsolveExt.jl
        a_unknown = solve_a_butcher_coeffs_unknown!(a_unknown, num_stages,
                                                    monomial_coeffs, cS2, c;
                                                    verbose)
    end
    # Fill A-matrix in P-ERK style
    a_matrix = zeros(num_stages - 2, 2)
    a_matrix[:, 1] = c[3:end]
    a_matrix[:, 1] -= a_unknown
    a_matrix[:, 2] = a_unknown

    return a_matrix, c
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 3
# using provided values of coefficients a in A-matrix of Butcher tableau
function compute_PairedExplicitRK3_butcher_tableau(path_a_coeffs::AbstractString;
                                                   cS2)
    @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
    a_coeffs = readdlm(path_a_coeffs, Float64)
    num_a_coeffs = size(a_coeffs, 1)

    # + 2 Since the first entry of A is always zero (explicit method) and the second is given by c_2 (consistency)
    num_stages = num_a_coeffs + 2

    # Initialize array of c
    c = compute_c_coeffs(num_stages, cS2)

    a_matrix = zeros(num_a_coeffs, 2)
    a_matrix[:, 1] = c[3:end]

    # Fill A-matrix in P-ERK style
    a_matrix[:, 1] -= a_coeffs
    a_matrix[:, 2] = a_coeffs

    return num_stages, a_matrix, c
end

@doc raw"""
    PairedExplicitRK3(path_a_coeffs::AbstractString;
                      cS2 = 1.0f0)
    PairedExplicitRK3(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, cS2 = 1.0f0)
    PairedExplicitRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, cS2 = 1.0f0)

    Parameters:
    - `path_a_coeffs` (`AbstractString`): Path to a file containing some coefficients in the A-matrix in 
      the Butcher tableau of the Runge Kutta method.
    - `num_stages` (`Int`): Number of stages in the paired explicit Runge-Kutta (P-ERK) method.
    - `tspan`: Time span of the simulation.
    - `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
    -  `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
      equation has been semidiscretized.
    - `verbose` (`Bool`, optional): Verbosity flag, default is false.
    - `cS2` (`Float64`, optional): Value of c in the Butcher tableau at c_{s-2}, when
      s is the number of stages, default is 1.0f0.

The following structures and methods provide an implementation of
the third-order paired explicit Runge-Kutta (P-ERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The original paper is
- Nasab, Vermeire (2022)
Third-order Paired Explicit Runge-Kutta schemes for stiff systems of equations
[DOI: 10.1016/j.jcp.2022.111470](https://doi.org/10.1016/j.jcp.2022.111470)
While the changes to SSPRK33 base-scheme are described in 
- Doehring, Schlottke-Lakemper, Gassner, Torrilhon (2024)
Multirate Time-Integration based on Dynamic ODE Partitioning through Adaptively Refined Meshes for Compressible Fluid Dynamics
[Arxiv: 10.48550/arXiv.2403.05144](https://doi.org/10.48550/arXiv.2403.05144)
"""
mutable struct PairedExplicitRK3 <: AbstractPairedExplicitRKSingle
    const num_stages::Int # S

    a_matrix::Matrix{Float64}
    c::Vector{Float64}
end # struct PairedExplicitRK3

# Constructor for previously computed A Coeffs
function PairedExplicitRK3(path_a_coeffs::AbstractString;
                           cS2 = 1.0f0)
    num_stages, a_matrix, c = compute_PairedExplicitRK3_butcher_tableau(path_a_coeffs;
                                                                        cS2)

    return PairedExplicitRK3(num_stages, a_matrix, c)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function PairedExplicitRK3(num_stages, tspan, semi::AbstractSemidiscretization;
                           verbose = false, cS2 = 1.0f0)
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return PairedExplicitRK3(num_stages, tspan, eig_vals; verbose, cS2)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function PairedExplicitRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                           verbose = false, cS2 = 1.0f0)
    a_matrix, c = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                            tspan,
                                                            eig_vals;
                                                            verbose, cS2)
    return PairedExplicitRK3(num_stages, a_matrix, c)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRK3Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
                                           PairedExplicitRKOptions} <:
               AbstractPairedExplicitRKSingleIntegrator
    u::uType
    du::uType
    u_tmp::uType
    t::RealT
    tdir::RealT
    dt::RealT # current time step
    dtcache::RealT # manually set time step
    iter::Int # current number of time steps (iteration)
    p::Params # will be the semidiscretization from Trixi
    sol::Sol # faked
    f::F
    alg::Alg # This is our own class written above; Abbreviation for ALGorithm
    opts::PairedExplicitRKOptions
    finalstep::Bool # added for convenience
    dtchangeable::Bool
    force_stepfail::Bool
    # PairedExplicitRK stages:
    k1::uType
    k_higher::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK3;
              dt, callback = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # PairedExplicitRK stages
    k1 = zero(u0)
    k_higher = zero(u0)

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK3Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
                                             ode.p,
                                             (prob = ode,), ode.f, alg,
                                             PairedExplicitRKOptions(callback,
                                                                     ode.tspan;
                                                                     kwargs...),
                                             false, true, false,
                                             k1, k_higher)

    # initialize callbacks
    if callback isa CallbackSet
        for cb in callback.continuous_callbacks
            error("Continuous callbacks are unsupported with paired explicit Runge-Kutta methods.")
        end
        for cb in callback.discrete_callbacks
            cb.initialize(cb, integrator.u, integrator.t, integrator)
        end
    elseif !isnothing(callback)
        error("unsupported")
    end

    return integrator
end

# Fakes `solve`: https://diffeq.sciml.ai/v6.8/basics/overview/#Solving-the-Problems-1
function solve(ode::ODEProblem, alg::PairedExplicitRK3;
               dt, callback = nothing, kwargs...)
    integrator = init(ode, alg, dt = dt, callback = callback; kwargs...)

    # Start actual solve
    solve!(integrator)
end

function solve!(integrator::PairedExplicitRK3Integrator)
    @unpack prob = integrator.sol

    integrator.finalstep = false

    @trixi_timeit timer() "main loop" while !integrator.finalstep
        step!(integrator)
    end # "main loop" timer

    return TimeIntegratorSolution((first(prob.tspan), integrator.t),
                                  (prob.u0, integrator.u),
                                  integrator.sol.prob)
end

function step!(integrator::PairedExplicitRK3Integrator)
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
        # k1
        integrator.f(integrator.du, integrator.u, prob.p, integrator.t)
        @threaded for i in eachindex(integrator.du)
            integrator.k1[i] = integrator.du[i] * integrator.dt
        end

        # Construct current state
        @threaded for i in eachindex(integrator.du)
            integrator.u_tmp[i] = integrator.u[i] + alg.c[2] * integrator.k1[i]
        end
        # k2
        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[2] * integrator.dt)

        @threaded for i in eachindex(integrator.du)
            integrator.k_higher[i] = integrator.du[i] * integrator.dt
        end

        # Higher stages
        for stage in 3:(alg.num_stages - 1)
            # Construct current state
            @threaded for i in eachindex(integrator.du)
                integrator.u_tmp[i] = integrator.u[i] +
                                      alg.a_matrix[stage - 2, 1] *
                                      integrator.k1[i] +
                                      alg.a_matrix[stage - 2, 2] *
                                      integrator.k_higher[i]
            end

            integrator.f(integrator.du, integrator.u_tmp, prob.p,
                         integrator.t + alg.c[stage] * integrator.dt)

            @threaded for i in eachindex(integrator.du)
                integrator.k_higher[i] = integrator.du[i] * integrator.dt
            end
        end

        # Last stage
        @threaded for i in eachindex(integrator.du)
            integrator.u_tmp[i] = integrator.u[i] +
                                  alg.a_matrix[alg.num_stages - 2, 1] *
                                  integrator.k1[i] +
                                  alg.a_matrix[alg.num_stages - 2, 2] *
                                  integrator.k_higher[i]
        end

        integrator.f(integrator.du, integrator.u_tmp, prob.p,
                     integrator.t + alg.c[alg.num_stages] * integrator.dt)

        @threaded for i in eachindex(integrator.u)
            # "Own" PairedExplicitRK based on SSPRK33.
            # Note that 'k_higher' carries the values of K_{S-1}
            # and that we construct 'K_S' "in-place" from 'integrator.du'
            integrator.u[i] += (integrator.k1[i] + integrator.k_higher[i] +
                                4.0 * integrator.du[i] * integrator.dt) / 6.0
        end
    end # PairedExplicitRK step timer

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

# used for AMR (Adaptive Mesh Refinement)
function Base.resize!(integrator::PairedExplicitRK3Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.k_higher, new_size)
end
end # @muladd
