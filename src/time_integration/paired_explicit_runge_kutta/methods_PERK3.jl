# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Initialize Butcher array abscissae c for PairedExplicitRK3 based on SSPRK33 base method
function compute_c_coeffs(num_stages, cS2)
    c = zeros(eltype(cS2), num_stages)

    # Last timesteps as for SSPRK33, see motivation in Section 3.3 of
    # https://doi.org/10.1016/j.jcp.2024.113223
    c[num_stages - 1] = one(cS2)
    c[num_stages] = convert(eltype(cS2), 0.5)

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

    # Calculate coefficients of the stability polynomial in monomial form
    consistency_order = 3
    dtmax = tspan[2] - tspan[1]
    dteps = 1.0f-9

    num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

    monomial_coeffs, dt_opt = bisect_stability_polynomial(consistency_order,
                                                          num_eig_vals, num_stages,
                                                          dtmax, dteps,
                                                          eig_vals; verbose)

    # Special case of e = 3
    if num_stages == consistency_order
        a_unknown = [0.25] # Use classic SSPRK33 (Shu-Osher) Butcher Tableau
    else
        undo_normalization!(monomial_coeffs, consistency_order, num_stages)

        # Solve the nonlinear system of equations from monomial coefficient and
        # Butcher array abscissae c to find Butcher matrix A
        # This function is extended in TrixiNLsolveExt.jl
        a_unknown = solve_a_butcher_coeffs_unknown!(a_unknown, num_stages,
                                                    monomial_coeffs, c;
                                                    verbose)
    end
    # Fill A-matrix in PERK style
    a_matrix = zeros(2, num_stages - 2)
    a_matrix[1, :] = c[3:end]
    a_matrix[1, :] -= a_unknown
    a_matrix[2, :] = a_unknown

    return a_matrix, c, dt_opt
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 3
# using provided values of coefficients a in A-matrix of Butcher tableau
function compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                   base_path_a_coeffs::AbstractString;
                                                   cS2)

    # Initialize array of c
    c = compute_c_coeffs(num_stages, cS2)

    # - 2 Since First entry of A is always zero (explicit method) and second is given by c_2 (consistency)
    a_coeffs_max = num_stages - 2

    a_matrix = zeros(2, a_coeffs_max)
    a_matrix[1, :] = c[3:end]

    path_a_coeffs = joinpath(base_path_a_coeffs,
                             "a_" * string(num_stages) * ".txt")

    @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
    a_coeffs = readdlm(path_a_coeffs, Float64)
    num_a_coeffs = size(a_coeffs, 1)

    @assert num_a_coeffs == a_coeffs_max
    # Fill A-matrix in PERK style
    a_matrix[1, :] -= a_coeffs
    a_matrix[2, :] = a_coeffs

    return a_matrix, c
end

@doc raw"""
    PairedExplicitRK3(num_stages, base_path_a_coeffs::AbstractString, dt_opt = nothing;
                      cS2 = 1.0f0)
    PairedExplicitRK3(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, cS2 = 1.0f0)
    PairedExplicitRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, cS2 = 1.0f0)

    Parameters:
    - `num_stages` (`Int`): Number of stages in the paired explicit Runge-Kutta (PERK) method.
    - `base_path_a_coeffs` (`AbstractString`): Path to a file containing some coefficients in the A-matrix in 
      the Butcher tableau of the Runge Kutta method.
      The matrix should be stored in a text file at `joinpath(base_path_a_coeffs, "a_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`, optional): Optimal time step size for the simulation setup. Can be `nothing` if it is unknown. 
       In this case the optimal CFL number cannot be computed and the [`StepsizeCallback`](@ref) cannot be used.
    - `tspan`: Time span of the simulation.
    - `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
    - `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
      equation has been semidiscretized.
    - `verbose` (`Bool`, optional): Verbosity flag, default is false.
    - `cS2` (`Float64`, optional): Value of $c_{S-2}$ in the Butcher tableau, where
      $S$ is the number of stages. Default is 1.0f0.

The following structures and methods provide an implementation of
the third-order paired explicit Runge-Kutta (PERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The original paper is
- Nasab, Vermeire (2022)
Third-order Paired Explicit Runge-Kutta schemes for stiff systems of equations
[DOI: 10.1016/j.jcp.2022.111470](https://doi.org/10.1016/j.jcp.2022.111470)
While the changes to SSPRK33 base-scheme are described in 
- Doehring, Schlottke-Lakemper, Gassner, Torrilhon (2024)
Multirate Time-Integration based on Dynamic ODE Partitioning through Adaptively Refined Meshes for Compressible Fluid Dynamics
[DOI: 10.1016/j.jcp.2024.113223](https://doi.org/10.1016/j.jcp.2024.113223)

Note: To use this integrator, the user must import the `Convex`, `ECOS`, and `NLsolve` packages
unless the A-matrix coefficients are provided in a "a_<num_stages>.txt" file.
"""
struct PairedExplicitRK3 <: AbstractPairedExplicitRKSingle
    num_stages::Int # S

    a_matrix::Matrix{Float64}
    c::Vector{Float64}

    dt_opt::Union{Float64, Nothing}
end

# Constructor for previously computed A Coeffs
function PairedExplicitRK3(num_stages, base_path_a_coeffs::AbstractString,
                           dt_opt = nothing;
                           cS2 = 1.0f0)
    @assert num_stages>=3 "PERK3 requires at least three stages"
    a_matrix, c = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                            base_path_a_coeffs;
                                                            cS2)

    return PairedExplicitRK3(num_stages, a_matrix, c, dt_opt)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function PairedExplicitRK3(num_stages, tspan, semi::AbstractSemidiscretization;
                           verbose = false, cS2 = 1.0f0)
    @assert num_stages>=3 "PERK3 requires at least three stages"
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return PairedExplicitRK3(num_stages, tspan, eig_vals; verbose, cS2)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function PairedExplicitRK3(num_stages, tspan, eig_vals::Vector{ComplexF64};
                           verbose = false, cS2 = 1.0f0)
    @assert num_stages>=3 "PERK3 requires at least three stages"
    a_matrix, c, dt_opt = compute_PairedExplicitRK3_butcher_tableau(num_stages,
                                                                    tspan,
                                                                    eig_vals;
                                                                    verbose, cS2)
    return PairedExplicitRK3(num_stages, a_matrix, c, dt_opt)
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
    # Additional PERK3 registers
    k1::uType
    kS1::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK3;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    # Additional PERK3 registers
    k1 = zero(u0)
    kS1 = zero(u0)

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
                                             k1, kS1)

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
        # First and second stage are identical across all single/standalone PERK methods
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        for stage in 3:(alg.num_stages - 1)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        # We need to store `du` of the S-1 stage in `kS1` for the final update:
        @threaded for i in eachindex(integrator.u)
            integrator.kS1[i] = integrator.du[i]
        end

        PERK_ki!(integrator, prob.p, alg, alg.num_stages)

        @threaded for i in eachindex(integrator.u)
            # "Own" PairedExplicitRK based on SSPRK33.
            # Note that 'kS1' carries the values of K_{S-1}
            # and that we construct 'K_S' "in-place" from 'integrator.du'
            integrator.u[i] += integrator.dt *
                               (integrator.k1[i] + integrator.kS1[i] +
                                4.0 * integrator.du[i]) / 6.0
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

function Base.resize!(integrator::PairedExplicitRK3Integrator, new_size)
    resize!(integrator.u, new_size)
    resize!(integrator.du, new_size)
    resize!(integrator.u_tmp, new_size)

    resize!(integrator.k1, new_size)
    resize!(integrator.kS1, new_size)
end
end # @muladd
