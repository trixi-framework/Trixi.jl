# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function PERK4_compute_c_coeffs(num_stages, cS3)
    c = ones(num_stages) # Best internal stability properties
    c[1] = 0.0

    c[num_stages - 3] = cS3
    c[num_stages - 2] = 0.479274057836310
    c[num_stages - 1] = sqrt(3) / 6 + 0.5
    c[num_stages] = -sqrt(3) / 6 + 0.5

    return c
end

# Constant/non-optimized part of the Butcher matrix
function PERK4_a_matrix_constant(cS3)
    return [(0.479274057836310-(0.114851811257441 / cS3)) 0.1397682537005989 0.1830127018922191
            0.114851811257441/cS3 0.648906880894214 0.028312163512968]
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 4
# using a list of eigenvalues
function compute_PairedExplicitRK4_butcher_tableau(num_stages, tspan,
                                                   eig_vals::Vector{ComplexF64};
                                                   verbose = false, cS3)
    c = PERK4_compute_c_coeffs(num_stages, cS3)

    num_coeffs_max = num_stages - 5
    a_matrix = zeros(2, num_coeffs_max)

    # Calculate coefficients of the stability polynomial in monomial form
    dtmax = tspan[2] - tspan[1]
    dteps = 1.0f-9

    num_eig_vals, eig_vals = filter_eig_vals(eig_vals; verbose)

    monomial_coeffs, dt_opt = bisect_stability_polynomial_PERK4(num_eig_vals,
                                                                num_stages,
                                                                dtmax, dteps,
                                                                eig_vals, cS3;
                                                                verbose)

    if num_stages > 5
        a_unknown = copy(monomial_coeffs)
        for i in 5:(num_stages - 2)
            a_unknown_1[i - 3] /= monomial_coeffs[i - 4]
        end
        reverse!(a_unknown)

        a_matrix = zeros(2, num_coeffs_max)
        a_matrix[1, :] = c[3:(num_stages - 3)]

        a_matrix[1, :] -= a_unknown
        a_matrix[2, :] = a_unknown
    end

    a_matrix_constant = PERK4_a_matrix_constant(cS3)

    return a_matrix, a_matrix_constant, c, dt_opt
end

# Compute the Butcher tableau for a paired explicit Runge-Kutta method order 4
# using provided values of coefficients a in A-matrix of Butcher tableau
function compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                   base_path_a_coeffs::AbstractString,
                                                   cS3)
    c = PERK4_compute_c_coeffs(num_stages, cS3)

    num_coeffs_max = num_stages - 5

    a_matrix = zeros(2, num_coeffs_max)
    a_matrix[1, :] = c[3:(num_stages - 3)]

    path_a_coeffs = joinpath(base_path_a_coeffs,
                             "a_" * string(num_stages) * ".txt")

    @assert isfile(path_a_coeffs) "Couldn't find file $path_a_coeffs"
    a_coeffs = readdlm(path_a_coeffs, Float64)
    num_a_coeffs = size(a_coeffs, 1)

    @assert num_a_coeffs == num_coeffs_max
    if num_coeffs_max > 0
        a_matrix[1, :] -= a_coeffs
        a_matrix[2, :] = a_coeffs
    end

    a_matrix_constant = PERK4_a_matrix_constant(cS3)

    return a_matrix, a_matrix_constant, c
end

@doc raw"""
    PairedExplicitRK4(num_stages, base_path_a_coeffs::AbstractString, dt_opt = nothing;
                      cS3 = 1.0f0)
    PairedExplicitRK4(num_stages, tspan, semi::AbstractSemidiscretization;
                      verbose = false, cS3 = 1.0f0)
    PairedExplicitRK4(num_stages, tspan, eig_vals::Vector{ComplexF64};
                      verbose = false, cS3 = 1.0f0)

    Parameters:
    - `num_stages` (`Int`): Number of stages in the paired explicit Runge-Kutta (P-ERK) method.
    - `base_path_a_coeffs` (`AbstractString`): Path to a file containing some coefficients in the A-matrix in 
      the Butcher tableau of the Runge Kutta method.
      The matrix should be stored in a text file at `joinpath(base_path_a_coeffs, "a_$(num_stages).txt")` and separated by line breaks.
    - `dt_opt` (`Float64`, optional): Optimal time step size for the simulation setup. Can be `nothing` if it is unknown. 
       In this case the optimal CFL number cannot be computed and the [`StepsizeCallback`](@ref) cannot be used.
    - `tspan`: Time span of the simulation.
    - `semi` (`AbstractSemidiscretization`): Semidiscretization setup.
    - `eig_vals` (`Vector{ComplexF64}`): Eigenvalues of the Jacobian of the right-hand side (rhs) of the ODEProblem after the
      equation has been semidiscretized.
    - `cS3` (`Float64`, optional): Value of $c_{S-3}$ in the Butcher tableau, where
      $S$ is the number of stages. Default is `1.0f0`.

The following structures and methods provide an implementation of
the fourth-order paired explicit Runge-Kutta (P-ERK) method
optimized for a certain simulation setup (PDE, IC & BC, Riemann Solver, DG Solver).
The method has been proposed in 
- D. Doehring, L. Christmann, M. Schlottke-Lakemper, G. J. Gassner and M. Torrilhon (2024).
  Fourth-Order Paired-Explicit Runge-Kutta Methods
  [DOI:10.48550/arXiv.2408.05470](https://doi.org/10.48550/arXiv.2408.05470)
"""
struct PairedExplicitRK4 <: AbstractPairedExplicitRKSingle
    num_stages::Int # S

    # Optimized coefficients, i.e., flexible part of the Butcher array matrix A.
    a_matrix::Union{Matrix{Float64}, Nothing}
    # This part of the Butcher array matrix A is constant for all PERK methods, i.e., 
    # regardless of the optimized coefficients.
    a_matrix_constant::Matrix{Float64}
    c::Vector{Float64}

    dt_opt::Union{Float64, Nothing}
end # struct PairedExplicitRK4

# Constructor for previously computed A Coeffs
function PairedExplicitRK4(num_stages, base_path_a_coeffs::AbstractString,
                           dt_opt = nothing;
                           cS3 = 1.0f0)  # Default value for best internal stability
    @assert num_stages>=5 "PERK4 requires at least five stages"
    a_matrix, a_matrix_constant, c = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                               base_path_a_coeffs,
                                                                               cS3)

    return PairedExplicitRK4(num_stages, a_matrix, a_matrix_constant, c, dt_opt)
end

# Constructor that computes Butcher matrix A coefficients from a semidiscretization
function PairedExplicitRK4(num_stages, tspan, semi::AbstractSemidiscretization;
                           verbose = false, cS3 = 1.0f0)
    @assert num_stages>=5 "PERK4 requires at least five stages"
    eig_vals = eigvals(jacobian_ad_forward(semi))

    return PairedExplicitRK4(num_stages, tspan, eig_vals; verbose, cS3)
end

# Constructor that calculates the coefficients with polynomial optimizer from a list of eigenvalues
function PairedExplicitRK4(num_stages, tspan, eig_vals::Vector{ComplexF64};
                           verbose = false, cS3 = 1.0f0)
    @assert num_stages>=5 "PERK4 requires at least five stages"
    a_matrix, a_matrix_constant, c, dt_opt = compute_PairedExplicitRK4_butcher_tableau(num_stages,
                                                                                       tspan,
                                                                                       eig_vals;
                                                                                       verbose,
                                                                                       cS3)
    return PairedExplicitRK4(num_stages, a_matrix, a_matrix_constant, c, dt_opt)
end

# This struct is needed to fake https://github.com/SciML/OrdinaryDiffEq.jl/blob/0c2048a502101647ac35faabd80da8a5645beac7/src/integrators/type.jl#L77
# This implements the interface components described at
# https://diffeq.sciml.ai/v6.8/basics/integrator/#Handing-Integrators-1
# which are used in Trixi.jl.
mutable struct PairedExplicitRK4Integrator{RealT <: Real, uType, Params, Sol, F, Alg,
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
    # Additional PERK register
    k1::uType
end

function init(ode::ODEProblem, alg::PairedExplicitRK4;
              dt, callback::Union{CallbackSet, Nothing} = nothing, kwargs...)
    u0 = copy(ode.u0)
    du = zero(u0)
    u_tmp = zero(u0)

    k1 = zero(u0) # Additional PERK register

    t0 = first(ode.tspan)
    tdir = sign(ode.tspan[end] - ode.tspan[1])
    iter = 0

    integrator = PairedExplicitRK4Integrator(u0, du, u_tmp, t0, tdir, dt, dt, iter,
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

# Computes last three stages, i.e., i = S-2, S-1, S
@inline function PERK4_kS2_to_kS!(integrator::PairedExplicitRK4Integrator, p, alg)
    for stage in 1:2
        @threaded for i in eachindex(integrator.u)
            integrator.u_tmp[i] = integrator.u[i] +
                                  integrator.dt *
                                  (alg.a_matrix_constant[1, stage] *
                                   integrator.k1[i] +
                                   alg.a_matrix_constant[2, stage] *
                                   integrator.du[i])
        end

        integrator.f(integrator.du, integrator.u_tmp, p,
                     integrator.t +
                     alg.c[alg.num_stages - 3 + stage] * integrator.dt)
    end

    # Last stage
    @threaded for i in eachindex(integrator.u)
        integrator.u_tmp[i] = integrator.u[i] +
                              integrator.dt *
                              (alg.a_matrix_constant[1, 3] * integrator.k1[i] +
                               alg.a_matrix_constant[2, 3] * integrator.du[i])
    end

    # Store K_{S-1} in `k1`:
    @threaded for i in eachindex(integrator.u)
        integrator.k1[i] = integrator.du[i]
    end

    integrator.f(integrator.du, integrator.u_tmp, p,
                 integrator.t + alg.c[alg.num_stages] * integrator.dt)

    @threaded for i in eachindex(integrator.u)
        # Note that 'k1' carries the values of K_{S-1}
        # and that we construct 'K_S' "in-place" from 'integrator.du'
        integrator.u[i] += 0.5 * integrator.dt *
                           (integrator.k1[i] + integrator.du[i])
    end
end

function step!(integrator::PairedExplicitRK4Integrator)
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
        PERK_k1!(integrator, prob.p)
        PERK_k2!(integrator, prob.p, alg)

        # Higher stages until "constant" stages
        for stage in 3:(alg.num_stages - 3)
            PERK_ki!(integrator, prob.p, alg, stage)
        end

        PERK4_kS2_to_kS!(integrator, prob.p, alg)
    end

    integrator.iter += 1
    integrator.t += integrator.dt

    @trixi_timeit timer() "Step-Callbacks" begin
        # handle callbacks
        if callbacks isa CallbackSet
            for cb in callbacks.discrete_callbacks
                if cb.condition(integrator.u, integrator.t, integrator)
                    cb.affect!(integrator)
                end
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
