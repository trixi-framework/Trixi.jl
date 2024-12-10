# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexECOSExt

# Required for coefficient optimization in PERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex: MOI, solve!, Variable, minimize, evaluate
    using ECOS: Optimizer
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex: MOI, solve!, Variable, minimize, evaluate
    using ..ECOS: Optimizer
end

# Use other necessary libraries
using LinearAlgebra: eigvals

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, undo_normalization!,
             bisect_stability_polynomial, bisect_stability_polynomial_PERK4, @muladd

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Undo normalization of stability polynomial coefficients by index factorial
# relative to consistency order.
function Trixi.undo_normalization!(gamma_opt, num_stage_evals,
                                   num_reduced_coeffs, fac_offset)
    for k in 1:(num_stage_evals - num_reduced_coeffs)
        gamma_opt[k] /= factorial(k + fac_offset)
    end
end

@inline function stability_polynomials_fixed_coeffs!(pnoms, num_eig_vals,
                                                     normalized_powered_eigvals_scaled,
                                                     consistency_order)
    # Initialize with zero'th order (z^0) coefficient
    for i in 1:num_eig_vals
        pnoms[i] = 1.0
    end

    # First `consistency_order` terms of the exponential
    for k in 1:consistency_order
        for i in 1:num_eig_vals
            pnoms[i] += normalized_powered_eigvals_scaled[i, k]
        end
    end
end

# Compute stability polynomials for paired explicit Runge-Kutta up to specified consistency
# order, including contributions from free coefficients for higher orders, and
# return the maximum absolute value
function stability_polynomials!(pnoms, consistency_order,
                                num_stage_evals,
                                num_eig_vals,
                                normalized_powered_eigvals_scaled,
                                gamma)
    stability_polynomials_fixed_coeffs!(pnoms, num_eig_vals,
                                        normalized_powered_eigvals_scaled,
                                        consistency_order)

    # Contribution from free coefficients
    for k in (consistency_order + 1):num_stage_evals
        pnoms += gamma[k - consistency_order] * normalized_powered_eigvals_scaled[:, k]
    end

    # For optimization only the maximum is relevant
    if consistency_order - num_stage_evals == 0
        return maximum(abs.(pnoms)) # If there is no variable to optimize, we need to use the broadcast operator.
    else
        return maximum(abs(pnoms))
    end
end

# Specialized form of the stability polynomials for fourth-order PERK schemes.
function stability_polynomials_PERK4!(pnoms, num_stage_evals,
                                      num_eig_vals,
                                      normalized_powered_eigvals,
                                      gamma,
                                      dt, cS3)
    # Constants arising from the particular form of Butcher tableau chosen for the 4th order PERK methods
    k1 = 0.001055026310046423 / cS3
    k2 = 0.03726406530405851 / cS3
    # Note: `cS3` = c_{S-3} is in principle free, while the other abscissae are fixed to 1.0

    stability_polynomials_fixed_coeffs!(pnoms, num_eig_vals, normalized_powered_eigvals,
                                        4)

    # "Fixed" term due to choice of the PERK4 Butcher tableau
    # Required to un-do the normalization of the eigenvalues here
    pnoms += k1 * dt^5 * normalized_powered_eigvals[:, 5] * factorial(5)

    # Contribution from free coefficients
    for k in 1:(num_stage_evals - 5)
        pnoms += (k2 * dt^(k + 4) * normalized_powered_eigvals[:, k + 4] * gamma[k] +
                  k1 * dt^(k + 5) * normalized_powered_eigvals[:, k + 5] * gamma[k] *
                  (k + 5))
    end

    # For optimization only the maximum is relevant
    if num_stage_evals == 5
        return maximum(abs.(pnoms)) # If there is no variable to optimize, we need to use the broadcast operator.
    else
        return maximum(abs(pnoms))
    end
end

@inline function normalized_power_eigvals!(normalized_powered_eigvals,
                                           num_eig_vals, eig_vals,
                                           num_stage_evals)
    for j in 1:num_stage_evals
        fac_j = factorial(j)
        for i in 1:num_eig_vals
            normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
        end
    end
end

#=
The following structures and methods provide a simplified implementation to 
discover optimal stability polynomial for a given set of `eig_vals`
These are designed for the one-step (i.e., Runge-Kutta methods) integration of initial value ordinary 
and partial differential equations.

- Ketcheson and Ahmadia (2012).
Optimal stability polynomials for numerical integration of initial value problems
[DOI: 10.2140/camcos.2012.7.247](https://doi.org/10.2140/camcos.2012.7.247)
=#

# Perform bisection to optimize timestep for stability of the polynomial
function Trixi.bisect_stability_polynomial(consistency_order, num_eig_vals,
                                           num_stage_evals,
                                           dtmax, dteps, eig_vals;
                                           verbose = false)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals - consistency_order)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)
    normalized_power_eigvals!(normalized_powered_eigvals,
                              num_eig_vals, eig_vals,
                              num_stage_evals)

    normalized_powered_eigvals_scaled = similar(normalized_powered_eigvals)

    if verbose
        println("Start optimization of stability polynomial \n")
    end

    # Bisection on timestep
    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        # Compute stability polynomial for current timestep
        for k in 1:num_stage_evals
            dt_k = dt^k
            for i in 1:num_eig_vals
                normalized_powered_eigvals_scaled[i, k] = dt_k *
                                                          normalized_powered_eigvals[i,
                                                                                     k]
            end
        end

        # Check if there are variables to optimize
        if num_stage_evals - consistency_order > 0
            # Use last optimal values for gamma in (potentially) next iteration
            problem = minimize(stability_polynomials!(pnoms, consistency_order,
                                                      num_stage_evals,
                                                      num_eig_vals,
                                                      normalized_powered_eigvals_scaled,
                                                      gamma))

            solve!(problem,
                   # Parameters taken from default values for EiCOS
                   MOI.OptimizerWithAttributes(Optimizer, "gamma" => 0.99,
                                               "delta" => 2e-7,
                                               "feastol" => 1e-9,
                                               "abstol" => 1e-9,
                                               "reltol" => 1e-9,
                                               "feastol_inacc" => 1e-4,
                                               "abstol_inacc" => 5e-5,
                                               "reltol_inacc" => 5e-5,
                                               "nitref" => 9,
                                               "maxit" => 100,
                                               "verbose" => 3); silent = true)

            abs_p = problem.optval
        else
            abs_p = stability_polynomials!(pnoms, consistency_order,
                                           num_stage_evals,
                                           num_eig_vals,
                                           normalized_powered_eigvals_scaled,
                                           gamma)
        end

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    if verbose
        println("Concluded stability polynomial optimization \n")
    end

    if num_stage_evals - consistency_order > 0
        gamma_opt = evaluate(gamma)
    else
        gamma_opt = nothing # If there is no variable to optimize, return gamma_opt as nothing.
    end

    # Catch case S = 3 (only one opt. variable)
    if isa(gamma_opt, Number)
        gamma_opt = [gamma_opt]
    end

    undo_normalization!(gamma_opt, num_stage_evals,
                        consistency_order, consistency_order)

    return gamma_opt, dt
end

# Specialized routine for PERK4.
# For details, see Section 4 in 
# - D. Doehring, L. Christmann, M. Schlottke-Lakemper, G. J. Gassner and M. Torrilhon (2024).
# Fourth-Order Paired-Explicit Runge-Kutta Methods
# [DOI:10.48550/arXiv.2408.05470](https://doi.org/10.48550/arXiv.2408.05470)
function Trixi.bisect_stability_polynomial_PERK4(num_eig_vals,
                                                 num_stage_evals,
                                                 dtmax, dteps, eig_vals, cS3;
                                                 verbose = false)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals - 5)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)
    normalized_power_eigvals!(normalized_powered_eigvals,
                              num_eig_vals, eig_vals,
                              num_stage_evals)

    if verbose
        println("Start optimization of stability polynomial \n")
    end

    # Bisection on timestep
    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        if num_stage_evals > 5
            # Use last optimal values for gamma in (potentially) next iteration
            problem = minimize(stability_polynomials_PERK4!(pnoms,
                                                            num_stage_evals,
                                                            num_eig_vals,
                                                            normalized_powered_eigvals,
                                                            gamma, dt, cS3))

            solve!(problem,
                   # Parameters taken from default values for EiCOS
                   MOI.OptimizerWithAttributes(Optimizer, "gamma" => 0.99,
                                               "delta" => 2e-7,
                                               "feastol" => 1e-9,
                                               "abstol" => 1e-9,
                                               "reltol" => 1e-9,
                                               "feastol_inacc" => 1e-4,
                                               "abstol_inacc" => 5e-5,
                                               "reltol_inacc" => 5e-5,
                                               "nitref" => 9,
                                               "maxit" => 100,
                                               "verbose" => 3); silent = true)

            abs_p = problem.optval
        else
            abs_p = stability_polynomials_PERK4!(pnoms, num_stage_evals,
                                                 num_eig_vals,
                                                 normalized_powered_eigvals,
                                                 gamma, dt, cS3)
        end

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    if verbose
        println("Concluded stability polynomial optimization \n")
    end

    if num_stage_evals > 5
        gamma_opt = evaluate(gamma)
    else
        gamma_opt = nothing # If there is no variable to optimize, return gamma_opt as nothing.
    end

    # Catch case S = 6 (only one opt. variable)
    if isa(gamma_opt, Number)
        gamma_opt = [gamma_opt]
    end

    undo_normalization!(gamma_opt, num_stage_evals, 5, 4)

    return gamma_opt, dt
end
end # @muladd

end # module TrixiConvexECOSExt
