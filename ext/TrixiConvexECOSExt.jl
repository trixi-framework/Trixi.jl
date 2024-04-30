# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexECOSExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex: MOI, solve!, Variable, minimize, evaluate
    using ECOS
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex: MOI, solve!, Variable, minimize, evaluate
    using ..ECOS
end

# Use other necessary libraries
using LinearAlgebra: eigvals

# Use additional symbols that are not exported
using Trixi: @muladd

# Import functions such that they can be extended with new methods
import Trixi: filter_eig_vals, undo_normalization!, stability_polynomials!,
              bisect_stability_polynomial

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Filter out eigenvalues with positive real parts, those with negative imaginary
# parts due to eigenvalues' symmetry around the real axis, or the eigenvalues
# that are smaller than a specified threshold.
function filter_eig_vals(eig_vals, verbose, threshold = 1e-12)
    filtered_eig_vals = Complex{Float64}[]

    for eig_val in eig_vals
        if real(eig_val) < 0 && imag(eig_val) > 0 && abs(eig_val) >= threshold
            push!(filtered_eig_vals, eig_val)
        end
    end

    filtered_eig_vals_count = length(eig_vals) - length(filtered_eig_vals)

    if verbose
        println("$filtered_eig_vals_count eigenvalue(s) are not passed on because " *
                "they either are in magnitude smaller than $threshold, have positive " *
                "real parts, or have negative imaginary parts.\n")
    end

    return length(filtered_eig_vals), filtered_eig_vals
end

# Undo normalization of stability polynomial coefficients by index factorial
# relative to consistency order.
function undo_normalization!(consistency_order, num_stage_evals, gamma_opt)
    for k in (consistency_order + 1):num_stage_evals
        gamma_opt[k - consistency_order] = gamma_opt[k - consistency_order] /
                                           factorial(k)
    end
    return gamma_opt
end

# Compute stability polynomials for paired explicit Runge-Kutta up to specified consistency
# order, including contributions from free coefficients for higher orders, and
# return the maximum absolute value
function stability_polynomials!(pnoms, consistency_order, num_stage_evals,
                                normalized_powered_eigvals_scaled,
                                gamma)
    num_eig_vals = length(pnoms)

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

    # Contribution from free coefficients
    for k in (consistency_order + 1):num_stage_evals
        pnoms += gamma[k - consistency_order] * normalized_powered_eigvals_scaled[:, k]
    end

    # For optimization only the maximum is relevant
    return maximum(abs(pnoms))
end

#=
bisection()

The following structures and methods provide a simplified implementation to 
discover optimal stability polynomial for a given set of `eig_vals`
These are designed for the one-step (i.e., Runge-Kutta methods) integration of initial value ordinary 
and partial differential equations.

- Ketcheson and Ahmadia (2012).
Optimal stability polynomials for numerical integration of initial value problems
[DOI: 10.2140/camcos.2012.7.247](https://doi.org/10.2140/camcos.2012.7.247)
=#

# Perform bisection to optimize timestep for stability of the polynomial
function bisect_stability_polynomial(consistency_order, num_eig_vals, num_stage_evals,
                                     dtmax, dteps,
                                     eig_vals,
                                     verbose)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals - consistency_order)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)

    for j in 1:num_stage_evals
        fac_j = factorial(j)
        for i in 1:num_eig_vals
            normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
        end
    end

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

        # Use last optimal values for gamma in (potentially) next iteration
        problem = minimize(stability_polynomials!(pnoms, consistency_order,
                                                  num_stage_evals,
                                                  normalized_powered_eigvals_scaled,
                                                  gamma))

        solve!(problem,
               # Parameters taken from default values for EiCOS
               MOI.OptimizerWithAttributes(ECOS.Optimizer, "gamma" => 0.99,
                                           "delta" => 2e-7,
                                           "feastol" => 1e-9,
                                           "abstol" => 1e-9,
                                           "reltol" => 1e-9,
                                           "feastol_inacc" => 1e-4,
                                           "abstol_inacc" => 5e-5,
                                           "reltol_inacc" => 5e-5,
                                           "nitref" => 9,
                                           "maxit" => 100,
                                           "verbose" => 3); silent_solver = true)

        abs_p = problem.optval

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    if verbose
        println("Concluded stability polynomial optimization \n")
    end

    return evaluate(gamma), dt
end
end # @muladd

end # module TrixiConvexECOSExt
