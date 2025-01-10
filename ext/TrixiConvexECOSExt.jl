# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexECOSExt

# Required for coefficient optimization in PERK scheme integrators
using Convex: MOI, solve!, Variable, minimize, evaluate
using ECOS: Optimizer

# Use other necessary libraries
using LinearAlgebra: eigvals

# Use functions that are to be extended and additional symbols that are not exported
using Trixi: Trixi, undo_normalization!, bisect_stability_polynomial, @muladd

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

# Compute stability polynomials for paired explicit Runge-Kutta up to specified consistency
# order, including contributions from free coefficients for higher orders, and
# return the maximum absolute value
function stability_polynomials!(pnoms, num_stage_evals,
                                normalized_powered_eigvals_scaled,
                                gamma, consistency_order)
    # Initialize with zero'th order (z^0) coefficient
    pnoms .= 1

    # First `consistency_order` terms of the exponential
    for k in 1:consistency_order
        pnoms += view(normalized_powered_eigvals_scaled, :, k)
    end

    # Contribution from free coefficients
    for k in (consistency_order + 1):num_stage_evals
        pnoms += gamma[k - consistency_order] *
                 view(normalized_powered_eigvals_scaled, :, k)
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
                                      normalized_powered_eigvals_scaled,
                                      gamma, cS3)
    # Constants arising from the particular form of Butcher tableau chosen for the 4th order PERK methods
    # cS3 = c_{S-3}
    k1 = 0.001055026310046423 / cS3
    k2 = 0.03726406530405851 / cS3

    # Initialize with zero'th order (z^0) coefficient
    pnoms .= 1

    # First `consistency_order` = 4 terms of the exponential
    for k in 1:4
        pnoms += view(normalized_powered_eigvals_scaled, :, k)
    end

    # "Fixed" term due to choice of the PERK4 Butcher tableau
    # Required to un-do the normalization of the eigenvalues here
    pnoms += k1 * view(normalized_powered_eigvals_scaled, :, 5) * factorial(5)

    # Contribution from free coefficients
    for k in 1:(num_stage_evals - 5)
        pnoms += (k2 * view(normalized_powered_eigvals_scaled, :, k + 4) *
                  gamma[k] +
                  k1 * view(normalized_powered_eigvals_scaled, :, k + 5) *
                  gamma[k] * (k + 5)) # Ensure same normalization of both summands
    end

    # For optimization only the maximum is relevant
    if num_stage_evals == 5
        return maximum(abs.(pnoms)) # If there is no variable to optimize, we need to use the broadcast operator.
    else
        return maximum(abs(pnoms))
    end
end

@inline function normalize_power_eigvals!(normalized_powered_eigvals,
                                          eig_vals,
                                          num_stage_evals)
    for j in 1:num_stage_evals
        @views normalized_powered_eigvals[:, j] = eig_vals[:] .^ j ./ factorial(j)
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

For the fourth-order PERK schemes, a specialized optimization routine is required, see
- D. Doehring, L. Christmann, M. Schlottke-Lakemper, G. J. Gassner and M. Torrilhon (2024).
 Fourth-Order Paired-Explicit Runge-Kutta Methods
 [DOI:10.48550/arXiv.2408.05470](https://doi.org/10.48550/arXiv.2408.05470)
=#

# Perform bisection to optimize timestep for stability of the polynomial
function Trixi.bisect_stability_polynomial(consistency_order, num_eig_vals,
                                           num_stage_evals,
                                           dtmax, dteps, eig_vals;
                                           kwargs...)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    if consistency_order == 4
        # Fourth-order scheme has one additional fixed coefficient
        num_reduced_unknown = 5
    else # p = 2, 3
        num_reduced_unknown = consistency_order
    end

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for monomial coefficients
    gamma = Variable(num_stage_evals - num_reduced_unknown)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)
    normalize_power_eigvals!(normalized_powered_eigvals,
                             eig_vals,
                             num_stage_evals)

    normalized_powered_eigvals_scaled = similar(normalized_powered_eigvals)

    if kwargs[:verbose]
        println("Start optimization of stability polynomial \n")
    end

    # Bisection on timestep
    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        for k in 1:num_stage_evals
            @views normalized_powered_eigvals_scaled[:, k] = dt^k .*
                                                             normalized_powered_eigvals[:,
                                                                                        k]
        end

        # Check if there are variables to optimize
        if num_stage_evals - num_reduced_unknown > 0
            # Use last optimal values for gamma in (potentially) next iteration
            if consistency_order == 4
                problem = minimize(stability_polynomials_PERK4!(pnoms,
                                                                num_stage_evals,
                                                                normalized_powered_eigvals_scaled,
                                                                gamma, kwargs[:cS3]))
            else # p = 2, 3
                problem = minimize(stability_polynomials!(pnoms,
                                                          num_stage_evals,
                                                          normalized_powered_eigvals_scaled,
                                                          gamma, consistency_order))
            end

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
            if consistency_order == 4
                abs_p = stability_polynomials_PERK4!(pnoms,
                                                     num_stage_evals,
                                                     normalized_powered_eigvals_scaled,
                                                     gamma, kwargs[:cS3])
            else
                abs_p = stability_polynomials!(pnoms,
                                               num_stage_evals,
                                               normalized_powered_eigvals_scaled,
                                               gamma, consistency_order)
            end
        end

        if abs_p < 1
            dtmin = dt
        else
            dtmax = dt
        end
    end

    if kwargs[:verbose]
        println("Concluded stability polynomial optimization \n")
    end

    if num_stage_evals - consistency_order > 0
        gamma_opt = evaluate(gamma)
    else
        gamma_opt = nothing # If there is no variable to optimize, return gamma_opt as nothing.
    end

    # Catch case with only one optimization variable
    if isa(gamma_opt, Number)
        gamma_opt = [gamma_opt]
    end

    undo_normalization!(gamma_opt, num_stage_evals,
                        num_reduced_unknown, consistency_order)

    return gamma_opt, dt
end
end # @muladd

end # module TrixiConvexECOSExt
