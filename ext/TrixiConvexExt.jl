# Package extension for adding Convex-based features to Trixi.jl
module TrixiConvexExt

# Required for coefficient optimization in P-ERK scheme integrators
if isdefined(Base, :get_extension)
    using Convex
    using ECOS
else
    # Until Julia v1.9 is the minimum required version for Trixi.jl, we still support Requires.jl
    using ..Convex
    using ..ECOS
end

# Use all exported symbols to avoid having to rewrite `recipes_makie.jl`
using Trixi: PolynomialOptimizer

import .PolynomialOptimizer: filter_eigvals, bisection, undo_normalization!

using LinearAlgebra: eigvals
const MOI = Convex.MOI

function stability_polynomials(cons_order, num_stage_evals, num_eig_vals,
                               normalized_powered_eigvals_scaled, pnoms, gamma::Variable)
    # Initialize with zero'th order (z^0) coefficient
    for i in 1:num_eig_vals
        pnoms[i] = 1.0
    end

    # First `cons_order` terms of the exponential
    for k in 1:cons_order
        for i in 1:num_eig_vals
            pnoms[i] += normalized_powered_eigvals_scaled[i, k]
        end
    end

    # Contribution from free coefficients
    for k in (cons_order + 1):num_stage_evals
        pnoms += gamma[k - cons_order] * normalized_powered_eigvals_scaled[:, k]
    end

    # For optimization only the maximum is relevant
    return maximum(abs(pnoms))
end

"""
bisection()

The following structures and methods provide a simplified implementation to 
discover optimal stability polynomial for a given set of `eig_vals`
These are designed for the one-step (i.e., Runge-Kutta methods) integration of initial value ordinary 
and partial differential equations.

- Ketcheson and Ahmadia (2012).
Optimal stability polynomials for numerical integration of initial value problems
[DOI: 10.2140/camcos.2012.7.247](https://doi.org/10.2140/camcos.2012.7.247)
"""
function bisection(cons_order, num_eig_vals, num_stage_evals, dtmax, dteps, eig_vals)
    dtmin = 0.0
    dt = -1.0
    abs_p = -1.0

    # Construct stability polynomial for each eigenvalue
    pnoms = ones(Complex{Float64}, num_eig_vals, 1)

    # Init datastructure for results
    gamma = Variable(num_stage_evals - cons_order)

    normalized_powered_eigvals = zeros(Complex{Float64}, num_eig_vals, num_stage_evals)

    for j in 1:num_stage_evals
        fac_j = factorial(j)
        for i in 1:num_eig_vals
            normalized_powered_eigvals[i, j] = eig_vals[i]^j / fac_j
        end
    end

    normalized_powered_eigvals_scaled = zeros(Complex{Float64}, num_eig_vals,
                                              num_stage_evals)

    println("Start optimization of stability polynomial \n")

    while dtmax - dtmin > dteps
        dt = 0.5 * (dtmax + dtmin)

        # Compute stability polynomial for current timestep
        for k in 1:num_stage_evals
            dt_k = dt^k
            for i in 1:num_eig_vals
                normalized_powered_eigvals_scaled[i, k] = dt_k *
                                                          normalized_powered_eigvals[i, k]
            end
        end

        # Use last optimal values for gamma in (potentially) next iteration
        problem = minimize(stability_polynomials(cons_order, num_stage_evals, num_eig_vals,
                                                 normalized_powered_eigvals_scaled, pnoms,
                                                 gamma))

        Convex.solve!(problem,
                      # Parameters taken from default values for EiCOS
                      MOI.OptimizerWithAttributes(ECOS.Optimizer, "gamma" => 0.99,
                                                  "delta" => 2e-7,
                                                  #"eps" => 1e13, # 1e13
                                                  "feastol" => 1e-9, # 1e-9
                                                  "abstol" => 1e-9, # 1e-9
                                                  "reltol" => 1e-9, # 1e-9
                                                  "feastol_inacc" => 1e-4,
                                                  "abstol_inacc" => 5e-5,
                                                  "reltol_inacc" => 5e-5,
                                                  "nitref" => 9,
                                                  "maxit" => 100,
                                                  "verbose" => 3); silent_solver = true)

        abs_p = problem.optval

        if abs_p < 1.0
            dtmin = dt
        else
            dtmax = dt
        end
    end
    println("Concluded stability polynomial optimization \n")

    return evaluate(gamma), dt
end

end # module
