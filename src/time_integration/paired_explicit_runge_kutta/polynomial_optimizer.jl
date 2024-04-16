module PolynomialOptimizer

# implemented here
export filter_eigvals, undo_normalization!

# implemented in TrixiConvexExt and only usable when user adds Convex
export stability_polynomials, bisection

function filter_eigvals(eig_vals, threshold)
    filtered_eigvals_counter = 0
    filtered_eig_vals = Complex{Float64}[]

    for eig_val in eig_vals
        # Filter out eigenvalues with positive real parts, those with negative imaginary parts due to eigenvalues' symmetry
        # around real axis, or the eigenvalues that are too small.
        if real(eig_val) > 0 || imag(eig_val) < 0 || abs(eig_val) < threshold
            filtered_eigvals_counter += 1
        else
            push!(filtered_eig_vals, eig_val)
        end
    end

    println("$filtered_eigvals_counter eigenvalue(s) are not passed on because " *
            "they either are in magnitude smaller than $threshold, have positive " *
            "real parts, or have negative imaginary parts.\n")

    return length(filtered_eig_vals), filtered_eig_vals
end

function undo_normalization!(cons_order, num_stage_evals, gamma_opt)
    for k in (cons_order + 1):num_stage_evals
        gamma_opt[k - cons_order] = gamma_opt[k - cons_order] / factorial(k)
    end
    return gamma_opt
end

function stability_polynomials end

function bisection end

end # module PolynomialOptimizer
