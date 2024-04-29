# implemented here
function filter_eigvals(eigvals, threshold = 1e-12)

    filtered_eigvals = Complex{Float64}[]

    for eig_val in eigvals
        # Filter out eigenvalues with positive real parts, those with negative imaginary parts due to eigenvalues' symmetry
        # around real axis, or the eigenvalues that are too small.
        if real(eig_val) < 0 && imag(eig_val) > 0 && abs(eig_val) >= threshold
            push!(filtered_eigvals, eig_val)
        end
    end

    filtered_eigvals_count = length(eigvals) - length(filtered_eigvals)
    println("$filtered_eigvals_count eigenvalue(s) are not passed on because " *
            "they either are in magnitude smaller than $threshold, have positive " *
            "real parts, or have negative imaginary parts.\n")

    return length(filtered_eigvals), filtered_eigvals
end

function undo_normalization!(cons_order, num_stage_evals, gamma_opt)
    for k in (cons_order + 1):num_stage_evals
        gamma_opt[k - cons_order] = gamma_opt[k - cons_order] / factorial(k)
    end
    return gamma_opt
end

# Add function definitions here such that they can be exported from Trixi.jl and extended in the
# TrixiConvexECOSExt package extension or by the Convex and ECOS-specific code loaded by Requires.jl
function stability_polynomials end

function bisection end
