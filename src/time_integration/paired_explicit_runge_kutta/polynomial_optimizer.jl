# Filter out eigenvalues with positive real parts, those with negative imaginary
# parts due to eigenvalues' symmetry around the real axis, or the eigenvalues
# that are smaller than a specified threshold.
function filter_eig_vals(eig_vals, threshold = 1e-12; verbose = false)
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

# Add definitions of functions related to polynomial optimization by Convex and ECOS here
# such that hey can be exported from Trixi.jl and extended in the TrixiConvexECOSExt package
# extension or by the Convex and ECOS-specific code loaded by Requires.jl
function undo_normalization! end
function bisect_stability_polynomial end
