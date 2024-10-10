# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_smoothing!(mesh::UnstructuredMesh2D, alpha, alpha_tmp, dg, cache)
    # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
    # Copy alpha values such that smoothing is indpedenent of the element access order
    alpha_tmp .= alpha

    # Loop over interfaces
    for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = cache.interfaces.element_ids[1, interface]
        right = cache.interfaces.element_ids[2, interface]

        # Apply smoothing
        alpha[left] = max(alpha_tmp[left], 0.5f0 * alpha_tmp[right], alpha[left])
        alpha[right] = max(alpha_tmp[right], 0.5f0 * alpha_tmp[left], alpha[right])
    end
end
end # @muladd
