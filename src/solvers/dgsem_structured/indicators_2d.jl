# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function apply_smoothing!(mesh::StructuredMesh{2}, alpha, alpha_tmp, dg, cache)
    # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
    # Copy alpha values such that smoothing is indpedenent of the element access order
    alpha_tmp .= alpha

    # So far, alpha smoothing doesn't work for non-periodic initial conditions for structured meshes.
    @assert isperiodic(mesh) "alpha smoothing for structured meshes works only with periodic initial conditions so far"

    @unpack left_neighbors = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access below.
    # The neighbor ids are element ids taken from the mesh connectivity.
    @boundscheck begin
        check_axes(alpha, (eachelement(dg, cache),))
        check_axes(alpha_tmp, (eachelement(dg, cache),))
        check_axes(left_neighbors, (Base.OneTo(2), eachelement(dg, cache)))
    end

    # Loop over elements, because there is no interface container
    @inbounds for element in eachelement(dg, cache)
        # Get neighboring element ids
        left = left_neighbors[1, element]
        lower = left_neighbors[2, element]

        # Apply smoothing
        alpha[left] = max(alpha_tmp[left], 0.5f0 * alpha_tmp[element], alpha[left])
        alpha[element] = max(alpha_tmp[element], 0.5f0 * alpha_tmp[left],
                             alpha[element])

        alpha[lower] = max(alpha_tmp[lower], 0.5f0 * alpha_tmp[element], alpha[lower])
        alpha[element] = max(alpha_tmp[element], 0.5f0 * alpha_tmp[lower],
                             alpha[element])
    end

    return nothing
end
end # @muladd
