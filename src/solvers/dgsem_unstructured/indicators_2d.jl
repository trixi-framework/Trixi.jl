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

    @unpack element_ids = cache.interfaces
    # Explicit bounds check, which allows us to assume inbounds access below.
    # The neighbor ids are element ids taken from the mesh connectivity.
    @boundscheck begin
        check_axes(alpha, (eachelement(dg, cache),))
        check_axes(alpha_tmp, (eachelement(dg, cache),))
        check_axes(element_ids, (Base.OneTo(2), eachinterface(dg, cache)))
    end

    # Loop over interfaces
    @inbounds for interface in eachinterface(dg, cache)
        # Get neighboring element ids
        left = element_ids[1, interface]
        right = element_ids[2, interface]

        # Apply smoothing
        alpha[left] = max(alpha_tmp[left], 0.5f0 * alpha_tmp[right], alpha[left])
        alpha[right] = max(alpha_tmp[right], 0.5f0 * alpha_tmp[left], alpha[right])
    end

    return nothing
end
end # @muladd
