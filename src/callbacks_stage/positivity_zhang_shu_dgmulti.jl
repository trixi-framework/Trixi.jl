# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# For `DGMultiSBP` solvers, the solution nodes are the volume quadrature nodes with
# positive quadrature weights, and the face quadrature nodes are a subset of them
# (`rd.Fmask`). Thus, the cell average is a convex combination of the nodal values, and
# the fluxes are evaluated only at solution nodes, so limiting the nodal values enforces
# the bounds at all points relevant for the scheme. This does not hold for `Polynomial`
# approximation types, for which the limiter is not available.
# Since the nodes of `DGMulti` solvers are indexed linearly, one method covers all
# spatial dimensions.
function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::DGMultiMesh, equations, dg::DGMultiSBP, cache)
    @threaded for element in eachelement(mesh, dg, cache)
        # determine minimum value
        value_min = typemax(eltype(eltype(u)))
        for i in each_quad_node(mesh, dg, cache)
            value_min = min(value_min, variable(u[i, element], equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)

        # This avoids the issue when `value_mean` is slightly smaller than `threshold`
        # (e.g., due to finite precision effects in PositivityPreservingLimiterLiuZhang),
        # which results in invalid theta values smaller than 0. Note that min(1, theta)
        # is not necessary since we are only enforcing lower bounds.
        theta = max(0, theta)

        for i in each_quad_node(mesh, dg, cache)
            u[i, element] = theta * u[i, element] + (1 - theta) * u_mean
        end
    end

    return nothing
end
end # @muladd
