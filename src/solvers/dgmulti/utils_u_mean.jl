# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# `compute_u_mean` is used in the stage callback `PositivityPreservingLimiterZhangShu`
# (and, via its local limiter, in `PositivityPreservingLimiterLiuZhang`).
# The methods are restricted to `DGMultiSBP`, i.e., to nodal solvers whose solution
# nodes coincide with the volume quadrature nodes. For `Polynomial` approximation types,
# the nodal values are not quadrature values, so a weighted nodal mean is not the cell
# average.

# `mesh.md.wJq` contains the volume quadrature weights multiplied by the Jacobian at the
# volume quadrature nodes, which are the solution nodes for `DGMultiSBP`. This also
# covers non-affine meshes, where the Jacobian varies within an element.
@inline function compute_u_mean(u, element, mesh::DGMultiMesh, equations,
                                dg::DGMultiSBP, cache)
    (; wJq) = mesh.md

    u_mean = zero(u[1, element])
    total_volume = zero(eltype(wJq))
    for i in each_quad_node(mesh, dg, cache)
        wJq_node = wJq[i, element]
        total_volume += wJq_node
        # explicit a = a + b * c instead of a += b * c to enable @muladd
        u_mean = u_mean + u[i, element] * wJq_node
    end
    return u_mean / total_volume # normalize with the total volume
end
end # @muladd
