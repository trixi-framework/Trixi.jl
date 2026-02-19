# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# `compute_u_mean` used in:
# (Stage-) Callbacks `EntropyBoundedLimiter` and `PositivityPreservingLimiterZhangShu`

# positional arguments `mesh` and `cache` passed in to match signature of 2D/3D functions
@inline function compute_u_mean(u::AbstractArray{<:Any, 3}, element,
                                mesh::AbstractMesh{1}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis

    u_mean = zero(get_node_vars(u, equations, dg, 1, element))
    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)
        u_mean += u_node * weights[i]
    end
    # normalize with the total volume
    # note that the reference element is [-1,1], thus the weights sum to 2
    return 0.5f0 * u_mean
end

@inline function compute_u_mean(u::AbstractArray{<:Any, 4}, element,
                                mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    node_volume = zero(real(mesh))
    total_volume = zero(node_volume)

    u_mean = zero(get_node_vars(u, equations, dg, 1, 1, element))
    for j in eachnode(dg), i in eachnode(dg)
        volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian, mesh,
                                                       i, j, element)))
        node_volume = weights[i] * weights[j] * volume_jacobian
        total_volume += node_volume

        u_node = get_node_vars(u, equations, dg, i, j, element)
        u_mean += u_node * node_volume
    end
    return u_mean / total_volume # normalize with the total volume
end

@inline function compute_u_mean(u::AbstractArray{<:Any, 5}, element,
                                mesh::AbstractMesh{3}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    node_volume = zero(real(mesh))
    total_volume = zero(node_volume)

    u_mean = zero(get_node_vars(u, equations, dg, 1, 1, 1, element))
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian, mesh,
                                                       i, j, k, element)))
        node_volume = weights[i] * weights[j] * weights[k] * volume_jacobian
        total_volume += node_volume

        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        u_mean += u_node * node_volume
    end
    return u_mean / total_volume # normalize with the total volume
end
end # @muladd
