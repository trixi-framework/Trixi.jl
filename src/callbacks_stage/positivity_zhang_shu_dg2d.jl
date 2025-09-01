# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function compute_u_mean(u::AbstractArray{<:Any, 4}, element,
                                mesh::AbstractMesh{2}, equations, weights, dg::DGSEM,
                                inverse_jacobian)
    node_volume = zero(real(mesh))
    total_volume = copy(node_volume)

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

function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        u_mean = compute_u_mean(u, element, mesh, equations, weights, dg,
                                inverse_jacobian)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, element)
        end
    end

    return nothing
end
end # @muladd
