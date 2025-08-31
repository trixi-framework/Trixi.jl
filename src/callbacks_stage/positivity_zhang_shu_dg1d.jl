# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function compute_u_mean(u::AbstractArray{<:Any, 3}, mesh::AbstractMesh{1},
                                equations, dg::DGSEM, weights,
                                element)
    u_mean = zero(get_node_vars(u, equations, dg, 1, element))
    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)
        u_mean += u_node * weights[i]
    end
    # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
    return u_mean / 2
end

function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{1}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis

    @threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        u_mean = compute_u_mean(u, mesh, equations, dg, weights,
                                element)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, element)
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, element)
        end
    end

    return nothing
end
end # @muladd
