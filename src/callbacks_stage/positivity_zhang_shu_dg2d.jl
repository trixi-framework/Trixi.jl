# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function limiter_zhang_shu!(u, threshold::Real, variable,
                            mesh::AbstractMesh{2}, equations, dg::DGSEM, cache)
    @threaded for element in eachelement(dg, cache)
        # determine minimum value
        value_min = typemax(eltype(u))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            value_min = min(value_min, variable(u_node, equations))
        end

        # detect if limiting is necessary
        value_min < threshold || continue

        # compute mean value
        u_mean = mean_value_element(u, element, mesh, equations, dg, cache)

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

function mean_value_element(u, element, mesh::TreeMesh{2}, equations, dg::DGSEM, cache)
    @unpack weights = dg.basis

    u_mean = zero(get_node_vars(u, equations, dg, 1, 1, element))
    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        u_mean += u_node * weights[i] * weights[j]
    end

    # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
    return u_mean / 2^ndims(mesh)
end

function mean_value_element(u, element,
                            mesh::Union{StructuredMesh{2}, StructuredMeshView{2},
                                        UnstructuredMesh2D, P4estMesh{2},
                                        T8codeMesh{2}},
                            equations, dg::DGSEM, cache)
    @unpack weights = dg.basis

    u_mean = zero(get_node_vars(u, equations, dg, 1, 1, element))
    total_volume = zero(real(mesh))
    for j in eachnode(dg), i in eachnode(dg)
        jacobian_node = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
        u_node = get_node_vars(u, equations, dg, i, j, element)
        u_mean += u_node * weights[i] * weights[j] * jacobian_node
        total_volume += weights[i] * weights[j] * jacobian_node
    end

    # normalize with the total volume
    return u_mean / total_volume
end
end # @muladd
