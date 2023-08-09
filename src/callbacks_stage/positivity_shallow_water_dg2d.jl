# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# TODO: TrixiShallowWater: 2D wet/dry limiter should move

function limiter_shallow_water!(u, threshold::Real, variable,
                                mesh::AbstractMesh{2},
                                equations::ShallowWaterEquations2D, dg::DGSEM, cache)
    @unpack weights = dg.basis

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
        u_mean = zero(get_node_vars(u, equations, dg, 1, 1, element))
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)
            u_mean += u_node * weights[i] * weights[j]
        end
        # note that the reference element is [-1,1]^ndims(dg), thus the weights sum to 2
        u_mean = u_mean / 2^ndims(mesh)

        # We compute the value directly with the mean values, as we assume that
        # Jensen's inequality holds (e.g. pressure for compressible Euler equations).
        value_mean = variable(u_mean, equations)
        theta = (value_mean - threshold) / (value_mean - value_min)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            # Cut off velocity in case that the water height is smaller than the threshold

            h_node, h_v1_node, h_v2_node, b_node = u_node
            h_mean, h_v1_mean, h_v2_mean, _ = u_mean # b_mean is not used as it must not be overwritten

            if h_node <= threshold
                h_v1_node = zero(eltype(u))
                h_v2_node = zero(eltype(u))
                h_v1_mean = zero(eltype(u))
                h_v2_mean = zero(eltype(u))
            end

            u_node = SVector(h_node, h_v1_node, h_v2_node, b_node)
            u_mean = SVector(h_mean, h_v1_mean, h_v2_mean, b_node)

            # When velocities are cut off, the only averaged value is the water height,
            # because the velocities are set to zero and this value is passed.
            # Otherwise, the velocities are averaged, as well.
            # Note that the auxiliary bottom topography variable `b` is never limited.
            set_node_vars!(u, theta * u_node + (1 - theta) * u_mean,
                           equations, dg, i, j, element)
        end
    end

    # "Safety" application of the wet/dry thresholds over all the DG nodes
    # on the current `element` after the limiting above in order to avoid dry nodes.
    # If the value_mean < threshold before applying limiter, there
    # could still be dry nodes afterwards due to logic of the limiting
    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            u_node = get_node_vars(u, equations, dg, i, j, element)

            h, h_v1, h_v2, b = u_node

            if h <= threshold
                h = threshold
                h_v1 = zero(eltype(u))
                h_v2 = zero(eltype(u))
            end

            u_node = SVector(h, h_v1, h_v2, b)

            set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end

    return nothing
end
end # @muladd
