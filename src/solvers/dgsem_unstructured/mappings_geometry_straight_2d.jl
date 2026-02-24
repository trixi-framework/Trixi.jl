# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point (x,y)
# in physical coordinate space for a quadrilateral element with straight sides
#     Alg. 95 from the blue book of Kopriva
function straight_side_quad_map(xi, eta, corner_points)
    x = 0.25f0 * (corner_points[1, 1] * (1 - xi) * (1 - eta)
         + corner_points[2, 1] * (1 + xi) * (1 - eta)
         + corner_points[3, 1] * (1 + xi) * (1 + eta)
         + corner_points[4, 1] * (1 - xi) * (1 + eta))

    y = 0.25f0 * (corner_points[1, 2] * (1 - xi) * (1 - eta)
         + corner_points[2, 2] * (1 + xi) * (1 - eta)
         + corner_points[3, 2] * (1 + xi) * (1 + eta)
         + corner_points[4, 2] * (1 - xi) * (1 + eta))

    return x, y
end

# Compute the metric terms for the straight sided quadrilateral mapping
#     Alg. 100 from the blue book of Kopriva
function straight_side_quad_map_metrics(xi, eta, corner_points)
    X_xi = 0.25f0 * ((1 - eta) * (corner_points[2, 1] - corner_points[1, 1]) +
            (1 + eta) * (corner_points[3, 1] - corner_points[4, 1]))

    X_eta = 0.25f0 * ((1 - xi) * (corner_points[4, 1] - corner_points[1, 1]) +
             (1 + xi) * (corner_points[3, 1] - corner_points[2, 1]))

    Y_xi = 0.25f0 * ((1 - eta) * (corner_points[2, 2] - corner_points[1, 2]) +
            (1 + eta) * (corner_points[3, 2] - corner_points[4, 2]))

    Y_eta = 0.25f0 * ((1 - xi) * (corner_points[4, 2] - corner_points[1, 2]) +
             (1 + xi) * (corner_points[3, 2] - corner_points[2, 2]))

    return X_xi, X_eta, Y_xi, Y_eta
end

# construct the (x,y) node coordinates in the volume of a straight sided element
function calc_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4}, element,
                                nodes, corners)
    for j in eachindex(nodes), i in eachindex(nodes)
        node_coordinates[:, i, j, element] .= straight_side_quad_map(nodes[i], nodes[j],
                                                                     corners)
    end

    return node_coordinates
end

# construct the metric terms for a straight sided element
function calc_metric_terms!(jacobian_matrix, element, nodes, corners)

    # storage format:
    #   jacobian_matrix[1,1,:,:,:] <- X_xi
    #   jacobian_matrix[1,2,:,:,:] <- X_eta
    #   jacobian_matrix[2,1,:,:,:] <- Y_xi
    #   jacobian_matrix[2,2,:,:,:] <- Y_eta
    for j in eachindex(nodes), i in eachindex(nodes)
        (jacobian_matrix[1, 1, i, j, element],
        jacobian_matrix[1, 2, i, j, element],
        jacobian_matrix[2, 1, i, j, element],
        jacobian_matrix[2, 2, i, j, element]) = straight_side_quad_map_metrics(nodes[i],
                                                                               nodes[j],
                                                                               corners)
    end

    return jacobian_matrix
end

# construct the normal direction vectors (but not actually normalized) for a straight sided element
# normalization occurs on the fly during the surface flux computation
function calc_normal_directions!(normal_directions, element, nodes, corners)

    # normal directions on the boundary for the left (local side 4) and right (local side 2)
    for j in eachindex(nodes)
        # side 2
        X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(1, nodes[j],
                                                                  corners)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, j, 2, element] = sign(Jtemp) * (Y_eta)
        normal_directions[2, j, 2, element] = sign(Jtemp) * (-X_eta)

        # side 4
        X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(-1, nodes[j],
                                                                  corners)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, j, 4, element] = -sign(Jtemp) * (Y_eta)
        normal_directions[2, j, 4, element] = -sign(Jtemp) * (-X_eta)
    end

    # normal directions on the boundary for the top (local side 3) and bottom (local side 1)
    for i in eachindex(nodes)
        # side 1
        X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(nodes[i], -1,
                                                                  corners)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, i, 1, element] = -sign(Jtemp) * (-Y_xi)
        normal_directions[2, i, 1, element] = -sign(Jtemp) * (X_xi)

        # side 3
        X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(nodes[i], 1,
                                                                  corners)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, i, 3, element] = sign(Jtemp) * (-Y_xi)
        normal_directions[2, i, 3, element] = sign(Jtemp) * (X_xi)
    end

    return normal_directions
end
end # @muladd
