# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# transfinite mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point
# (x,y) in physical coordinate space for a quadrilateral element with general curved sides
#     Alg. 98 from the blue book of Kopriva
function transfinite_quad_map(xi, eta, surface_curves::AbstractVector{<:CurvedSurface})

    # evaluate the gamma curves to get the four corner points of the element
    x_corner1, y_corner1 = evaluate_at(-1, surface_curves[1])
    x_corner2, y_corner2 = evaluate_at(1, surface_curves[1])
    x_corner3, y_corner3 = evaluate_at(1, surface_curves[3])
    x_corner4, y_corner4 = evaluate_at(-1, surface_curves[3])

    # evaluate along the gamma curves at a particular point (ξ, η) in computational space to get
    # the value (x,y) in physical space
    x1, y1 = evaluate_at(xi, surface_curves[1])
    x2, y2 = evaluate_at(eta, surface_curves[2])
    x3, y3 = evaluate_at(xi, surface_curves[3])
    x4, y4 = evaluate_at(eta, surface_curves[4])

    x = (0.5f0 * ((1 - xi) * x4 + (1 + xi) * x2 + (1 - eta) * x1 + (1 + eta) * x3)
         -
         0.25f0 * ((1 - xi) * ((1 - eta) * x_corner1 + (1 + eta) * x_corner4) +
          (1 + xi) * ((1 - eta) * x_corner2 + (1 + eta) * x_corner3)))

    y = (0.5f0 * ((1 - xi) * y4 + (1 + xi) * y2 + (1 - eta) * y1 + (1 + eta) * y3)
         -
         0.25f0 * ((1 - xi) * ((1 - eta) * y_corner1 + (1 + eta) * y_corner4) +
          (1 + xi) * ((1 - eta) * y_corner2 + (1 + eta) * y_corner3)))

    return x, y
end

# Compute the metric terms for the general curved sided quadrilateral transfitie mapping
#     Alg. 99 from the blue book of Kopriva
function transfinite_quad_map_metrics(xi, eta,
                                      surface_curves::AbstractVector{<:CurvedSurface})

    # evaluate the gamma curves to get the four corner points of the element
    x_corner1, y_corner1 = evaluate_at(-1, surface_curves[1])
    x_corner2, y_corner2 = evaluate_at(1, surface_curves[1])
    x_corner3, y_corner3 = evaluate_at(1, surface_curves[3])
    x_corner4, y_corner4 = evaluate_at(-1, surface_curves[3])

    # evaluate along the gamma curves at a particular point (ξ, η) in computational space to get
    # the value (x,y) in physical space
    x1, y1 = evaluate_at(xi, surface_curves[1])
    x2, y2 = evaluate_at(eta, surface_curves[2])
    x3, y3 = evaluate_at(xi, surface_curves[3])
    x4, y4 = evaluate_at(eta, surface_curves[4])

    # evaluate along the derivative of the gamma curves at a particular point (ξ, η) in
    # computational space to get the value (x_prime,y_prime) in physical space
    x1_prime, y1_prime = derivative_at(xi, surface_curves[1])
    x2_prime, y2_prime = derivative_at(eta, surface_curves[2])
    x3_prime, y3_prime = derivative_at(xi, surface_curves[3])
    x4_prime, y4_prime = derivative_at(eta, surface_curves[4])

    X_xi = (0.5f0 * (x2 - x4 + (1 - eta) * x1_prime + (1 + eta) * x3_prime)
            -
            0.25f0 * ((1 - eta) * (x_corner2 - x_corner1) +
             (1 + eta) * (x_corner3 - x_corner4)))

    X_eta = (0.5f0 * ((1 - xi) * x4_prime + (1 + xi) * x2_prime + x3 - x1)
             -
             0.25f0 * ((1 - xi) * (x_corner4 - x_corner1) +
              (1 + xi) * (x_corner3 - x_corner2)))

    Y_xi = (0.5f0 * (y2 - y4 + (1 - eta) * y1_prime + (1 + eta) * y3_prime)
            -
            0.25f0 * ((1 - eta) * (y_corner2 - y_corner1) +
             (1 + eta) * (y_corner3 - y_corner4)))

    Y_eta = (0.5f0 * ((1 - xi) * y4_prime + (1 + xi) * y2_prime + y3 - y1)
             -
             0.25f0 * ((1 - xi) * (y_corner4 - y_corner1) +
              (1 + xi) * (y_corner3 - y_corner2)))

    return X_xi, X_eta, Y_xi, Y_eta
end

# construct the (x,y) node coordinates in the volume of a curved sided element
function calc_node_coordinates!(node_coordinates::AbstractArray{<:Any, 4}, element,
                                nodes,
                                surface_curves::AbstractVector{<:CurvedSurface})
    for j in eachindex(nodes), i in eachindex(nodes)
        node_coordinates[:, i, j, element] .= transfinite_quad_map(nodes[i], nodes[j],
                                                                   surface_curves)
    end

    return node_coordinates
end

# construct the metric terms for a curved sided element
function calc_metric_terms!(jacobian_matrix, element, nodes,
                            surface_curves::AbstractVector{<:CurvedSurface})

    # storage format:
    #   jacobian_matrix[1,1,:,:,:] <- X_xi
    #   jacobian_matrix[1,2,:,:,:] <- X_eta
    #   jacobian_matrix[2,1,:,:,:] <- Y_xi
    #   jacobian_matrix[2,2,:,:,:] <- Y_eta
    for j in eachindex(nodes), i in eachindex(nodes)
        (jacobian_matrix[1, 1, i, j, element],
        jacobian_matrix[1, 2, i, j, element],
        jacobian_matrix[2, 1, i, j, element],
        jacobian_matrix[2, 2, i, j, element]) = transfinite_quad_map_metrics(nodes[i],
                                                                             nodes[j],
                                                                             surface_curves)
    end

    return jacobian_matrix
end

# construct the normal direction vectors (but not actually normalized) for a curved sided element
# normalization occurs on the fly during the surface flux computation
function calc_normal_directions!(normal_directions, element, nodes,
                                 surface_curves::AbstractVector{<:CurvedSurface})

    # normal directions on the boundary for the left (local side 4) and right (local side 2)
    for j in eachindex(nodes)
        # side 2
        X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(1, nodes[j],
                                                                surface_curves)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, j, 2, element] = sign(Jtemp) * (Y_eta)
        normal_directions[2, j, 2, element] = sign(Jtemp) * (-X_eta)

        # side 4
        X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(-1, nodes[j],
                                                                surface_curves)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, j, 4, element] = -sign(Jtemp) * (Y_eta)
        normal_directions[2, j, 4, element] = -sign(Jtemp) * (-X_eta)
    end

    # normal directions on the boundary for the top (local side 3) and bottom (local side 1)
    for i in eachindex(nodes)
        # side 1
        X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(nodes[i], -1,
                                                                surface_curves)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, i, 1, element] = -sign(Jtemp) * (-Y_xi)
        normal_directions[2, i, 1, element] = -sign(Jtemp) * (X_xi)

        # side 3
        X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(nodes[i], 1,
                                                                surface_curves)
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, i, 3, element] = sign(Jtemp) * (-Y_xi)
        normal_directions[2, i, 3, element] = sign(Jtemp) * (X_xi)
    end

    return normal_directions
end
end # @muladd
