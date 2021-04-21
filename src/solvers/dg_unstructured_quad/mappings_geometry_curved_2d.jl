# transfinite mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point
# (x,y) in physical coordinate space for a quadrilateral element with general curved sides
#     Alg. 98 from the blue book of Kopriva
function transfinite_quad_map(xi, eta, surface_curves::AbstractVector{<:CurvedSurface})

  # evaluate the gamma curves to get the four corner points of the element
  x_corner1, y_corner1 = evaluate_at(-1.0, surface_curves[1])
  x_corner2, y_corner2 = evaluate_at( 1.0, surface_curves[1])
  x_corner3, y_corner3 = evaluate_at( 1.0, surface_curves[3])
  x_corner4, y_corner4 = evaluate_at(-1.0, surface_curves[3])

  # evaluate along the gamma curves at a particular point (ξ, η) in computational space to get
  # the value (x,y) in physical space
  x1, y1 = evaluate_at(xi , surface_curves[1])
  x2, y2 = evaluate_at(eta, surface_curves[2])
  x3, y3 = evaluate_at(xi , surface_curves[3])
  x4, y4 = evaluate_at(eta, surface_curves[4])

  x = ( 0.5 * (  (1.0 - xi)  * x4 + (1.0 + xi)  * x2 + (1.0 - eta) * x1 + (1.0 + eta) * x3 )
       - 0.25 * (  (1.0 - xi) * ( (1.0 - eta) * x_corner1 + (1.0 + eta) * x_corner4 )
                 + (1.0 + xi) * ( (1.0 - eta) * x_corner2 + (1.0 + eta) * x_corner3 ) ) )

  y = ( 0.5 * (  (1.0 - xi)  * y4 + (1.0 + xi)  * y2 + (1.0 - eta) * y1 + (1.0 + eta) * y3 )
       - 0.25 * (  (1.0 - xi) * ( (1.0 - eta) * y_corner1 + (1.0 + eta) * y_corner4 )
                 + (1.0 + xi) * ( (1.0 - eta) * y_corner2 + (1.0 + eta) * y_corner3 ) ) )

  return x, y
end


# Compute the metric terms for the general curved sided quadrilateral transfitie mapping
#     Alg. 99 from the blue book of Kopriva
function transfinite_quad_map_metrics(xi, eta, surface_curves::AbstractVector{<:CurvedSurface})

  # evaluate the gamma curves to get the four corner points of the element
  x_corner1, y_corner1 = evaluate_at(-1.0, surface_curves[1])
  x_corner2, y_corner2 = evaluate_at( 1.0, surface_curves[1])
  x_corner3, y_corner3 = evaluate_at( 1.0, surface_curves[3])
  x_corner4, y_corner4 = evaluate_at(-1.0, surface_curves[3])

  # evaluate along the gamma curves at a particular point (ξ, η) in computational space to get
  # the value (x,y) in physical space
  x1, y1 = evaluate_at(xi , surface_curves[1])
  x2, y2 = evaluate_at(eta, surface_curves[2])
  x3, y3 = evaluate_at(xi , surface_curves[3])
  x4, y4 = evaluate_at(eta, surface_curves[4])

  # evaluate along the derivative of the gamma curves at a particular point (ξ, η) in
  # computational space to get the value (x_prime,y_prime) in physical space
  x1_prime, y1_prime = derivative_at(xi , surface_curves[1])
  x2_prime, y2_prime = derivative_at(eta, surface_curves[2])
  x3_prime, y3_prime = derivative_at(xi , surface_curves[3])
  x4_prime, y4_prime = derivative_at(eta, surface_curves[4])

  X_xi  = ( 0.5 * (x2 - x4 + (1.0 - eta) * x1_prime + (1.0 + eta) * x3_prime)
          -0.25 * ((1.0 - eta) * (x_corner2 - x_corner1) + (1.0 + eta) * (x_corner3 - x_corner4)) )

  X_eta = ( 0.5  * ((1.0 - xi) * x4_prime + (1.0 + xi) * x2_prime + x3 - x1)
           -0.25 * ((1.0 - xi) * (x_corner4 - x_corner1) + (1.0 + xi) * (x_corner3 - x_corner2)) )

  Y_xi = ( 0.5  * (y2 - y4 + (1.0 - eta) * y1_prime + (1.0 + eta) * y3_prime)
          -0.25 * ((1.0 - eta) * (y_corner2 - y_corner1) + (1.0 + eta) * (y_corner3 - y_corner4)) )

  Y_eta = ( 0.5  * ((1.0 - xi) * y4_prime + (1.0 + xi) * y2_prime + y3 - y1)
           -0.25 * ((1.0 - xi) * (y_corner4 - y_corner1) + (1.0 + xi) * (y_corner3 - y_corner2)) )

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the (x,y) node coordinates in the volume of a curved sided element
function calc_node_coordinates!(node_coordinates, element, nodes,
                                surface_curves::AbstractVector{<:CurvedSurface})

  for j in eachindex(nodes), i in eachindex(nodes)
    node_coordinates[:, i, j, element] .= transfinite_quad_map(nodes[i], nodes[j], surface_curves)
  end

  return node_coordinates
end


# construct the metric terms for a curved sided element
function calc_metric_terms!(X_xi, X_eta, Y_xi, Y_eta, element, nodes,
                            surface_curves::AbstractVector{<:CurvedSurface})

  for j in eachindex(nodes), i in eachindex(nodes)
    (X_xi[i, j, element],
     X_eta[i, j, element],
     Y_xi[i, j, element],
     Y_eta[i, j, element]) = transfinite_quad_map_metrics(nodes[i], nodes[j], surface_curves)
  end

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the normals, and their normalization scalings for a curved sided element
function calc_normals_and_scaling!(normals, scaling, element, nodes,
                                   surface_curves::AbstractVector{<:CurvedSurface})

  # normals and boundary information for the left (local side 4) and right (local side 2)
  for j in eachindex(nodes)
    # side 2
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(1.0, nodes[j], surface_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[j, 2, element]    = sqrt(Y_eta^2 + X_eta^2)
    normals[1, j, 2, element] = sign(Jtemp) * ( Y_eta / scaling[j, 2, element])
    normals[2, j, 2, element] = sign(Jtemp) * (-X_eta / scaling[j, 2, element])

    # side 4
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(-1.0, nodes[j], surface_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[j, 4, element]    =  sqrt(Y_eta^2 + X_eta^2)
    normals[1, j, 4, element] = -sign(Jtemp) * ( Y_eta / scaling[j, 4, element])
    normals[2, j, 4, element] = -sign(Jtemp) * (-X_eta / scaling[j, 4, element])
  end

  #  normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in eachindex(nodes)
    # side 1
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(nodes[i], -1.0, surface_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[i, 1, element]    =  sqrt(Y_xi^2 + X_xi^2)
    normals[1, i, 1, element] = -sign(Jtemp) * (-Y_xi / scaling[i, 1, element])
    normals[2, i, 1, element] = -sign(Jtemp) * ( X_xi / scaling[i, 1, element])

    # side 3
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(nodes[i], 1.0, surface_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[i, 3, element]    = sqrt(Y_xi^2 + X_xi^2)
    normals[1, i, 3, element] = sign(Jtemp) * (-Y_xi / scaling[i, 3, element])
    normals[2, i, 3, element] = sign(Jtemp) * ( X_xi / scaling[i, 3, element])
  end

  return normals, scaling
end
