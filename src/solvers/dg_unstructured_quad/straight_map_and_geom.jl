# mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point (x,y)
# in physical coordinate space for a quadrilateral element with straight sides
#     Alg. 95 from the blue book of Kopriva
function straight_side_quad_map(xi, eta, corner_points)

  x = (0.25 * (  corner_points[1,1] * (1.0 - xi) * (1.0 - eta)
               + corner_points[2,1] * (1.0 + xi) * (1.0 - eta)
               + corner_points[3,1] * (1.0 + xi) * (1.0 + eta)
               + corner_points[4,1] * (1.0 - xi) * (1.0 + eta)) )

  y = (0.25 * (  corner_points[1,2] * (1.0 - xi) * (1.0 - eta)
               + corner_points[2,2] * (1.0 + xi) * (1.0 - eta)
               + corner_points[3,2] * (1.0 + xi) * (1.0 + eta)
               + corner_points[4,2] * (1.0 - xi) * (1.0 + eta)) )

  return x, y
end


# Compute the metric terms for the straight sided quadrilateral mapping
#     Alg. 100 from the blue book of Kopriva
function straight_side_quad_map_metrics(xi, eta, corner_points)

  X_xi  = ( 0.25 * (  (1.0 - eta) * (corner_points[2,1] - corner_points[1,1])
                    + (1.0 + eta) * (corner_points[3,1] - corner_points[4,1])) )

  X_eta = ( 0.25 * (  (1.0 - xi) * (corner_points[4,1] - corner_points[1,1])
                    + (1.0 + xi) * (corner_points[3,1] - corner_points[2,1])) )

  Y_xi  = ( 0.25 * (  (1.0 - eta) * (corner_points[2,2] - corner_points[1,2])
                    + (1.0 + eta) * (corner_points[3,2] - corner_points[4,2])) )

  Y_eta = ( 0.25 * (  (1.0 - xi) * (corner_points[4,2] - corner_points[1,2])
                    + (1.0 + xi) * (corner_points[3,2] - corner_points[2,2])) )

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the (x,y) node coordinates in the volume of a straight sided element
function calc_node_coordinates!(node_coordinates, element, nodes, corners)

  for j in eachindex(nodes), i in eachindex(nodes)
    node_coordinates[:, i ,j ,element] .= straight_side_quad_map(nodes[i], nodes[j], corners)
  end

  return node_coordinates
end


# construct the metric terms for a straight sided element
function calc_metric_terms!(X_xi, X_eta, Y_xi, Y_eta, element, nodes, corners)

  for j in eachindex(nodes), i in eachindex(nodes)
    (X_xi[i, j, element],
     X_eta[i, j, element],
     Y_xi[i, j, element],
     Y_eta[i, j, element]) = straight_side_quad_map_metrics(nodes[i], nodes[j], corners)
  end

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the normals, their normailzation scalings for a straight sided element
function calc_normals_and_scaling!(normals, scaling, element, nodes, corners)

  # normals and boundary information for the left (local side 4) and right (local side 2)
  for j in eachindex(nodes)
    # side 2
    X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(1.0, nodes[j], corners)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[j, 2, element]     = sqrt(Y_eta * Y_eta + X_eta * X_eta)
    normals[1, j, 2, element]  = sign(Jtemp) * ( Y_eta / scaling[j, 2, element])
    normals[2, j, 2, element]  = sign(Jtemp) * (-X_eta / scaling[j, 2, element])

    # side 4
    X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(-1.0, nodes[j], corners)
    Jtemp =  X_xi * Y_eta - X_eta * Y_xi
    scaling[j, 4, element]     =  sqrt(Y_eta * Y_eta + X_eta * X_eta)
    normals[1, j, 4, element]  = -sign(Jtemp) * ( Y_eta / scaling[j, 4, element])
    normals[2, j, 4, element]  = -sign(Jtemp) * (-X_eta / scaling[j, 4, element])
  end

  # normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in eachindex(nodes)
    # side 1
    X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(nodes[i], -1.0, corners)
    Jtemp =  X_xi * Y_eta - X_eta * Y_xi
    scaling[i, 1, element]     =  sqrt(Y_xi * Y_xi + X_xi * X_xi)
    normals[1, i, 1, element]  = -sign(Jtemp) * (-Y_xi / scaling[i, 1, element])
    normals[2, i, 1, element]  = -sign(Jtemp) * ( X_xi / scaling[i, 1, element])

    # side 3
    X_xi, X_eta, Y_xi, Y_eta = straight_side_quad_map_metrics(nodes[i], 1.0, corners)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[i, 3, element]     = sqrt(Y_xi * Y_xi + X_xi * X_xi)
    normals[1, i, 3, element]  = sign(Jtemp) * (-Y_xi / scaling[i, 3, element])
    normals[2, i, 3, element]  = sign(Jtemp) * ( X_xi / scaling[i, 3, element])
  end

  return normals, scaling
end
