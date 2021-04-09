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
function calc_node_coordinates!(node_coordinates, element_id, nodes, poly_deg, corners)

  nnodes = poly_deg + 1
  for j in 1:nnodes, i in 1:nnodes
    node_coordinates[:, i ,j ,element_id] .= straight_side_quad_map(nodes[i], nodes[j], corners)
  end

  return node_coordinates
end


# construct the metric terms for a straight sided element
function calc_metric_terms!(X_xi, X_eta, Y_xi, Y_eta, element_id, nodes, poly_deg, corners)

  nnodes = poly_deg + 1
  for j in 1:nnodes, i in 1:nnodes
    X_xi[i, j, element_id],
    X_eta[i, j, element_id],
    Y_xi[i, j, element_id],
    Y_eta[i, j, element_id] = straight_side_quad_map_metrics(nodes[i], nodes[j], corners)
  end

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the normals, their scalings, and the tangents for a straight sided element
function calc_normals_scaling_and_tangents!(normals, scaling, tangents, element_id, nodes, poly_deg,
                                            corners)

  nnodes = poly_deg + 1
  # normals and boundary information for the left (local side 4) and right (local side 2)
  for j in 1:nnodes
    # side 2
    Xxi, Xeta, Yxi, Yeta = straight_side_quad_map_metrics(1.0, nodes[j], corners)
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[j, 2, element_id]     = sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[1, j, 2, element_id]  = sign(Jtemp) * ( Yeta / scaling[j, 2, element_id])
    normals[2, j, 2, element_id]  = sign(Jtemp) * (-Xeta / scaling[j, 2, element_id])
    tangents[1, j, 2, element_id] =  normals[2, j, 2, element_id]
    tangents[2, j, 2, element_id] = -normals[1, j, 2, element_id]

    # side 4
    Xxi, Xeta, Yxi, Yeta = straight_side_quad_map_metrics(-1.0, nodes[j], corners)
    Jtemp =  Xxi * Yeta - Xeta * Yxi
    scaling[j, 4, element_id]     =  sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[1, j, 4, element_id]  = -sign(Jtemp) * ( Yeta / scaling[j, 4, element_id])
    normals[2, j, 4, element_id]  = -sign(Jtemp) * (-Xeta / scaling[j, 4, element_id])
    tangents[1, j, 4, element_id] =  normals[2, j, 4, element_id]
    tangents[2, j, 4, element_id] = -normals[1, j, 4, element_id]
  end

  # normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in 1:nnodes
    # side 1
    Xxi, Xeta, Yxi, Yeta = straight_side_quad_map_metrics(nodes[i], -1.0, corners)
    Jtemp =  Xxi * Yeta - Xeta * Yxi
    scaling[i, 1, element_id]     =  sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[1, i, 1, element_id]  = -sign(Jtemp) * (-Yxi / scaling[i, 1, element_id])
    normals[2, i, 1, element_id]  = -sign(Jtemp) * ( Xxi / scaling[i, 1, element_id])
    tangents[1, i, 1, element_id] =  normals[2, i, 1, element_id]
    tangents[2, i, 1, element_id] = -normals[1, i, 1, element_id]

    # side 3
    Xxi, Xeta, Yxi, Yeta = straight_side_quad_map_metrics(nodes[i], 1.0, corners)
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[i, 3, element_id]     = sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[1, i, 3, element_id]  = sign(Jtemp) * (-Yxi / scaling[i, 3, element_id])
    normals[2, i, 3, element_id]  = sign(Jtemp) * ( Xxi / scaling[i, 3, element_id])
    tangents[1, i, 3, element_id] =  normals[2, i, 3, element_id]
    tangents[2, i, 3, element_id] = -normals[1, i, 3, element_id]
  end

  return normals, scaling, tangents
end
