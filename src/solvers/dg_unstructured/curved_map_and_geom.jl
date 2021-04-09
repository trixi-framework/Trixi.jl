# transfinite mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point
# (x,y) in physical coordinate space for a quadrilateral element with general curved sides
#     Alg. 98 from the blue book of Kopriva
function transfinite_quad_map(xi, eta, GammaCurves::Array{GammaCurve,1})

  Xref  = zeros(4,2)
  Xcomp = zeros(4,2)

  Xref[1,1], Xref[1,2] = evaluate_at( -1.0, GammaCurves[1] )
  Xref[2,1], Xref[2,2] = evaluate_at(  1.0, GammaCurves[1] )
  Xref[3,1], Xref[3,2] = evaluate_at(  1.0, GammaCurves[3] )
  Xref[4,1], Xref[4,2] = evaluate_at( -1.0, GammaCurves[3] )

  Xcomp[1,1], Xcomp[1,2] = evaluate_at( xi , GammaCurves[1] )
  Xcomp[2,1], Xcomp[2,2] = evaluate_at( eta, GammaCurves[2] )
  Xcomp[3,1], Xcomp[3,2] = evaluate_at( xi , GammaCurves[3] )
  Xcomp[4,1], Xcomp[4,2] = evaluate_at( eta, GammaCurves[4] )

  x = ( 0.5 * (  (1.0 - xi)  * Xcomp[4,1] + (1.0 + xi)  * Xcomp[2,1]
               + (1.0 - eta) * Xcomp[1,1] + (1.0 + eta) * Xcomp[3,1] )
       - 0.25 * (  (1.0 - xi) * ( (1.0 - eta) * Xref[1,1] + (1.0 + eta) * Xref[4,1] )
                 + (1.0 + xi) * ( (1.0 - eta) * Xref[2,1] + (1.0 + eta) * Xref[3,1] ) ) )

  y = ( 0.5 * (  (1.0 - xi)  * Xcomp[4,2] + (1.0 + xi)  * Xcomp[2,2]
               + (1.0 - eta) * Xcomp[1,2] + (1.0 + eta) * Xcomp[3,2] )
       - 0.25 * (  (1.0 - xi) * ( (1.0 - eta) * Xref[1,2] + (1.0 + eta) * Xref[4,2] )
                 + (1.0 + xi) * ( (1.0 - eta) * Xref[2,2] + (1.0 + eta) * Xref[3,2] ) ) )

  return x, y
end


# Compute the metric terms for the general curved sided quadrilateral transfitie mapping
#     Alg. 99 from the blue book of Kopriva
function transfinite_quad_map_metrics(xi, eta, GammaCurves::Array{GammaCurve,1})

  Xref   = zeros(4,2)
  Xcomp  = zeros(4,2)
  Xpcomp = zeros(4,2)

  Xref[1,1], Xref[1,2] = evaluate_at( -1.0, GammaCurves[1] )
  Xref[2,1], Xref[2,2] = evaluate_at(  1.0, GammaCurves[1] )
  Xref[3,1], Xref[3,2] = evaluate_at(  1.0, GammaCurves[3] )
  Xref[4,1], Xref[4,2] = evaluate_at( -1.0, GammaCurves[3] )

  Xcomp[1,1], Xcomp[1,2] = evaluate_at( xi , GammaCurves[1] )
  Xcomp[2,1], Xcomp[2,2] = evaluate_at( eta, GammaCurves[2] )
  Xcomp[3,1], Xcomp[3,2] = evaluate_at( xi , GammaCurves[3] )
  Xcomp[4,1], Xcomp[4,2] = evaluate_at( eta, GammaCurves[4] )

  Xpcomp[1,1], Xpcomp[1,2] = derivative_at( xi , GammaCurves[1] )
  Xpcomp[2,1], Xpcomp[2,2] = derivative_at( eta, GammaCurves[2] )
  Xpcomp[3,1], Xpcomp[3,2] = derivative_at( xi , GammaCurves[3] )
  Xpcomp[4,1], Xpcomp[4,2] = derivative_at( eta, GammaCurves[4] )

  X_xi  = ( 0.5 * (Xcomp[2,1] - Xcomp[4,1] + (1.0 - eta) * Xpcomp[1,1] + (1.0 + eta) * Xpcomp[3,1])
          -0.25 * ((1.0 - eta) * (Xref[2,1] - Xref[1,1]) + (1.0 + eta) * (Xref[3,1] - Xref[4,1])) )

  X_eta = ( 0.5  * ((1.0 - xi) * Xpcomp[4,1] + (1.0 + xi) * Xpcomp[2,1] + Xcomp[3,1] - Xcomp[1,1])
           -0.25 * ((1.0 - xi) * (Xref[4,1] - Xref[1,1]) + (1.0 + xi) * (Xref[3,1] - Xref[2,1])) )

  Y_xi = ( 0.5  * (Xcomp[2,2] - Xcomp[4,2] + (1.0 - eta) * Xpcomp[1,2] + (1.0 + eta) * Xpcomp[3,2])
          -0.25 * ((1.0 - eta) * (Xref[2,2] - Xref[1,2]) + (1.0 + eta) * (Xref[3,2] - Xref[4,2])) )

  Y_eta = ( 0.5  * ((1.0 - xi) * Xpcomp[4,2] + (1.0 + xi) * Xpcomp[2,2] + Xcomp[3,2] - Xcomp[1,2])
           -0.25 * ((1.0 - xi) * (Xref[4,2] - Xref[1,2]) + (1.0 + xi) * (Xref[3,2] - Xref[2,2])) )

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the (x,y) node coordinates in the volume of a curved sided element
function calc_node_coordinates!(node_coordinates, element_id, nodes, poly_deg,
                                GammaCurves::Array{GammaCurve,1})

  nnodes = poly_deg + 1
  for j in 1:nnodes, i in 1:nnodes
    node_coordinates[:, i, j, element_id] .= transfinite_quad_map(nodes[i], nodes[j], GammaCurves)
  end

  return node_coordinates
end


# construct the metric terms for a curved sided element
function calc_metric_terms!(X_xi, X_eta, Y_xi, Y_eta, element_id, nodes, poly_deg,
                            GammaCurves::Array{GammaCurve,1})

  nnodes = poly_deg + 1
  for j in 1:nnodes, i in 1:nnodes
    X_xi[i, j, element_id],
    X_eta[i, j, element_id],
    Y_xi[i, j, element_id],
    Y_eta[i, j, element_id] = transfinite_quad_map_metrics(nodes[i], nodes[j], GammaCurves)
  end

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the normals, their scalings, and the tangents for a curved sided element
function calc_normals_scaling_and_tangents!(normals, scaling, tangents, element_id, nodes, poly_deg,
                                            GammaCurves::Array{GammaCurve,1})

  nnodes = poly_deg + 1
  # normals and boundary information for the left (local side 4) and right (local side 2)
  for j in 1:nnodes
    # side 2
    Xxi, Xeta, Yxi, Yeta = transfinite_quad_map_metrics(1.0, nodes[j], GammaCurves)
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[j, 2, element_id]     = sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[1, j, 2, element_id]  = sign(Jtemp) * ( Yeta / scaling[j, 2, element_id])
    normals[2, j, 2, element_id]  = sign(Jtemp) * (-Xeta / scaling[j, 2, element_id])
    tangents[1, j, 2, element_id] =  normals[2, j, 2, element_id]
    tangents[2, j, 2, element_id] = -normals[1, j, 2, element_id]

    # side 4
    Xxi, Xeta, Yxi, Yeta = transfinite_quad_map_metrics(-1.0, nodes[j], GammaCurves)
    Jtemp =  Xxi * Yeta - Xeta * Yxi
    scaling[j, 4, element_id]     =  sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[1, j, 4, element_id]  = -sign(Jtemp) * ( Yeta / scaling[j, 4, element_id])
    normals[2, j, 4, element_id]  = -sign(Jtemp) * (-Xeta / scaling[j, 4, element_id])
    tangents[1, j, 4, element_id] =  normals[2, j, 4, element_id]
    tangents[2, j, 4, element_id] = -normals[1, j, 4, element_id]
  end

  #  normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in 1:nnodes
    # side 1
    Xxi, Xeta, Yxi, Yeta = transfinite_quad_map_metrics(nodes[i], -1.0, GammaCurves)
    Jtemp =  Xxi * Yeta - Xeta * Yxi
    scaling[i, 1, element_id]     =  sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[1, i, 1, element_id]  = -sign(Jtemp) * (-Yxi / scaling[i, 1, element_id])
    normals[2, i, 1, element_id]  = -sign(Jtemp) * ( Xxi / scaling[i, 1, element_id])
    tangents[1, i, 1, element_id] =  normals[2, i, 1, element_id]
    tangents[2, i, 1, element_id] = -normals[1, i, 1, element_id]

    # side 3
    Xxi, Xeta, Yxi, Yeta = transfinite_quad_map_metrics(nodes[i], 1.0, GammaCurves)
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[i, 3, element_id]     = sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[1, i, 3, element_id]  = sign(Jtemp) * (-Yxi / scaling[i, 3, element_id])
    normals[2, i, 3, element_id]  = sign(Jtemp) * ( Xxi / scaling[i, 3, element_id])
    tangents[1, i, 3, element_id] =  normals[2, i, 3, element_id]
    tangents[2, i, 3, element_id] = -normals[1, i, 3, element_id]
  end

  return normals, scaling, tangents
end
