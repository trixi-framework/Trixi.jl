# transfinite mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point
# (x,y) in physical coordinate space for a quadrilateral element with general curved sides
#     Alg. 98 from the blue book of Kopriva
function transfinite_quad_map(xi, eta, gamma_curves::AbstractVector{<:GammaCurve})

  Xref11, Xref12 = evaluate_at(-1.0, gamma_curves[1])
  Xref21, Xref22 = evaluate_at( 1.0, gamma_curves[1])
  Xref31, Xref32 = evaluate_at( 1.0, gamma_curves[3])
  Xref41, Xref42 = evaluate_at(-1.0, gamma_curves[3])

  Xcomp11, Xcomp12 = evaluate_at(xi , gamma_curves[1])
  Xcomp21, Xcomp22 = evaluate_at(eta, gamma_curves[2])
  Xcomp31, Xcomp32 = evaluate_at(xi , gamma_curves[3])
  Xcomp41, Xcomp42 = evaluate_at(eta, gamma_curves[4])

  x = ( 0.5 * (  (1.0 - xi)  * Xcomp41 + (1.0 + xi)  * Xcomp21
               + (1.0 - eta) * Xcomp11 + (1.0 + eta) * Xcomp31 )
       - 0.25 * (  (1.0 - xi) * ( (1.0 - eta) * Xref11 + (1.0 + eta) * Xref41 )
                 + (1.0 + xi) * ( (1.0 - eta) * Xref21 + (1.0 + eta) * Xref31 ) ) )

  y = ( 0.5 * (  (1.0 - xi)  * Xcomp42 + (1.0 + xi)  * Xcomp22
               + (1.0 - eta) * Xcomp12 + (1.0 + eta) * Xcomp32 )
       - 0.25 * (  (1.0 - xi) * ( (1.0 - eta) * Xref12 + (1.0 + eta) * Xref42 )
                 + (1.0 + xi) * ( (1.0 - eta) * Xref22 + (1.0 + eta) * Xref32 ) ) )

  return x, y
end


# Compute the metric terms for the general curved sided quadrilateral transfitie mapping
#     Alg. 99 from the blue book of Kopriva
function transfinite_quad_map_metrics(xi, eta, gamma_curves::AbstractVector{<:GammaCurve})

  Xref11, Xref12 = evaluate_at(-1.0, gamma_curves[1])
  Xref21, Xref22 = evaluate_at( 1.0, gamma_curves[1])
  Xref31, Xref32 = evaluate_at( 1.0, gamma_curves[3])
  Xref41, Xref42 = evaluate_at(-1.0, gamma_curves[3])

  Xcomp11, Xcomp12 = evaluate_at(xi , gamma_curves[1])
  Xcomp21, Xcomp22 = evaluate_at(eta, gamma_curves[2])
  Xcomp31, Xcomp32 = evaluate_at(xi , gamma_curves[3])
  Xcomp41, Xcomp42 = evaluate_at(eta, gamma_curves[4])

  Xpcomp11, Xpcomp12 = derivative_at(xi , gamma_curves[1])
  Xpcomp21, Xpcomp22 = derivative_at(eta, gamma_curves[2])
  Xpcomp31, Xpcomp32 = derivative_at(xi , gamma_curves[3])
  Xpcomp41, Xpcomp42 = derivative_at(eta, gamma_curves[4])

  X_xi  = ( 0.5 * (Xcomp21 - Xcomp41 + (1.0 - eta) * Xpcomp11 + (1.0 + eta) * Xpcomp31)
          -0.25 * ((1.0 - eta) * (Xref21 - Xref11) + (1.0 + eta) * (Xref31 - Xref41)) )

  X_eta = ( 0.5  * ((1.0 - xi) * Xpcomp41 + (1.0 + xi) * Xpcomp21 + Xcomp31 - Xcomp11)
           -0.25 * ((1.0 - xi) * (Xref41 - Xref11) + (1.0 + xi) * (Xref31 - Xref21)) )

  Y_xi = ( 0.5  * (Xcomp22 - Xcomp42 + (1.0 - eta) * Xpcomp12 + (1.0 + eta) * Xpcomp32)
          -0.25 * ((1.0 - eta) * (Xref22 - Xref12) + (1.0 + eta) * (Xref32 - Xref42)) )

  Y_eta = ( 0.5  * ((1.0 - xi) * Xpcomp42 + (1.0 + xi) * Xpcomp22 + Xcomp32 - Xcomp12)
           -0.25 * ((1.0 - xi) * (Xref42 - Xref12) + (1.0 + xi) * (Xref32 - Xref22)) )

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the (x,y) node coordinates in the volume of a curved sided element
function calc_node_coordinates!(node_coordinates, element, nodes,
                                gamma_curves::AbstractVector{<:GammaCurve})

  for j in eachindex(nodes), i in eachindex(nodes)
    node_coordinates[:, i, j, element] .= transfinite_quad_map(nodes[i], nodes[j], gamma_curves)
  end

  return node_coordinates
end


# construct the metric terms for a curved sided element
function calc_metric_terms!(X_xi, X_eta, Y_xi, Y_eta, element, nodes,
                            gamma_curves::AbstractVector{<:GammaCurve})

  for j in eachindex(nodes), i in eachindex(nodes)
    (X_xi[i, j, element],
     X_eta[i, j, element],
     Y_xi[i, j, element],
     Y_eta[i, j, element]) = transfinite_quad_map_metrics(nodes[i], nodes[j], gamma_curves)
  end

  return X_xi, X_eta, Y_xi, Y_eta
end


# construct the normals, their scalings, and the tangents for a curved sided element
function calc_normals_scaling_and_tangents!(normals, scaling, tangents, element, nodes,
                                            gamma_curves::AbstractVector{<:GammaCurve})

  # normals and boundary information for the left (local side 4) and right (local side 2)
  for j in eachindex(nodes)
    # side 2
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(1.0, nodes[j], gamma_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[j, 2, element]     = sqrt(Y_eta * Y_eta + X_eta * X_eta)
    normals[1, j, 2, element]  = sign(Jtemp) * ( Y_eta / scaling[j, 2, element])
    normals[2, j, 2, element]  = sign(Jtemp) * (-X_eta / scaling[j, 2, element])
    tangents[1, j, 2, element] =  normals[2, j, 2, element]
    tangents[2, j, 2, element] = -normals[1, j, 2, element]

    # side 4
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(-1.0, nodes[j], gamma_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[j, 4, element]     =  sqrt(Y_eta * Y_eta + X_eta * X_eta)
    normals[1, j, 4, element]  = -sign(Jtemp) * ( Y_eta / scaling[j, 4, element])
    normals[2, j, 4, element]  = -sign(Jtemp) * (-X_eta / scaling[j, 4, element])
    tangents[1, j, 4, element] =  normals[2, j, 4, element]
    tangents[2, j, 4, element] = -normals[1, j, 4, element]
  end

  #  normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in eachindex(nodes)
    # side 1
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(nodes[i], -1.0, gamma_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[i, 1, element]     =  sqrt(Y_xi * Y_xi + X_xi * X_xi)
    normals[1, i, 1, element]  = -sign(Jtemp) * (-Y_xi / scaling[i, 1, element])
    normals[2, i, 1, element]  = -sign(Jtemp) * ( X_xi / scaling[i, 1, element])
    tangents[1, i, 1, element] =  normals[2, i, 1, element]
    tangents[2, i, 1, element] = -normals[1, i, 1, element]

    # side 3
    X_xi, X_eta, Y_xi, Y_eta = transfinite_quad_map_metrics(nodes[i], 1.0, gamma_curves)
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    scaling[i, 3, element]     = sqrt(Y_xi * Y_xi + X_xi * X_xi)
    normals[1, i, 3, element]  = sign(Jtemp) * (-Y_xi / scaling[i, 3, element])
    normals[2, i, 3, element]  = sign(Jtemp) * ( X_xi / scaling[i, 3, element])
    tangents[1, i, 3, element] =  normals[2, i, 3, element]
    tangents[2, i, 3, element] = -normals[1, i, 3, element]
  end

  return normals, scaling, tangents
end
