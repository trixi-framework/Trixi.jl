
# mapping formula from a point (xi, eta) in reference space [-1,1]^2 to a point (x,y)
# in physical coordinate space for a quadrilateral element with straight sides
#     Alg. 95 from the blue book of Kopriva
function straight_side_quad_map(corner_points, xi, eta)

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
function straight_side_quad_map_metrics(corner_points, xi, eta)

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
