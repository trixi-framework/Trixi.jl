
using StaticArrays: SArray

struct ElementGeometry{RealT<:Real, NNODES}
  x       ::SArray{Tuple{NNODES, NNODES}, RealT}
  y       ::SArray{Tuple{NNODES, NNODES}, RealT}
  X_xi    ::SArray{Tuple{NNODES, NNODES}, RealT}
  X_eta   ::SArray{Tuple{NNODES, NNODES}, RealT}
  Y_xi    ::SArray{Tuple{NNODES, NNODES}, RealT}
  Y_eta   ::SArray{Tuple{NNODES, NNODES}, RealT}
  Jac     ::SArray{Tuple{NNODES, NNODES}, RealT}
  invJac  ::SArray{Tuple{NNODES, NNODES}, RealT}
  x_bndy  ::SArray{Tuple{NNODES, 4}, RealT}
  y_bndy  ::SArray{Tuple{NNODES, 4}, RealT}
  normals ::SArray{Tuple{NNODES, 4, 2}, RealT}
  scaling ::SArray{Tuple{NNODES, 4}, RealT}
end


# Constructor for the element geometry of a curved element
function ElementGeometry(RealT, polydeg, nodes, GammaCurves::Array{GammaCurve,1})

  nnodes_ = polydeg + 1
  x       = zeros(nnodes_ , nnodes_)
  y       = zeros(nnodes_ , nnodes_)
  X_xi    = zeros(nnodes_ , nnodes_)
  X_eta   = zeros(nnodes_ , nnodes_)
  Y_xi    = zeros(nnodes_ , nnodes_)
  Y_eta   = zeros(nnodes_ , nnodes_)
  Jac     = zeros(nnodes_ , nnodes_)
  invJac  = zeros(nnodes_ , nnodes_)
  x_bndy  = zeros(nnodes_ , 4)
  y_bndy  = zeros(nnodes_ , 4)
  normals = zeros(nnodes_ , 4, 2)
  scaling = zeros(nnodes_ , 4)

# (x,y) values and Jacobian information for the volume of an element
  for j in 1:nnodes_
    for i in 1:nnodes_
      x[i,j], y[i,j]         = transfinite_quad_map(nodes[i], nodes[j], GammaCurves)
      X_xi[i,j], X_eta[i,j],
      Y_xi[i,j], Y_eta[i,j]  = transfinite_quad_map_metrics(nodes[i], nodes[j], GammaCurves)

      Jac[i,j]    = X_xi[i,j] * Y_eta[i,j] - X_eta[i,j] * Y_xi[i,j]
      invJac[i,j] = 1.0 / Jac[i,j]
    end
  end

# normals and boundary information for the left (local side 4) and right (local side 2)
  for j in 1:nnodes_
    # side 2
    x_bndy[j,2], y_bndy[j,2] = transfinite_quad_map(1.0, nodes[j], GammaCurves)
    Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(1.0, nodes[j], GammaCurves)

    Jtemp          = Xxi * Yeta - Xeta * Yxi
    scaling[j,2]   = sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[j,2,1] = sign(Jtemp) * ( Yeta / scaling[j,2])
    normals[j,2,2] = sign(Jtemp) * (-Xeta / scaling[j,2])
    # side 4
    x_bndy[j,4], y_bndy[j,4] = transfinite_quad_map(-1.0, nodes[j], GammaCurves)
    Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(-1.0, nodes[j], GammaCurves)

    Jtemp          =  Xxi * Yeta - Xeta * Yxi
    scaling[j,4]   =  sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[j,4,1] = -sign(Jtemp) * ( Yeta / scaling[j,4])
    normals[j,4,2] = -sign(Jtemp) * (-Xeta / scaling[j,4])
  end

#  normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in 1:nnodes_
    # side 1
    x_bndy[i,1], y_bndy[i,1] = transfinite_quad_map(nodes[i], -1.0, GammaCurves)
    Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(nodes[i], -1.0, GammaCurves)

    Jtemp          =  Xxi * Yeta - Xeta * Yxi
    scaling[i,1]   =  sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[i,1,1] = -sign(Jtemp) * (-Yxi / scaling[i,1])
    normals[i,1,2] = -sign(Jtemp) * ( Xxi / scaling[i,1])
    # side 3
    x_bndy[i,3], y_bndy[i,3] = transfinite_quad_map(nodes[i], 1.0, GammaCurves)
    Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(nodes[i], 1.0, GammaCurves)

    Jtemp          = Xxi * Yeta - Xeta * Yxi
    scaling[i,3]   = sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[i,3,1] = sign(Jtemp) * (-Yxi / scaling[i,3])
    normals[i,3,2] = sign(Jtemp) * ( Xxi / scaling[i,3])
  end

  return ElementGeometry{RealT, nnodes_}( x, y, X_xi, X_eta, Y_xi, Y_eta, Jac, invJac,
                                           x_bndy, y_bndy, normals, scaling )
end


# Constructor for the element geometry of a straight-sides element
function ElementGeometry(RealT, polydeg, nodes, corners)

  nnodes_ = polydeg + 1
  x       = zeros(nnodes_ , nnodes_)
  y       = zeros(nnodes_ , nnodes_)
  X_xi    = zeros(nnodes_ , nnodes_)
  X_eta   = zeros(nnodes_ , nnodes_)
  Y_xi    = zeros(nnodes_ , nnodes_)
  Y_eta   = zeros(nnodes_ , nnodes_)
  Jac     = zeros(nnodes_ , nnodes_)
  invJac  = zeros(nnodes_ , nnodes_)
  x_bndy  = zeros(nnodes_ , 4)
  y_bndy  = zeros(nnodes_ , 4)
  normals = zeros(nnodes_ , 4, 2)
  scaling = zeros(nnodes_ , 4)

# (x,y) values and Jacobian information for the volume of an element
  for j in 1:nnodes_
    for i in 1:nnodes_
      x[i,j], y[i,j]         = straight_side_quad_map(nodes[i], nodes[j], corners)
      X_xi[i,j], X_eta[i,j],
      Y_xi[i,j], Y_eta[i,j]  = straight_side_quad_map_metrics(nodes[i], nodes[j], corners)

      Jac[i,j]    = X_xi[i,j] * Y_eta[i,j] - X_eta[i,j] * Y_xi[i,j]
      invJac[i,j] = 1.0 / Jac[i,j]
    end
  end

# normals and boundary information for the left (local side 4) and right (local side 2)
  for j in 1:nnodes_
    # side 2
    x_bndy[j,2], y_bndy[j,2] = straight_side_quad_map(1.0, nodes[j], corners)
    Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(1.0, nodes[j], corners)

    Jtemp          = Xxi * Yeta - Xeta * Yxi
    scaling[j,2]   = sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[j,2,1] = sign(Jtemp) * ( Yeta / scaling[j,2])
    normals[j,2,2] = sign(Jtemp) * (-Xeta / scaling[j,2])
    # side 4
    x_bndy[j,4], y_bndy[j,4] = straight_side_quad_map(-1.0, nodes[j], corners)
    Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(-1.0, nodes[j], corners)

    Jtemp          =  Xxi * Yeta - Xeta * Yxi
    scaling[j,4]   =  sqrt(Yeta * Yeta + Xeta * Xeta)
    normals[j,4,1] = -sign(Jtemp) * ( Yeta / scaling[j,4])
    normals[j,4,2] = -sign(Jtemp) * (-Xeta / scaling[j,4])
  end

#  normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in 1:nnodes_
    # side 1
    x_bndy[i,1], y_bndy[i,1] = straight_side_quad_map(nodes[i], -1.0, corners)
    Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(nodes[i], -1.0, corners)

    Jtemp          =  Xxi * Yeta - Xeta * Yxi
    scaling[i,1]   =  sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[i,1,1] = -sign(Jtemp) * (-Yxi / scaling[i,1])
    normals[i,1,2] = -sign(Jtemp) * ( Xxi / scaling[i,1])
    # side 3
    x_bndy[i,3], y_bndy[i,3] = straight_side_quad_map(nodes[i], 1.0, corners)
    Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(nodes[i], 1.0, corners)

    Jtemp          = Xxi * Yeta - Xeta * Yxi
    scaling[i,3]   = sqrt(Yxi * Yxi + Xxi * Xxi)
    normals[i,3,1] = sign(Jtemp) * (-Yxi / scaling[i,3])
    normals[i,3,2] = sign(Jtemp) * ( Xxi / scaling[i,3])
  end

  return ElementGeometry{RealT, nnodes_}( x, y, X_xi, X_eta, Y_xi, Y_eta, Jac, invJac,
                                           x_bndy, y_bndy, normals, scaling )
end
