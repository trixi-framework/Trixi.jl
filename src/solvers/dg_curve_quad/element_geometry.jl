
using StaticArrays: MArray

mutable struct ElementGeometry{RealT<:Real, NNODES}
  x       ::MArray{Tuple{NNODES, NNODES}, RealT}
  y       ::MArray{Tuple{NNODES, NNODES}, RealT}
  X_xi    ::MArray{Tuple{NNODES, NNODES}, RealT}
  X_eta   ::MArray{Tuple{NNODES, NNODES}, RealT}
  Y_xi    ::MArray{Tuple{NNODES, NNODES}, RealT}
  Y_eta   ::MArray{Tuple{NNODES, NNODES}, RealT}
  Jac     ::MArray{Tuple{NNODES, NNODES}, RealT}
  invJac  ::MArray{Tuple{NNODES, NNODES}, RealT}
  x_bndy  ::MArray{Tuple{NNODES, 4}, RealT}
  y_bndy  ::MArray{Tuple{NNODES, 4}, RealT}
  n_vecs  ::MArray{Tuple{NNODES, 4, 2}, RealT}
  scaling ::MArray{Tuple{NNODES, 4}, RealT}
end


# Constructor for the element geometry of an element (curved or straight sided)
function ElementGeometry(RealT, polydeg, nodes, corners, GammaCurves::Array{GammaCurve,1}, is_curved::Bool)

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
  n_vecs  = zeros(nnodes_ , 4, 2)
  scaling = zeros(nnodes_ , 4)

# (x,y) values and Jacobian information for the volume of an element
  for j in 1:nnodes_
    for i in 1:nnodes_
      if (is_curved)
        x[i,j], y[i,j] = transfinite_quad_map(nodes[i], nodes[j], GammaCurves)
        X_xi[i,j], X_eta[i,j], Y_xi[i,j], Y_eta[i,j] = transfinite_quad_map_metrics(nodes[i], nodes[j], GammaCurves)
      else
        x[i,j], y[i,j] = straight_side_quad_map(corners, nodes[i], nodes[j])
        X_xi[i,j], X_eta[i,j], Y_xi[i,j], Y_eta[i,j] = straight_side_quad_map_metrics(corners, nodes[i], nodes[j])
      end
      Jac[i,j]    = X_xi[i,j] * Y_eta[i,j] - X_eta[i,j] * Y_xi[i,j]
      invJac[i,j] = 1.0 / Jac[i,j]
    end
  end

# normals and boundary information for the left (local side 4) and right (local side 2)
  for j in 1:nnodes_
    # side 2
    if (is_curved)
      x_bndy[j,2], y_bndy[j,2] = transfinite_quad_map(1.0, nodes[j], GammaCurves)
      Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(1.0, nodes[j], GammaCurves)
    else
      x_bndy[j,2], y_bndy[j,2] = straight_side_quad_map(corners, 1.0, nodes[j])
      Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(corners, 1.0, nodes[j])
    end
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[j,2]  = sqrt(Yeta * Yeta + Xeta * Xeta)
    n_vecs[j,2,1] = sign(Jtemp) * ( Yeta / scaling[j,2])
    n_vecs[j,2,2] = sign(Jtemp) * (-Xeta / scaling[j,2])
    # side 4
    if (is_curved)
      x_bndy[j,4], y_bndy[j,4] = transfinite_quad_map(-1.0, nodes[j], GammaCurves)
      Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(-1.0, nodes[j], GammaCurves)
    else
      x_bndy[j,4], y_bndy[j,4] = straight_side_quad_map(corners, -1.0, nodes[j])
      Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(corners, -1.0, nodes[j])
    end
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[j,4]  =  sqrt(Yeta * Yeta + Xeta * Xeta)
    n_vecs[j,4,1] = -sign(Jtemp) * ( Yeta / scaling[j,4])
    n_vecs[j,4,2] = -sign(Jtemp) * (-Xeta / scaling[j,4])
  end

#  normals and boundary information for the top (local side 3) and bottom (local side 1)
  for i in 1:nnodes_
    # side 1
    if (is_curved)
      x_bndy[i,1], y_bndy[i,1] = transfinite_quad_map(nodes[i], -1.0, GammaCurves)
      Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(nodes[i], -1.0, GammaCurves)
    else
      x_bndy[i,1], y_bndy[i,1] = straight_side_quad_map(corners, nodes[i], -1.0)
      Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(corners, nodes[i], -1.0)
    end
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[i,1]  =  sqrt(Yxi * Yxi + Xxi * Xxi)
    n_vecs[i,1,1] = -sign(Jtemp) * (-Yxi / scaling[i,1])
    n_vecs[i,1,2] = -sign(Jtemp) * ( Xxi / scaling[i,1])
    # side 3
    if (is_curved)
      x_bndy[i,3], y_bndy[i,3] = transfinite_quad_map(nodes[i], 1.0, GammaCurves)
      Xxi, Xeta, Yxi, Yeta     = transfinite_quad_map_metrics(nodes[i], 1.0, GammaCurves)
    else
      x_bndy[i,3], y_bndy[i,3] = straight_side_quad_map(corners, nodes[i], 1.0)
      Xxi, Xeta, Yxi, Yeta     = straight_side_quad_map_metrics(corners, nodes[i], 1.0)
    end
    Jtemp = Xxi * Yeta - Xeta * Yxi
    scaling[i,3]  = sqrt(Yxi * Yxi + Xxi * Xxi)
    n_vecs[i,3,1] = sign(Jtemp) * (-Yxi / scaling[i,3])
    n_vecs[i,3,2] = sign(Jtemp) * ( Xxi / scaling[i,3])
  end

  return ElementGeometry{RealT, nnodes_}( x, y, X_xi, X_eta, Y_xi, Y_eta, Jac, invJac,
                                           x_bndy, y_bndy, n_vecs, scaling )
end
