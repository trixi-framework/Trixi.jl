# Based on https://hal.archives-ouvertes.fr/hal-03017566v1/document

#######################################
### Bilinaer B Spline interpolation ###
#######################################

function spline_interpolation(b_spline::BilinearBSpline, tx, ty)

  x  = b_spline.x
  y  = b_spline.y
  hx = b_spline.hx
  hy = b_spline.hy
  Q  = b_spline.Q
  IP = b_spline.IP

  i = max(1, min(searchsortedlast(x, tx), length(x) - 1))
  j = max(1, min(searchsortedlast(y, ty), length(y) - 1))

  my = (tx - x[i])/hx
  ny = (ty - y[j])/hy

  Q_temp = [Q[i, j:(j+1)] Q[(i+1), j:(j+1)]]

  S = [ny, 1]' * IP * Q_temp * IP' * [my, 1]
  
  return S
end

######################################
### Bicubic B Spline interpolation ###
######################################

function spline_interpolation(b_spline::BicubicBSpline, tx, ty)

  x  = b_spline.x
  y  = b_spline.y
  hx = b_spline.hx
  hy = b_spline.hy
  Q  = b_spline.Q
  IP = b_spline.IP

  i = max(1, min(searchsortedlast(x, tx), length(x) - 1))
  j = max(1, min(searchsortedlast(y, ty), length(y) - 1))

  my = (tx - x[i])/hx
  ny = (ty - y[j])/hy

  Q_temp = [Q[i, j:(j+3)] Q[(i+1), j:(j+3)] Q[(i+2), j:(j+3)] Q[(i+3), j:(j+3)]]

  S = 1/36 * [ny^3, ny^2, ny, 1]' * IP * Q_temp * IP' * [my^3, my^2, my, 1]
  
  return S
end