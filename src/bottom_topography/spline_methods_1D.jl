# Based on https://hal.archives-ouvertes.fr/hal-03017566v1/document

# Linear B Spline interpolation
function spline_interpolation(b_spline::LinearBSpline, t)

  x  = b_spline.x
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP


  i = max(1, min(searchsortedlast(x, t), length(x)-1))

  kappa = (t - x[i])/h

  c = [kappa, 1]' * IP * Q[i:(i+1)]

  return c  
end

# Cubic B Spline interpolation
function spline_interpolation(b_spline::CubicBSpline, t)

  x  = b_spline.x
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP


  i = max(1, min(searchsortedlast(x, t), length(x) - 1))

  kappa = (t - x[i])/h

  c = 1/6 * [kappa^3, kappa^2, kappa, 1]' * IP * Q[i:(i+3)]

  return c
end