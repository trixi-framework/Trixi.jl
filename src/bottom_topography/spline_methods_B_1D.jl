# Based on https://hal.archives-ouvertes.fr/hal-03017566v1/document

# Cubic B Spline interpolation
function spline_interpolation(b_spline::CubicBSpline, t)

  x  = b_spline.x
  h  = b_spline.h
  Q  = b_spline.Q
  IP = b_spline.IP


  i = max(1, min(searchsortedlast(x, t), length(x) - 1))

  chi = (t - x[i])/h

  c = 1/6 * [chi^3, chi^2, chi, 1]' * IP * Q[i:(i+3)]

  return c  
end