# Based on https://www.rajgunesh.com/resources/downloads/numerical/cubicsplineinterpol.pdf

# Liner spline interpolation
function spline_interpolation(spline::LinSpline, t)
    
  x = spline.x
  y = spline.y
  h = spline.h

  # Get index
  i = max(1, min(searchsortedlast(x, t), length(x)-1))

  # Calculate coefficients
  ai = (y[i+1]-y[i])/(h[i])
  bi =  y[i]

  # Calculate interpolation value
  interp_value = ai*(t-x[i]) + bi

  return interp_value
end
  
# Quadratic spline interpolation
function spline_interpolation(spline::QuadSpline, t)
    
  x = spline.x
  y = spline.y
  h = spline.h
  m = spline.m

  # Get index
  i = max(1, min(searchsortedlast(x, t), length(x)-1))

  # Calculate coefficients
  ai = (y[i+1] - m[i]*h[i] - y[i])/(h[i]^2)
  bi =  m[i]
  ci = y[i]

  # Calculate interpolation value
  interp_value = ai*(t-x[i])^2 + bi*(t-x[i]) + ci

  return interp_value
end

# Cubic spline interpolation
function spline_interpolation(spline::CubicSpline, t)
    
  x = spline.x
  y = spline.y
  h = spline.h
  m = spline.m

  # Get index
  i = max(1, min(searchsortedlast(x, t), length(x)-1))

  # Calculate coefficients
  ai = (m[i+1]-m[i])/(6*h[i])
  bi =  m[i]/2
  ci = (y[i+1]-y[i])/h[i]-((m[i+1]+2*m[i])/6)*h[i]
  di =  y[i]

  # Calculate interpolation value
  interp_value = ai*(t-x[i])^3 + bi*(t-x[i])^2 + ci*(t-x[i]) + di

  return interp_value
end


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