function spline_interpolation(spline::LinSpline, t)
    
    x = spline.x
    y = spline.y
    h = spline.h
  
    i = max(1, min(searchsortedlast(x, t), length(x)-1))
  
    ai = (y[i+1]-y[i])/(h[i])
    bi =  y[i]
  
    return ai*(t-x[i]) + bi
  end
  
  function spline_interpolation(spline::QuadSpline, t)
      
    x = spline.x
    y = spline.y
    h = spline.h
    m = spline.m
  
    i = max(1, min(searchsortedlast(x, t), length(x)-1))
  
    ai = (y[i+1] - m[i]*h[i] - y[i])/(h[i]^2)
    bi =  m[i]
    ci = y[i]
  
    return ai*(t-x[i])^2 + bi*(t-x[i]) + ci
  end
  
  function spline_interpolation(spline::CubicSpline, t)
      
    x = spline.x
    y = spline.y
    h = spline.h
    m = spline.m
  
    i = max(1, min(searchsortedlast(x, t), length(x)-1))
  
    ai = (m[i+1]-m[i])/(6*h[i])
    bi =  m[i]/2
    ci = (y[i+1]-y[i])/h[i]-((m[i+1]+2*m[i])/6)*h[i]
    di =  y[i]
  
    return ai*(t-x[i])^3 + bi*(t-x[i])^2 + ci*(t-x[i]) + di
  end