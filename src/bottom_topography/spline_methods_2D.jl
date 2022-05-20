# Bilinear spline interpolation
function spline_interpolation(spline::BiLinSpline, tx, ty)

  x  = spline.x
  y  = spline.y
  z  = spline.z
  hx = spline.hx
  hy = spline.hy
  nx = length(x)
  ny = length(y)

  # Get indices
  ix = max(1, min(searchsortedlast(x, tx), nx-1))
  iy = max(1, min(searchsortedlast(y, ty), ny-1))

  # Calculate coefficients
  aij =  z[iy     , ix   ]
  bij = (z[iy    , (ix+1)] - aij)/hx[ix]
  cij = (z[(iy+1), ix    ] - aij)/hy[iy]
  dij = (z[(iy+1), (ix+1)] - aij - bij*hx[ix] - cij*hy[iy])/(hx[ix]*hy[iy])

  # Calculate interpolation value
  interp_val = aij + bij*(tx-x[ix]) + cij*(ty-y[iy]) + dij*((tx-x[ix]) * (ty-y[iy]))
  
  return interp_val
end

# Bicubic spline interpolation
function spline_interpolation(spline::BiCubicSpline, tx, ty)

  x     = spline.x
  y     = spline.y
  z     = spline.z
  fx    = spline.fx
  fy    = spline.fy
  fxy   = spline.fxy
  nx    = length(x)
  ny    = length(y)

  # Get indices
  ix = max(1, min(searchsortedlast(x, tx), nx-1))
  iy = max(1, min(searchsortedlast(y, ty), ny-1))

  # Non derivative values
  I00 = z[ iy   ,  ix   ]
  I10 = z[ iy   , (ix+1)]
  I01 = z[(iy+1),  ix   ]
  I11 = z[(iy+1), (ix+1)]

  # Horizontally derived values
  Ix00 = fx[ iy   ,  ix   ]
  Ix10 = fx[ iy   , (ix+1)]
  Ix01 = fx[(iy+1),  ix   ]
  Ix11 = fx[(iy+1), (ix+1)]

  # Vertically derived values
  Iy00 = fy[ iy   ,  ix   ]
  Iy10 = fy[ iy   , (ix+1)]
  Iy01 = fy[(iy+1),  ix   ]
  Iy11 = fy[(iy+1), (ix+1)]

  # Cross derived values
  Ixy00 = fxy[ iy   ,  ix   ]
  Ixy10 = fxy[ iy   , (ix+1)]
  Ixy01 = fxy[(iy+1),  ix   ]
  Ixy11 = fxy[(iy+1), (ix+1)]

  # Target vector
  beta = [I00, I10, I01, I11,
          Ix00, Ix10, Ix01, Ix11,
          Iy00, Iy10, Iy01, Iy11,
          Ixy00, Ixy10, Ixy01, Ixy11]

  # Calculate coefficients
  alpha = spline.M_inv*beta
  coeff = reshape(alpha, (4,4))

  # Calculate interpolation value
  hx = x[ix+1] - x[ix]
  hy = y[iy+1] - y[iy]

  u = (tx-x[ix]) / hx
  v = (ty-y[iy]) / hy

  interp_val = 0
  for m in 0:3
    for n in 0:3
      interp_val += coeff[m+1,n+1] * (u^m) * (v^n)
    end
  end

  return interp_val  
end