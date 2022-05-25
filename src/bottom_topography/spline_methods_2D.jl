# Bilinear spline interpolation
function spline_interpolation(spline::BiLinSpline, tx, ty)

  # Local valriables
  x  = spline.x
  y  = spline.y
  z  = spline.z
  hx = spline.hx
  hy = spline.hy
  nx = length(x)
  ny = length(y)

  # Get position of interpolation point in values
  ix = max(1, min(searchsortedlast(x, tx), nx-1))
  iy = max(1, min(searchsortedlast(y, ty), ny-1))

  # Calculate coefficients
  aij =  z[iy     , ix   ]
  bij = (z[iy    , (ix+1)] - aij)/hx[ix]
  cij = (z[(iy+1), ix    ] - aij)/hy[iy]
  dij = (z[(iy+1), (ix+1)] - aij - bij*hx[ix] - cij*hy[iy])/(hx[ix]*hy[iy])

  # Calculate interpolated value
  interp_val = aij + bij*(tx-x[ix]) + cij*(ty-y[iy]) + dij*((tx-x[ix]) * (ty-y[iy]))

  return interp_val
end

# Natural bicubic spline interpolation
function spline_interpolation(spline::BiCubicSpline, tx, ty)

  # Local valriables
  x     = spline.x
  y     = spline.y
  z     = spline.z
  fx    = spline.fx
  fy    = spline.fy
  fxy   = spline.fxy
  nx    = length(x)
  ny    = length(y)
  
  # Get position of interpolation point in values
  ix = max(1, min(searchsortedlast(x, tx), nx-1))
  iy = max(1, min(searchsortedlast(y, ty), ny-1))

  # Get grid size
  hx = x[ix+1] - x[ix]
  hy = y[iy+1] - y[iy]

  # Non derivative values
  I00 = z[ iy   ,  ix   ]
  I10 = z[ iy   , (ix+1)]
  I01 = z[(iy+1),  ix   ]
  I11 = z[(iy+1), (ix+1)]

  # Horizontally derived values
  Ix00 = fx[ iy   ,  ix   ] * hx
  Ix10 = fx[ iy   , (ix+1)] * hx
  Ix01 = fx[(iy+1),  ix   ] * hx
  Ix11 = fx[(iy+1), (ix+1)] * hx

  # Vertically derived values
  Iy00 = fy[ iy   ,  ix   ] * hy
  Iy10 = fy[ iy   , (ix+1)] * hy
  Iy01 = fy[(iy+1),  ix   ] * hy
  Iy11 = fy[(iy+1), (ix+1)] * hy

  # Cross derived values
  Ixy00 = fxy[ iy   ,  ix   ] * hx * hy
  Ixy10 = fxy[ iy   , (ix+1)] * hx * hy
  Ixy01 = fxy[(iy+1),  ix   ] * hx * hy
  Ixy11 = fxy[(iy+1), (ix+1)] * hx * hy

  # Target vector
  beta = [I00, I10, I01, I11,
          Ix00, Ix10, Ix01, Ix11,
          Iy00, Iy10, Iy01, Iy11,
          Ixy00, Ixy10, Ixy01, Ixy11]

  # Coefficients
  alpha = spline.M_inv*beta
  coeff = reshape(alpha, (4,4))

  # Normalize values
  u = (tx-x[ix]) / hx
  v = (ty-y[iy]) / hy

  # Calculate interpolation values
  interp_val = 0
  for m in 0:3
    for n in 0:3
      interp_val += coeff[m+1,n+1] * (u^m) * (v^n)
    end
  end

  return interp_val
end