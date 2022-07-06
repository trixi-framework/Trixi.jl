# Helpfunctions for 2D spline interpolation

# Sort data to follow logic of the remaining code.
# x and y are sorted ascending. The matrix z contains
# the corresponding values.
function sort_data(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  zx          = transpose(z)
  orig_data_x = hcat(x, zx)
  sort_data_x = orig_data_x[sortperm(orig_data_x[:,1]), :]

  x_sorted = sort_data_x[:,1]
  z_inter  = sort_data_x[:, 2:end]

  zy          = transpose(z_inter)
  orig_data_y = hcat(y, zy)
  sort_data_y = orig_data_y[sortperm(orig_data_y[:,1]), :]

  y_sorted = sort_data_y[:,1]
  z_temp   = sort_data_y[:,2:end]
  z_sorted = z_temp
  
  return x_sorted, y_sorted, Matrix(z_sorted)
end

# Calculate the horizontal derivative using the method of centered differences
function horizontal_derivative(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  nx = length(x)
  ny = length(y)

  h_deriv = zeros(nx, ny)

  for iy in 1:ny
    for ix in 2:(nx-1)
      inv_hx = 1/(x[ix+1] - x[ix-1])
      h_deriv[iy, ix] = inv_hx * (z[iy, ix+1] - z[iy, ix-1])
    end
  end

  return h_deriv
end

# Calculate the vertical derivative using the method of centered differences
function vertical_derivative(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  nx = length(x)
  ny = length(y)

  v_deriv = zeros(nx, ny)

  for iy in 2:(ny-1)
    inv_hy = 1/(y[iy+1] - y[iy-1])
    for ix in 1:nx
      v_deriv[iy, ix] = inv_hy * (z[iy+1, ix] - z[iy-1, ix])
    end
  end

  return v_deriv
end

# Calculate the cross derivation using the method of centered differences
function cross_derivative(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  nx = length(x)
  ny = length(y)

  c_deriv = zeros(nx, ny)

  for iy in 2:(ny-1)
    inv_hy = 1/(y[iy+1] - y[iy-1])
    for ix in 2:(nx-1)
      inv_hx = 1/(x[ix+1] - x[ix-1])
      c_deriv[iy, ix] = inv_hy * inv_hx * ((z[iy-1, ix-1] + z[iy+1, ix+1]) - z[iy-1, ix+1] - z[iy+1, ix-1])
    end
  end

  return c_deriv
end


###################################
### Thin plate spline smoothing ###
###################################

# Base function for thin plate spline
function tps_base_func(r)

  if r == 0
    return 0
  else
    return r*r*log(r)
  end
    
end

# restructure input data to be able to use the thin plate spline functionality
# data is restructured in the following form:
# x = (x_1, x_2, ..., x_n)
# y = (y_1, y_2, ..., y_m)
# z = (z_11, ..., z_1n;
#      ...., ..., ....;
#      z_m1, ..., z_mn)
# to control_points = (x_1, y_1, z_11;
#                      x_2, y_1, z_12;
#                      ..., ..., ....;
#                      x_n, y_m, z_mn)
function restructure_data(x, y, z)

  x_mat = repeat(x, 1, length(y))
  y_mat = repeat(y, 1, length(x))
  
  p = length(z)

  x_vec = vec(reshape(x_mat , (p, 1)))
  y_vec = vec(reshape(y_mat', (p, 1)))
  z_vec = vec(reshape(z'    , (p, 1)))

  control_points = [x_vec y_vec z_vec]

  return control_points

end

# Thin plate spline approximation  
# Based on:
# Approximate Thin Plate Spline Mappings
# by Gianluca Donato and Serge Belongie, 2002
function calc_tps(λ, x, y, z)

  control_points = restructure_data(x,y,z)

  nx = length(x)
  ny = length(y)
  p  = length(z)

  mtx_l = zeros(p+3,p+3)
  mtx_v = zeros(p+3,1)
  grid  = zeros(nx, ny)

  # Fill K part of matrix A
  a = 0
  for i in 1:p
    for j in (i+1):p
      pt_i = control_points[i,1:2]
      pt_j = control_points[j,1:2]
      elen = norm(pt_i .- pt_j)
      mtx_l[i,j]      = tps_base_func(elen)
      mtx_l[j,i]      = tps_base_func(elen)
      a = a + elen*2
    end
  end

  a = a/(p*p)

  # Fill rest of matrix A
  mtx_l[1:p,1:p] = mtx_l[1:p,1:p] + λ * (a*a) * diagm(ones(p))
  
  mtx_l[1:p,p+1] = ones(p)
  mtx_l[1:p,p+2] = control_points[:,1]
  mtx_l[1:p,p+3] = control_points[:,2]

  mtx_l[p+1,1:p] = ones(p)
  mtx_l[p+2,1:p] = control_points[:,1]
  mtx_l[p+3,1:p] = control_points[:,2]

  # Fill Vektor v
  mtx_v[1:p,1] = control_points[1:p,3]

  # Calculate solution vector
  lsg_vec = mtx_l\mtx_v

  # Fill matrix grid with smoothed z values
  for i in 1:nx
    for j in 1:ny
      h = lsg_vec[p+1] + lsg_vec[p+2]*x[i] + lsg_vec[p+3]*y[j]
      for k in 1:p
        elen = norm(control_points[k,1:2] .- [x[i], y[j]])
        h = h + lsg_vec[k] * tps_base_func(elen)
      end
      grid[j,i]= h
    end
  end

  return grid
end