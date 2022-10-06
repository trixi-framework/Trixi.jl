# Help functions for 2D spline interpolation

# Sort data to follow logic of the remaining code.
# x and y are sorted ascending. The matrix z contains
# the corresponding values.
function sort_data(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  zx              = transpose(z)
  original_data_x = hcat(x, zx)
  sorted_data_x   = original_data_x[sortperm(original_data_x[:,1]), :]

  x_sorted  = sorted_data_x[:,1]
  z_interim = sorted_data_x[:, 2:end]

  zy              = transpose(z_interim)
  original_data_y = hcat(y, zy)
  sorted_data_y   = original_data_y[sortperm(original_data_y[:,1]), :]

  y_sorted = sorted_data_y[:,1]
  z_sorted   = sorted_data_y[:,2:end]
  
  return x_sorted, y_sorted, Matrix(z_sorted)
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
#      z_nm, ..., z_nm)
# to control_points = (x_1, y_1, z_11;
#                      x_2, y_1, z_21;
#                      ..., ..., ....;
#                      x_n, y_m, z_nm)
function restructure_data(x, y, z)

  x_mat = repeat(x, 1, length(y))
  y_mat = repeat(y, 1, length(x))
  
  p = length(z)

  x_vec = vec(reshape(x_mat , (p, 1)))
  y_vec = vec(reshape(y_mat', (p, 1)))
  z_vec = vec(reshape(z'    , (p, 1)))

  return [x_vec y_vec z_vec]
end

# Thin plate spline approximation  
# Based on:
# Approximate Thin Plate Spline Mappings
# by Gianluca Donato and Serge Belongie, 2002
function calc_tps(lambda, x, y, z)

  restructured_data = restructure_data(x,y,z)
  x_hat = restructured_data[:,1]
  y_hat = restructured_data[:,2]
  z_hat = restructured_data[:,3]

  n = length(x)
  m = length(y)
  p = length(z)

  L   = zeros(p+3, p+3)
  rhs = zeros(p+3, 1  )
  H_f = zeros(p  , 1  )

  # Fill K part of matrix L
  for i in 1:p
    for j in (i+1):p
      p_i    = [x_hat[i], y_hat[i]]
      p_j    = [x_hat[j], y_hat[j]]
      U      = tps_base_func(norm(p_i .- p_j))
      L[i,j] = U
      L[j,i] = U
    end
  end

  # Fill rest of matrix L
  L[1:p,1:p] = L[1:p,1:p] + lambda * diagm(ones(p))
  
  L[1:p,p+1] = ones(p)
  L[1:p,p+2] = x_hat
  L[1:p,p+3] = y_hat

  L[p+1,1:p] = ones(p)
  L[p+2,1:p] = x_hat
  L[p+3,1:p] = y_hat

  # Fill Vektor v
  rhs[1:p,1] = z_hat

  # Calculate solution vector
  coeff = L\rhs

  # Fill matrix grid with smoothed z values
  for i in 1:p
    H_f[i] = coeff[p+1] + coeff[p+2]*x_hat[i] + coeff[p+3]*y_hat[i]
    p_i    = [x_hat[i], y_hat[i]]
    for k in 1:p
      p_k    = [x_hat[k], y_hat[k]]
      H_f[i] = H_f[i] + coeff[k] * tps_base_func(norm(p_i .- p_k))
    end
  end

  return transpose(reshape(H_f, (n,m)))
end