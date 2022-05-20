# Sort data to follow logic of the remaining code
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

# Calculate the horizontal derivation using the method of centered differences
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

# Calculate the vertical derivation using the method of centered differences
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