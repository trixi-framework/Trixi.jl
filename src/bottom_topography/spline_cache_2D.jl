#######################
### Bilinear Spline ###
#######################

# Bilinear spline structure
mutable struct BiLinSpline{x_type, y_type, z_type, hx_type, hy_type}

  x::x_type
  y::y_type
  z::z_type
  hx::hx_type
  hy::hy_type

  BiLinSpline(x, y, z, hx, hy) = new{typeof(x), typeof(y), typeof(z), typeof(hx), 
      typeof(hy)}(x, y, z, hx, hy)

end

# Fill structure
function bilin_spline(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  x, y, z = sort_data(x, y, z)

  nx = length(x) - 1
  ny = length(y) - 1

  hx = map(i -> x[i+1] - x[i], 1:nx)
  hy = map(i -> y[i+1] - y[i], 1:ny)
    
  BiLinSpline(x, y, z, hx, hy)
end

# Read from file
function bilin_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)
  
  num_elements_x = parse(Int64, lines[2])
  num_elements_y = parse(Int64, lines[4])

  x = [parse(Float64, val) for val in lines[6:5+num_elements_x]]
  y = [parse(Float64, val) for val in lines[(7+num_elements_x):(6+num_elements_x+num_elements_y)]]
  z_tmp = [parse(Float64, val) for val in lines[(8+num_elements_x+num_elements_y):end]]

  z = transpose(reshape(z_tmp, (num_elements_x, num_elements_y)))

  bilin_spline(x, y, Matrix(z))
end

##############################
### Bicubic Spline Natural ###
##############################

# Bicubic spline structure
mutable struct BiCubicSpline{x_type, y_type, z_type, Minv_type, fx_type, fy_type, fxy_type}

  x::x_type
  y::y_type
  z::z_type
  M_inv::Minv_type
  fx::fx_type
  fy::fy_type
  fxy::fxy_type

  BiCubicSpline(x, y, z, M_inv, fx, fy, fxy) = new{typeof(x), typeof(y), typeof(z), 
      typeof(M_inv), typeof(fx), typeof(fy), typeof(fxy)}(x, y, z, M_inv, fx, fy, fxy)
end

# Fill structure
function bicubic_spline(x::Vector{Float64}, y::Vector{Float64}, z::Matrix{Float64})

  x, y, z = sort_data(x, y, z)
    
  row_id = [1, 
            2, 
            3, 3, 3, 3, 
            4, 4, 4, 4, 
            5, 
            6,
            7, 7, 7, 7, 
            8, 8, 8, 8, 
            9, 9, 9, 9,
            10, 10, 10, 10,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            13, 13, 13, 13,
            14, 14, 14, 14,
            15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
            16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,]
  col_id = [1, 
            5, 
            1, 2, 5, 6,
            1, 2, 5, 6,
            9, 
            13,
            9, 10, 13, 14,
            9, 10, 13, 14,
            1, 3, 9, 11,
            5, 7, 13, 15,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            1, 3, 9, 11,
            5, 7, 13, 15,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,]
  val = [1,
         1,
         -3, 3, -2, -1,
         2, -2, 1, 1,
         1,
         1,
         -3, 3, -2, -1,
         2, -2, 1, 1,
         -3, 3, -2, -1,
         -3, 3, -2, -1,
         9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
         -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
         2, -2, 1, 1,
         2, -2, 1, 1,
         -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
         4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]

  M_inv = sparse(row_id, col_id, val)

  fx  = horizontal_derivative(x, y, z)
  fy  = vertical_derivative(x, y, z)
  fxy = cross_derivative(x, y, z)

  BiCubicSpline(x, y, z, M_inv, fx, fy, fxy)
end

# Read from file
function bicubic_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements_x = parse(Int64, lines[2])
  num_elements_y = parse(Int64, lines[4])

  x  = [parse(Float64, val) for val in lines[6:5+num_elements_x]]
  y  = [parse(Float64, val) for val in lines[(7+num_elements_x):(6+num_elements_x+num_elements_y)]]
  z_tmp = [parse(Float64, val) for val in lines[(8+num_elements_x+num_elements_y):end]]

  z = transpose(reshape(z_tmp, (num_elements_x, num_elements_y)))

  bicubic_spline(x, y, Matrix(z))
end