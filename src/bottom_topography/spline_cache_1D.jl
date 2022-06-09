# 

######################
### Linear Splines ###
######################

# Spline structure
mutable struct LinSpline{x_type, y_type, h_type}

  x::x_type
  y::y_type
  h::h_type

  LinSpline(x, y, h) = new{typeof(x), typeof(y), typeof(h)}(x, y, h)

end
  
# Fill structure
function lin_spline(x::Vector{Float64},y::Vector{Float64})

  x,y = sort_data(x,y)
  n = length(x) - 1
  h = map(i -> x[i+1] - x[i], 1:n)

  LinSpline(x, y, h)
end

# Read from file
function lin_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  lin_spline(x,y)
end

#########################
### Quadratic Splines ###
#########################

# Spline structure
mutable struct QuadSpline{x_type, y_type, h_type, m_type}

  x::x_type
  y::y_type
  h::h_type
  m::m_type

  QuadSpline(x, y, h, m) = new{typeof(x), typeof(y), typeof(h), typeof(m)}(x, y, h, m)

end

# Fill structure
function quad_spline(x::Vector{Float64},y::Vector{Float64})

  x,y = sort_data(x,y)
  n = length(x) - 1
  h = map(i -> x[i+1] - x[i], 1:n)

  dl = map(i -> 1 - 1/(2*h[i]), 1:n-1)
  dm = vcat(1, map(i -> 1/(2*h[i]), 1:n-1))
  du = zeros(n-1)
  h_mat = Tridiagonal(dl, dm, du)

  y_vec = map(i -> i == 1 ? 0 : y[i] - y[i-1], 1:n)

  m = h_mat\y_vec

  QuadSpline(x, y, h, m)
end

# Read from file
function quad_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  quad_spline(x,y)
end

#####################
### Cubic Splines ###
#####################

# Based on https://www.rajgunesh.com/resources/downloads/numerical/cubicsplineinterpol.pdf
# for natural boundary condition and
# https://www.youtube.com/watch?v=kDowzxuIzzs
# for not-a-knot boundary condition

# Spline structure 
mutable struct CubicSpline{x_type, y_type, h_type, m_type}

  x::x_type
  y::y_type
  h::h_type
  m::m_type

  CubicSpline(x, y, h, m) = new{typeof(x), typeof(y), typeof(h), typeof(m)}(x, y, h, m)
end

# Filter boundary condition
function cubic_spline(x::Vector{Float64},y::Vector{Float64}; boundary = "natural", smoothing_factor = 0.0)

  if boundary == "natural"
    cubic_spline_nat(x, y; smoothing_factor)
  elseif boundary == "not-a-knot"
    cubic_spline_knot(x, y)
  else
    @error("Only natural and not-a-knot boundary conditions are implemented")
  end
end
    

# Fill structure natural boundary
function cubic_spline_nat(x::Vector{Float64}, y::Vector{Float64}; smoothing_factor = 0.0)

  x,y = sort_data(x,y)

  n = length(x) - 1
  h = map(i -> x[i+1] - x[i], 1:n)

  # Consider spline smoothing if required
  if smoothing_factor > 0.0
    y = cubic_spline_smoothing(smoothing_factor, h, y)
  end

  dl = vcat(h[1:n-1], 0)
  dm = vcat(1, 2 * (h[1:n-1] + h[2:n]), 1)
  du = vcat(0, h[2:n])
  h_mat = Tridiagonal(dl, dm, du)

  y_vec = map(i -> i == 1 || i == n+1 ? 0 : 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1]), 1:n+1)

  m = h_mat\y_vec
  
  CubicSpline(x, y, h, m)
end

# Fill structure not-a-knot boundary
function cubic_spline_knot(x::Vector{Float64}, y::Vector{Float64}; smoothing_factor = 0.0)

  x,y = sort_data(x,y)
  n = length(x) - 1
  h = map(i -> x[i+1] - x[i], 1:n)

  # Consider spline smoothing if required
  if smoothing_factor > 0.0
    y = cubic_spline_smoothing(smoothing_factor, h, y)
  end
  
  dl = vcat(h[1:n-1], (h[n-1] + h[n]))
  dm = vcat(-h[2], 2 * (h[1:n-1] + h[2:n]), -h[n-1])
  du = vcat((h[1]+h[2]), h[2:n])
  
  h_mat             = Matrix(Tridiagonal(dl, dm, du))
  h_mat[1,3]        = -h[1]
  h_mat[end, end-2] = -h[n]
  h_mat             = sparse(h_mat)

  y_vec = map(i -> i == 1 || i == n+1 ? 0 : 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1]), 1:n+1)

  m = h_mat\y_vec
  
  CubicSpline(x, y, h, m)
end

# Read from file
function cubic_spline(path::String; boundary = "natural", smoothing_factor = 0.0)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  cubic_spline(x, y; boundary = boundary, smoothing_factor = smoothing_factor)
end