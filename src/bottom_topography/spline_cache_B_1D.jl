######################
### Cubic B Spline ###
######################

# Cubic B Spline structure 
mutable struct CubicBSpline{x_type, y_type, h_type, Q_type, IP_type}

  x::x_type
  y::y_type
  h::h_type
  Q::Q_type
  IP::IP_type

  CubicBSpline(x, y, h, Q, IP) = new{typeof(x), typeof(y), typeof(h), typeof(Q), 
  typeof(IP)}(x, y, h, Q, IP)
end

# fill structure
function cubic_b_spline(x::Vector, y::Vector; boundary = "free")

  n = length(x)
  P = vcat(0, y, 0)
  h = x[2] - x[1]
  IP = [-1  3 -3 1;
         3 -6  3 0;
        -3  0  3 0;
         1  4  1 0]

  # free end condition
  if boundary == "free"
    du = vcat(-2, ones(n))
    dm = vcat(1, 4*ones(n), 1)
    dl = vcat(ones(n), -2)
    
    Phi             = Matrix(Tridiagonal(dl, dm, du))
    Phi[1  , 3    ] = 1
    Phi[end, end-2] = 1
    Phi_free        = sparse(Phi)

    Q_free = 6 * (Phi_free\P) 

    CubicBSpline(x, y, h, Q_free, IP)

  # not-a-knot end condition
  elseif boundary == "not-a-knot"
    du = vcat(4, ones(n))
    dm = vcat(-1, 4*ones(n), -1)
    dl = vcat(ones(n), 4)
    
    Phi                       = Matrix(Tridiagonal(dl, dm, du))
    Phi[1  , 3:5            ] = [-6 4 -1]
    Phi[end, (end-4):(end-2)] = [-1 4 -6]
    Phi_knot                  = sparse(Phi)

    Q_knot = 6 * (Phi_knot\P) 

    CubicBSpline(x, y, h, Q_knot, IP)

  # fit points as control points for smoothing
  elseif boundary == "smooth"
    Q_smooth = vcat(y[1], y, y[end])
    
    CubicBSpline(x, y, h, Q_smooth, IP)

  else
    @error("Only free, not-a-knot and smooth boundary conditions are implemented!")
  end
end

# read from file
function cubic_b_spline(path::String; boundary = "free")

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  cubic_b_spline(x, y; boundary = boundary)
end