#########################
### Spline Strucutres ###
#########################

# Linear splines
struct LinSpline   
  x::Vector{Float64}
  y::Vector{Float64}
  coeff::Vector{Float64}
end

# Quadratic splines
struct QuadSpline   
  x::Vector{Float64}
  y::Vector{Float64}
  coeff::Vector{Float64}
end

# Cubic splines
struct CubicSpline   
  x::Vector{Float64}
  y::Vector{Float64}
  coeff::Vector{Float64}
end

##########################
### Filling structures ###
##########################

## Linear spline
function lin_spline(x::Vector{Float64},y::Vector{Float64})
  u,t = sort_data(x,y)
  coeff = calc_coeff_lin(u,t)

  LinSpline(u, t, coeff)
end

# From .txt file
function lin_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  lin_spline(x,y)
end

## Quadratic spline
function quad_spline(x::Vector{Float64},y::Vector{Float64})
  u,t = sort_data(x,y)
  coeff = calc_coeff_quad(u,t)

  QuadSpline(u, t, coeff)
end

# From .txt file
function quad_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  quad_spline(x,y)
end

# Cubic splines
function cubic_spline(x::Vector{Float64},y::Vector{Float64})
  u,t = sort_data(x,y)
  coeff = calc_coeff_cubic(u,t)

  CubicSpline(u, t, coeff)
end

function cubic_spline(path::String)

  file = open(path)
  lines = readlines(file)
  close(file)

  num_elements = parse(Int64,lines[2])
  x = [parse(Float64, val) for val in lines[4:3+num_elements]]
  y = [parse(Float64, val) for val in lines[5+num_elements:end]]

  cubic_spline(x,y)
end

#####################
### Helpfunctions ###
######################

# Sorting the inputs
function sort_data(x::Vector{Float64},y::Vector{Float64})
    
  orig_data = hcat(x,y)
  sort_data = orig_data[sortperm(orig_data[:,1]), :]

  u = sort_data[:,1]
  t = sort_data[:,2]

  return u,t
end


## Calculating the coefficients ##


# Linear spline coefficients
function calc_coeff_lin(x::Vector{Float64},y::Vector{Float64})

  num_spline    = length(x) - 1     
  num_coeff     = 2*num_spline      
  num_mat_val   = num_coeff^2       

  # Target vector
  target_vec = zeros(num_coeff)

  idx_target_v1 = collect(1:2:num_coeff)
  idx_target_v2 = collect(2:2:num_coeff)

  target_vec[idx_target_v1] = y[1:num_spline    ]
  target_vec[idx_target_v2] = y[2:num_spline + 1]

  # Creating coeffizient matrix
  coeff_m_v = zeros(num_mat_val)

  idx_si_xi_ai   = collect(1             : (2*num_coeff+2) : num_mat_val)
  idx_si_xip1_ai = collect(2             : (2*num_coeff+2) : num_mat_val)
  idx_si_xip1_bi = collect((num_coeff+2) : (2*num_coeff+2) : num_mat_val)

  h = map(k -> x[k+1] - x[k], 1:num_spline)

  coeff_m_v[idx_si_xi_ai  ] = ones(num_spline)
  coeff_m_v[idx_si_xip1_ai] = ones(num_spline)
  coeff_m_v[idx_si_xip1_bi] = h

  coeff_m = reshape(coeff_m_v, (num_coeff, num_coeff))

  return coeff_m\target_vec
end

# Quadratic spline coefficients
function calc_coeff_quad(x,y)

  num_spline    = length(x) - 1     
  num_coeff     = 3*num_spline      
  num_mat_val   = num_coeff^2       

  # Target vector
  target_vec = zeros(num_coeff)

  idx_target_v1 = collect(1 : 2 :(2*num_spline))
  idx_target_v2 = collect(2 : 2 :(2*num_spline))

  target_vec[idx_target_v1] = y[1:num_spline    ]
  target_vec[idx_target_v2] = y[2:num_spline + 1]

  # Creating coeffizient matrix
  coeff_m_v = zeros(num_mat_val)

  idx_si_xi_ai    = collect(1                            : (3*num_coeff+2) : num_mat_val             )
  idx_si_xip1_ai  = collect(2                            : (3*num_coeff+2) : num_mat_val             )
  idx_si_xip1_bi  = collect((num_coeff  +2)              : (3*num_coeff+2) : num_mat_val             )
  idx_dsi_bi      = collect((num_coeff  +2*num_spline+1) : (3*num_coeff+1) : (num_coeff)*((num_spline-1)*3))
  idx_dsi_bip1    = collect((4*num_coeff+2*num_spline+1) : (3*num_coeff+1) : (num_coeff)*(num_spline    *3))
  idx_ds1_b1      = 2*num_coeff
  idx_si_ci       = collect((2*num_coeff+2)              : (3*num_coeff+2) : num_mat_val             )
  idx_dsi_ci      = collect((2*num_coeff+2*num_spline+1) : (3*num_coeff+1) : (num_coeff)*((num_spline-1)*3))
  idx_ds1_c1      = 3*num_coeff

  h1 = map(k -> x[k+1] - x[k]    , 1:num_spline)
  h2 = map(k -> (x[k+1] - x[k])^2, 1:num_spline)

  coeff_m_v[idx_si_xi_ai  ] = ones(num_spline)
  coeff_m_v[idx_si_xip1_ai] = ones(num_spline)
  coeff_m_v[idx_si_xip1_bi] = h1
  coeff_m_v[idx_dsi_bi    ] = ones(num_spline -1)
  coeff_m_v[idx_dsi_bip1  ] = -1 * ones(num_spline -1)
  coeff_m_v[idx_ds1_b1    ] = 1
  coeff_m_v[idx_si_ci     ] = h2
  coeff_m_v[idx_dsi_ci    ] = 2 * h1[1 : (num_spline-1)]
  coeff_m_v[idx_ds1_c1    ] = 2 * h1[1]

  coeff_m = reshape(coeff_m_v, (num_coeff, num_coeff))

  return coeff_m\target_vec
end

# Cubic spline coeffizienten
function calc_coeff_cubic(x,y)

  num_spline    = length(x) - 1     
  num_coeff     = 4*num_spline      
  num_mat_val   = num_coeff^2       

  # Target vector
  target_vec = zeros(num_coeff)

  idx_target_v1 = collect(1 : 2 :(2*num_spline))
  idx_target_v2 = collect(2 : 2 :(2*num_spline))

  target_vec[idx_target_v1] = y[1:num_spline    ]
  target_vec[idx_target_v2] = y[2:num_spline + 1]

  # Creating coeffizient matrix
  coeff_m_v = zeros(num_mat_val)

  idx_si_xi_ai    = collect(1                            : (4*num_coeff+2) : num_mat_val             )
  idx_si_xip1_ai  = collect(2                            : (4*num_coeff+2) : num_mat_val             )

  idx_si_xip1_bi  = collect((num_coeff  +2)              : (4*num_coeff+2) : num_mat_val             )
  idx_dsi_bi      = collect((num_coeff  +2*num_spline+1) : (4*num_coeff+1) : (num_coeff)*((num_spline-1)*4))
  idx_dsi_bip1    = collect((5*num_coeff+2*num_spline+1) : (4*num_coeff+1) : (num_coeff)*(num_spline    *4))

  idx_si_ci       = collect((2*num_coeff+2)              : (4*num_coeff+2) : num_mat_val             )
  idx_dsi_ci      = collect((2*num_coeff+2*num_spline+1) : (4*num_coeff+1) : (num_coeff)*((num_spline-1)*4))
  idx_ddsi_ci     = collect((2*num_coeff+3*num_spline)   : (4*num_coeff+1) : (num_coeff)*((num_spline-1)*4))
  idx_ddsi_cip1   = collect((6*num_coeff+3*num_spline)   : (4*num_coeff+1) : (num_coeff)*((num_spline  )*4))
  idx_dds1_c1     = 3*num_coeff - 1
  idx_ddsn_cn     = (num_coeff-1) * num_coeff

  idx_si_di       = collect((3*num_coeff+2)              : (4*num_coeff+2) : num_mat_val             )
  idx_dsi_di      = collect((3*num_coeff+2*num_spline+1) : (4*num_coeff+1) : (num_coeff)*((num_spline-1)*4))
  idx_ddsi_di     = collect((3*num_coeff+3*num_spline)   : (4*num_coeff+1) : (num_coeff)*((num_spline-1)*4))
  idx_ddsn_dn     = num_mat_val

  h1 = map(k -> x[k+1] - x[k]    , 1:num_spline)
  h2 = map(k -> (x[k+1] - x[k])^2, 1:num_spline)
  h3 = map(k -> (x[k+1] - x[k])^3, 1:num_spline)

  coeff_m_v[idx_si_xi_ai  ] = ones(num_spline)
  coeff_m_v[idx_si_xip1_ai] = ones(num_spline)

  coeff_m_v[idx_si_xip1_bi] = h1
  coeff_m_v[idx_dsi_bi    ] = ones(num_spline -1)
  coeff_m_v[idx_dsi_bip1  ] = -1 * ones(num_spline -1)

  coeff_m_v[idx_si_ci     ] = h2
  coeff_m_v[idx_dsi_ci    ] = 2  * h1[1 : (num_spline-1)]
  coeff_m_v[idx_ddsi_ci   ] = 2  * ones(num_spline -1)
  coeff_m_v[idx_ddsi_cip1 ] = -2 * ones(num_spline -1)
  coeff_m_v[idx_dds1_c1   ] = 2
  coeff_m_v[idx_ddsn_cn   ] = 2

  coeff_m_v[idx_si_di     ] = h3
  coeff_m_v[idx_dsi_di    ] = 3 * h2[1 : (num_spline-1)]
  coeff_m_v[idx_ddsi_di   ] = 6 * h1[1 : (num_spline-1)]
  coeff_m_v[idx_ddsn_dn   ] = 6 * h1[end]

  coeff_m = reshape(coeff_m_v, (num_coeff, num_coeff))

  return coeff_m\target_vec
end

##############################################################
### Functions to create the spline interpolation functions ###
##############################################################

# Linear splines
function get_func(p, Interpolation::LinSpline)

  idx_in_x        = max(1, min(searchsortedlast(Interpolation.x, p), length(Interpolation.x) -1))
  idx_in_coeff    = (idx_in_x-1) * 2 + 1

  return Interpolation.coeff[idx_in_coeff] + Interpolation.coeff[idx_in_coeff+1] * (p - Interpolation.x[idx_in_x])
end

# Quadratic splines
function get_func(p, Interpolation::QuadSpline)

  idx_in_x        = max(1, min(searchsortedlast(Interpolation.x, p), length(Interpolation.x) -1))
  idx_in_coeff    = (idx_in_x-1) * 3 + 1

  return Interpolation.coeff[idx_in_coeff] + Interpolation.coeff[idx_in_coeff+1] * (p - Interpolation.x[idx_in_x])
end

# Cubic splines
function get_func(p, Interpolation::CubicSpline)

  idx_in_x        = max(1, min(searchsortedlast(Interpolation.x, p), length(Interpolation.x) -1))
  idx_in_coeff    = (idx_in_x-1) * 4 + 1

  return Interpolation.coeff[idx_in_coeff] + Interpolation.coeff[idx_in_coeff+1] * (p - Interpolation.x[idx_in_x])
end