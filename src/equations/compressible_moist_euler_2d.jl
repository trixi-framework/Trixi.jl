# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


struct CompressibleMoistEulerEquations2D{RealT<:Real} <: AbstractCompressibleMoistEulerEquations{2, 6}
  p_0::RealT   # constant reference pressure 1000 hPa(100000 Pa)
  c_pd::RealT   # dry air constant
  c_vd::RealT   # dry air constant
  R_d::RealT   # dry air gas constant
  c_pv::RealT   # moist air constant
  c_vv::RealT   # moist air constant
  R_v::RealT   # moist air gas constant
  c_pl::RealT # liqid water constant
  g::RealT # gravitation constant
  kappa::RealT # ratio of the gas constand R_d
  gamma::RealT # = inv(kappa- 1); can be used to write slow divisions as fast multiplications
  a::RealT
  c_r::RealT
  N_0r::RealT
  rho_w::RealT # massdensity of water
  L_00::RealT # latent evaporation heat at 0 K
end

function CompressibleMoistEulerEquations2D(;RealT=Float64)
   p_0 = 100000.0
   c_pd = 1004.0
   c_vd = 717.0
   R_d = c_pd-c_vd
   c_pv = 1885.0
   c_vv = 1424.0
   R_v = c_pv-c_vv
   c_pl =  4186.0
   g = 9.81
   gamma = c_pd / c_vd # = 1/(1 - kappa)
   kappa = 1 - inv(gamma)
   a = 360.0
   c_r = 130.0
   N_0r = 8000000.0
   rho_w = 1000.0
   L_00 = 3147620.0
   return CompressibleMoistEulerEquations2D{RealT}(p_0, c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl,  g, kappa, gamma, a, c_r, N_0r, rho_w, L_00)
  end


varnames(::typeof(cons2cons), ::CompressibleMoistEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_E", "rho_qv", "rho_ql")
varnames(::typeof(cons2prim), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "p", "qv", "ql")
varnames(::typeof(cons2pot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "pottemp", "qv", "ql")


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho
  E = rho_E / rho


  p, W_f, T, e, E_l = get_moist_profile(u, equations)

  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = (rho_E + p) * v1
    f5 = rho_v1 * qv
    f6 = rho_v1 * ql
  else # flux in z direction includes condensation terms 
    f1 = rho_v2 + rho_ql * W_f
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = (rho_E + p) * v2 + rho_ql * W_f * E_l
    f5 = rho_v2 * qv
    f6 = rho_ql * (v2 + W_f)
  end
  return SVector(f1, f2, f3, f4, f5, f6)
end


function boundary_condition_reflection(u_inner, orientation::Integer, direction, x, t,
                                       surface_flux_function,
                                       equations::CompressibleMoistEulerEquations2D)
  # Orientation 3 neg-y-direction/unten
  # Orientation 4 pos-y-direction/oben

  if !(orientation == 2)
    @info(orientation)
    error("This boundary condition is not supposed to be called in x direction")
  end

  rho, rho_v1, rho_v2, rho_E = u_inner
  p = (equations.gamma - 1) * (rho_E - 0.5 * inv(rho) * (rho_v1^2 + rho_v2^2))
  a_local = sqrt(equations.gamma * p * inv(rho))

  if direction == 3
    p_wall = p + (a_local * rho_v2 / rho)
  else # direction == 4
    p_wall = p - (a_local * rho_v2 / rho)
  end

return SVector(0, 0,  p_wall, 0)
end


function boundary_condition_slip_wall(u_inner, orientation::Integer, direction, x, t,
                                       surface_flux_function,
                                       equation::CompressibleMoistEulerEquations2D)
if orientation == 1 # interface in x-direction
u_boundary = SVector(u_inner[1], -u_inner[2],  u_inner[3], u_inner[4], u_inner[5],u_inner[6])
else # interface in y-direction
u_boundary = SVector(u_inner[1],  u_inner[2], -u_inner[3], u_inner[4], u_inner[5],u_inner[6])
end

# Calculate boundary flux
if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
else # u_boundary is "left" of boundary, u_inner is "right" of boundary
flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
end

return flux
end


function initial_condition_warm_bubble(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, kappa, g, c_pd, c_vd, R_d, R_v = equations
  xc = 0
  zc = 2000
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rc = 2000
  θ_ref = 300
  Δθ = 0

  if r <= rc
     Δθ = 2 * cospi(0.5*r/rc)^2
  end

  #Perturbed state:
  θ = θ_ref + Δθ # potential temperature
  π_exner = 1 - g / (c_pd * θ) * x[2] # exner pressure
  ρ = p_0 / (R_d * θ) * (π_exner)^(c_vd / R_d) # density
  p = p_0 * (1-kappa * g * x[2] / (R_d * θ_ref))^(c_pd / R_d)
  T = p / (R_d * ρ)

  v1 = 20
  v2 = 0
  ρ_v1 = ρ * v1
  ρ_v2 = ρ * v2
  ρ_e = ρ * c_vd * T + 1/2 * ρ * (v1^2 + v2^2)  
  return SVector(ρ, ρ_v1, ρ_v2, ρ_e, 0 ,0)
end


function source_terms_warm_bubble(du, u, equations::CompressibleMoistEulerEquations2D, dg, cache)
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      # TODO: performance use temp
      #x1 = x[1, i, j, element_id]
      #x2 = x[2, i, j, element_id]
      du[3, i, j, element] -=  equations.g * u[1, i, j, element]
      du[4, i, j, element] -=  equations.g * u[3, i, j, element]
    end
  end
  return nothing
end

function source_terms_moist_air(du, u, equations::CompressibleMoistEulerEquations2D, dg, cache)
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = u[:, i, j ,element]
      _, _, _, _, _, E_v = get_moist_profile(u, equations)
      Q_v = ground_vapor_source_term(u, equations)
      Q_ph = phase_change_term(u, equations)
      du[3, i, j, element] += -equations.g * u_local[1]
      du[4, i, j, element] += -equations.g * u_local[3] + E_v * Q_v
      du[5, i, j, element] += Q_v + Q_ph
      du[6, i, j, element] += -Q_ph
    end
  end
  return nothing
end

function get_energy_factor(u, equations::CompressibleMoistEulerEquations2D)

  
end

function get_moist_profile(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_vd, R_d, c_vv, c_pv, R_v, c_pl, c_r, N_0r, rho_w, L_00, g = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  rho_qd = rho - rho_qv - rho_ql

  gm = exp(2.45374) # Gamma(4.5)

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho
  E = rho_E / rho

  # inner energy
  e = E - rho * (g * v2 + 0.5 * (v1^2 + v2^2))
  # Absolute Temperature
  T = inv(rho_qd * c_vd + rho_qv * c_vv + rho_ql * c_pl) * (e - L_00 * rho_qv)
      

  # Pressure
  p = (rho_qd * R_d + rho_qv * R_v) * T

  # Parametrisation by Frisius and Wacker
  W_f = - c_r * gm * inv(6) * (rho_ql * inv(pi * rho_w * N_0r))^(1/8)

  # Energy factors for cource terms
  e_l = equations.c_pl * T
  E_l = E - e + e_l

  h_v = c_pv * T + L_00
  E_v = E - e + h_v

  return SVector(p, W_f, T, e, E_l, E_v)
end


# This source term models the phase chance between could water and vapor
function phase_change_term(u, equations::CompressibleMoistEulerEquations2D)
@unpack R_v= equations
rho, _ , _, _, rho_qv, rho_ql = u
_, _, T, _ = get_moist_profile(u, equations)
rho_qd = rho - rho_qv - rho_ql

T_C = T - 273.15
# saturation vapor pressure
p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))

# saturation density of vapor 
 rho_star_qv = p_vs / (R_v * T)

# Fisher-Burgmeister-Function
a = rho_star_qv - rho_qv
b = rho - rho_qv - rho_qd

# saturation control factor  
# < 1: stronger saturation effect
# > 1: weaker saturation effect
C = 1

return (a + b - sqrt(a^2 + b^2)) * C
end


# This source term models the water vapor generated by the ground topography
function ground_vapor_source_term(u, equations::CompressibleMoistEulerEquations2D)

  return zero(eltype(u))
end


# This source term models the condensed water falling down as rain
function condensation_source_term(u, equations::CompressibleMoistEulerEquations2D)

  return nothing
end



@inline function flux_LMARS(u_ll, u_rr, orientation::Integer , equations::CompressibleMoistEulerEquations2D)
  @unpack a, gamma = equations
  
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_E_ll = u_ll
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll

  rho_rr, rho_v1_rr, rho_v2_rr, rho_E_rr = u_rr
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  
  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?

  p_ll = (gamma - 1) * (rho_E_ll - 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll))
  p_rr = (gamma - 1) * (rho_E_rr - 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr))

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = rho_E_ll + p_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = rho_E_rr + p_rr
    end

    flux = SVector(f1, f2, f3, f4) * v_interface + SVector(0, 1, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = rho_E_ll + p_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = rho_E_rr + p_rr
    end

    flux = SVector(f1, f2, f3, f4) * v_interface + SVector(0, 0, 1, 0) * p_interface
  end

  return flux
end


@inline function flux_LMARS(u_ll, u_rr, normal_direction::AbstractVector , equations::CompressibleMoistEulerEquations2D)
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_E_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_E_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_v2_ll
  
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_v2_rr
  # Calculate scalar product with normal vector
  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  
  # Compute the necessary interface flux components

  rho_mean = 0.5*(rho_ll + rho_rr) # TODO why choose the mean value here?

  v_add = rho_v1 * v1 + rho_v2 * v2
  p_ll = (equations.gamma - 1) * (rho_E_ll - 0.5 * v_add)
  p_rr = (equations.gamma - 1) * (rho_E_rr - 0.5 * v_add)

  # diffusion parameter <= 1 
  beta = 1 

  v_interface = 0.5*(v_dot_n_rr + v_dot_n_ll) - beta*inv(2*rho_mean*equations.a)*(p_rr-p_ll)
  p_interface = 0.5*(p_rr + p_ll) - beta*0.5*rho_mean*equations.a*(v_dot_n_rr-v_dot_n_ll)

  if (v_interface > 0)
    f1 = rho_ll
    f2 = f1*v_dot_n_ll[1]
    f3 = f1*v_dot_n_ll[2]
    f4 = rho_E_ll
  else
    f1 = rho_rr
    f2 = f1*v_dot_n_rr[1]
    f3 = f1*v_dot_n_rr[2]
    f4 = rho_E_rr
  end

  flux = SVector(f1, f2, f3, f4)*v_interface + SVector(0, normal_direction[1], normal_direction[2], 0)*p_interface

  return flux
end

#TODO: is it u_ll-u_rr?
function flux_rusanov(u_ll, u_rr, orientation::Integer , equations::CompressibleMoistEulerEquations2D)
  lambda = max(max_abs_speeds(u_ll, equations)[orientation], max_abs_speeds(u_rr, equations)[orientation])
  #F_rusanov = SVector{length(u_ll)}(zeros(eltype(u_ll[1]), length(u_ll)))  
  F_rusanov = 0.5*(flux(u_ll, orientation, equations) + flux(u_rr, orientation, equations)) + lambda*(u_ll - u_rr) 
  return F_rusanov
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::CompressibleMoistEulerEquations2D)
  # cos and sin of the angle between the x-axis and the normalized normal_vector are
  # the normalized vector's x and y coordinates respectively (see unit circle).
  c = normal_vector[1]
  s = normal_vector[2]

  # Apply the 2D rotation matrix with normal and tangent directions of the form
  # [ 1    0    0   0;
  #   0   n_1  n_2  0;
  #   0   t_1  t_2  0;
  #   0    0    0   1 ]
  # where t_1 = -n_2 and t_2 = n_1

  return SVector(u[1],
                 c * u[2] + s * u[3],
                 -s * u[2] + c * u[3],
                 u[4])
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector, equations::CompressibleMoistEulerEquations2D)
  # cos and sin of the angle between the x-axis and the normalized normal_vector are
  # the normalized vector's x and y coordinates respectively (see unit circle).
  c = normal_vector[1]
  s = normal_vector[2]

  # Apply the 2D back-rotation matrix with normal and tangent directions of the form
  # [ 1    0    0   0;
  #   0   n_1  t_1  0;
  #   0   n_2  t_2  0;
  #   0    0    0   1 ]
  # where t_1 = -n_2 and t_2 = n_1

  return SVector(u[1],
                 c * u[2] - s * u[3],
                 s * u[2] + c * u[3],
                 u[4])
end


@inline function max_abs_speeds(u, equations::CompressibleMoistEulerEquations2D)
  rho, v1, v2, p, qv, ql = cons2prim(u, equations)
  c = sqrt(equations.gamma * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = get_moist_profile(u, equations)[1]
  qv = rho_qv / rho
  ql = rho_ql / rho

  return SVector(rho, v1, v2, p, qv, ql)
end

#TODO:
# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  p = (equations.gamma - 1) * (rho_E - 0.5 * rho * v_square)

  s = log(p) - equations.gamma*log(rho)
  rho_p = rho / p

  w1 = (equations.gamma - s) * inv(equations.gamma - 1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = -rho_p

  return SVector(w1, w2, w3, w4, 0, 0)
end

#TODO:
@inline function entropy2cons(w, equations::CompressibleMoistEulerEquations2D)
  # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
  # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
  @unpack gamma = equations

  # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
  # instead of `-rho * s / (gamma - 1)`
  V1, V2, V3, V5 = w .* (gamma-1)

  # s = specific entropy, eq. (53)
  s = gamma - V1 + (V2^2 + V3^2)/(2*V5)

  # eq. (52)
  rho_iota = ((gamma-1) / (-V5)^gamma)^(equations.inv_gamma_minus_one)*exp(-s * equations.inv_gamma_minus_one)

  # eq. (51)
  rho      = -rho_iota * V5
  rho_v1   =  rho_iota * V2
  rho_v2   =  rho_iota * V3
  rho_E    =  rho_iota * (1-(V2^2 + V3^2)/(2*V5))
  return SVector(rho, rho_v1, rho_v2, rho_E, 0, 0)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleMoistEulerEquations2D)
  rho, v1, v2, p, qv, ql = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_qv = rho * qv
  rho_ql = rho * ql
  rho_E  = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end

@inline function cons2pot(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho

  pot1 = rho
  pot2 = v1
  pot3 = v2
  pot4 = pottemp_thermodynamic(u, equations)
  pot5 = qv
  pot6 = ql

  return SVector(pot1, pot2, pot3, pot4, pot5, pot6)
end


@inline function density(u, equations::CompressibleMoistEulerEquations2D)
 rho = u[1]
 return rho
end


@inline function pressure(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
 p = (equations.gamma - 1) * (rho_E - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
 return p
end


@inline function density_pressure(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
 rho_times_p = (equations.gamma - 1) * (rho * rho_E - 0.5 * (rho_v1^2 + rho_v2^2))
 return rho_times_p
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  # Pressure
  p = (equations.gamma - 1) * (cons[4] - 1/2 * (cons[2]^2 + cons[3]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::CompressibleMoistEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) * cons[1] * equations.inv_gamma_minus_one

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleMoistEulerEquations2D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleMoistEulerEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  return (rho_v1^2 + rho_v2^2) / (2 * rho)
end

@inline function velocity(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  return 0.5 * sqrt(v1^2 + v2^2) 
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::CompressibleMoistEulerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end

@inline function pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, p_0, kappa = equations
  # Pressure
  p = get_moist_profile(cons, equations)[1]

  # Potential temperature
  pot = p_0 * (p / p_0)^(1 - kappa) / (R_d * cons[1])

  return pot
end


end # @muladd
