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
  c_r::RealT
  N_0r::RealT
  rho_w::RealT # massdensity of water
  L_00::RealT # latent evaporation heat at 0 K
  a::RealT
  RainConst::RealT # Entropie correction Term
  Rain::Bool
end

function CompressibleMoistEulerEquations2D(;g= 9.81, Rain=false, RealT=Float64)
   p_0 = 100000.0
   c_pd = 1004.0
   c_vd = 717.0
   R_d = c_pd-c_vd
   c_pv = 1885.0
   c_vv = 1424.0
   R_v = c_pv-c_vv
   c_pl =  4186.0
   gamma = c_pd / c_vd # = 1/(1 - kappa)
   kappa = 1 - inv(gamma)
   c_r = 130.0
   N_0r = 8000000.0
   rho_w = 1000.0
   L_00 = 3147620.0
   a = 360.0
   RainConst = - c_r * exp(2.45374) * inv(3) * inv(pi * rho_w * N_0r)
   return CompressibleMoistEulerEquations2D{RealT}(p_0, c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl,  g, kappa, gamma, c_r, N_0r, rho_w, L_00, a, RainConst, Rain)
  end


varnames(::typeof(cons2cons), ::CompressibleMoistEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_E", "rho_qv", "rho_ql")
varnames(::typeof(cons2prim), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "p", "qv", "ql")
varnames(::typeof(cons2temp), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "T", "qv", "ql")
varnames(::typeof(cons2drypot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "drypottemp", "qv", "ql")
varnames(::typeof(cons2moistpot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "moistpottemp", "qv", "ql")
varnames(::typeof(cons2moist), ::CompressibleMoistEulerEquations2D) = ("qv", "ql", "rt", "T", "H", "aeqpottemp")
varnames(::typeof(cons2aeqpot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "aeqpottemp", "rv", "rt")


#=
# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho
  E = rho_E / rho


  p, _, W_f, _, E_l, _ = get_moist_profile(u, equations)

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
=#


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pl, Rain = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho
  p, T = get_current_condition(u, equations)
  if orientation == 1
    f1 = rho_v1 
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2 
    f4 = (rho_E + p) * v1 
    f5 = rho_v1 * qv
    f6 = rho_v1 * ql 
  elseif Rain
    W_f = fall_speed_rain(rho_ql, rho_ql, equations)
    f1 = rho_v2 + rho_ql * W_f 
    f2 = rho_v2 * v1 
    f3 = rho_v2 * v2 + rho_ql * W_f * v2 + p
    f4 = (rho_E + p) * v2 + W_f * rho_ql * (c_pl * T + 0.5*(v1*v1 + v2*v2))
    f5 = rho_v2 * qv
    f6 = rho_v2 * ql + rho_ql * W_f   
  else
    f1 = rho_v2 
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p 
    f4 = (rho_E + p) * v2  
    f5 = rho_v2 * qv
    f6 = rho_v2 * ql
  end
  return SVector(f1, f2, f3, f4, f5, f6)
end


@inline function flux(u, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pl, R_d, R_v, Rain = equations
  @unpack c_pl, Rain = equations
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho
  p, T = get_current_condition(u, equations)
  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  W_f = 0
  if Rain
    W_f = normal_direction[2] * fall_speed_rain(rho_ql, rho_ql, equations)
  end
  rho_v_normal = rho * v_normal
  f1 = rho_v_normal + rho_ql*W_f
  f2 = (rho_v_normal ) * v1 + p * normal_direction[1]
  f3 = (rho_v_normal + rho_ql*W_f) * v2 + p * normal_direction[2]
  f4 = (rho_e + p) * v_normal + W_f * rho_ql * (c_pl * T + 0.5*(v1*v1 + v2*v2))
  f5 = rho_v_normal * qv 
  f6 = (rho_v_normal + rho*W_f) * ql 
  return SVector(f1, f2, f3, f4, f5, f6)
end


@inline function boundary_condition_slip_wall(u_inner, orientation::Integer, direction, x, t,
                                       surface_flux_function,
                                       equation::CompressibleMoistEulerEquations2D)
  if orientation == 1 # interface in x-direction
    u_boundary = SVector(u_inner[1], -u_inner[2],  u_inner[3], u_inner[4], u_inner[5], u_inner[6])
  else # interface in y-direction
    u_boundary = SVector(u_inner[1],  u_inner[2], -u_inner[3], u_inner[4], u_inner[5], u_inner[6])
  end

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
  end

  return flux
end


@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
  surface_flux_function, equations::CompressibleMoistEulerEquations2D)

  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_
  
  # rotate the internal solution state
  u_local = rotate_to_x(u_inner, normal, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent, p_local, qv_local, ql_local = cons2prim(u_local, equations)

  # Get the solution of the pressure Riemann problem
  # See Section 6.3.3 of
  # Eleuterio F. Toro (2009)
  # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
  if v_normal <= 0.0
  sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
  p_star = p_local * (1.0 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2.0 * inv(equations.kappa))
  else # v_normal > 0.0
  A = 2.0 / ((equations.gamma + 1) * rho_local)
  B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
  p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
  end

  # For the slip wall we directly set the flux as the normal velocity is zero
  return SVector(zero(eltype(u_inner)),
  p_star * normal[1],
  p_star * normal[2],
  zero(eltype(u_inner)),
  zero(eltype(u_inner)),
  zero(eltype(u_inner))) * norm_
end


@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, direction, x, t,
  surface_flux_function, equations::CompressibleMoistEulerEquations2D)
  # flip sign of normal to make it outward pointing, then flip the sign of the normal flux back
  # to be inward pointing on the -x and -y sides due to the orientation convention used by StructuredMesh
  if isodd(direction)
    boundary_flux = -boundary_condition_slip_wall(u_inner, -normal_direction,
              x, t, surface_flux_function, equations)
  else
    boundary_flux = boundary_condition_slip_wall(u_inner, normal_direction,
             x, t, surface_flux_function, equations)
  end

  return boundary_flux
end


@inline function rotate_to_x(u, normal_vector::AbstractVector, equations::CompressibleMoistEulerEquations2D)
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
                 u[4], u[5], u[6])
end


# Calculates the absolute temperature of a moist state by solving the 
# equivalentpotential temperature equation f(T) - theta_e = 0 using nsolve
function solve_for_absolute_temperature(moist_state, H, equations, T_0)
  @unpack p_0, c_pd, c_pv, c_pl, L_00 = equations
  theta_e, rho_d, r_v, r_t = moist_state

  c_p = c_pd + c_pl * r_t
  p_d = R_d * rho_d * T
  L_v = L_00 - (c_pl - c_pv) * T

  return nlsolve((T) -> (T * (p_0 / p_d)^(R_d / c_p) * 
                 H^(-r_v * R_v / c_p) * exp(L_v * r_v / (c_p * T)) - theta_e), T_0)
end


function initial_condition_convergence_test_dry(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations

  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  qv = 0
  ql = 0

  mu = ((1 - qv - ql)*c_vd + qv*c_vv + ql*c_pl)

  T = (ini - 1) / c_vd  
  E = (mu*T + qv*L_00 + 1)

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_e = E * ini
  rho_qv = qv * ini 
  rho_ql = ql * ini
  
  return SVector(rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql)
end


@inline function source_terms_convergence_test_dry(u, x, t, equations::CompressibleMoistEulerEquations2D)
  # Same settings as in `initial_condition`
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f

  x1, x2 = x
  si, co = sincos(ω * (x1 + x2 - t))
  rho = c + A * si
  rho_x = ω * A * co
  

  qv = 0
  ql = 0
  mu = ((1 - qv - ql)*c_vd + qv*c_vv + ql*c_pl)
  xi = ((1 - qv - ql) * R_d + qv * R_v)


  T = (rho - 1) / c_vd  
  dT = rho_x / c_vd
  E = (mu * T + qv * L_00 + 1)
  dE = E * rho_x + rho * mu * dT
  dp = xi * (T * rho_x + rho * dT)
  # Note that d/dt rho = -d/dx rho = -d/dy rho.

  du1, du2, du3, du4, du5, du6 = source_terms_moist_bubble(u, x, t, equations)


  du1 += rho_x
  du2 += rho_x + dp
  du3 += du2
  du4 += dE + 2*dp
  du5 += qv * du1
  du6 += ql * du1

  return SVector(du1, du2, du3, du4, du5, du6)
end


function initial_condition_convergence_test_moist(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations

  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

 
  qv = 1/L_00
  ql = qv / 10

  mu = ((1 - qv - ql)*c_vd + qv*c_vv + ql*c_pl)

  T = (ini - 1) /mu + 10/c_vd
  E = (mu*T + qv*L_00 + 1)

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_e = E * ini
  rho_qv = qv * ini 
  rho_ql = ql * ini
  
  return SVector(rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql)
end


@inline function source_terms_convergence_test_all(u, x, t, equations::CompressibleMoistEulerEquations2D)
  # Same settings as in `initial_condition`
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00, c_r, N_0r, rho_w, Rain = equations
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f

  x1, x2 = x
  si, co = sincos(ω * (x1 + x2 - t))
  rho = c + A * si
  rho_x = ω * A * co
  

  qv = 1/L_00
  ql = qv / 10
  mu = ((1 - qv - ql) * c_vd + qv * c_vv + ql * c_pl)
  xi = ((1 - qv - ql) * R_d + qv * R_v)


  T = (rho - 1) / c_vd  
  dT = rho_x / c_vd
  E = (mu * T + qv * L_00 + 1)
  dE = E * rho_x + rho * mu * dT
  dp = xi * (T * rho_x + rho * dT)


  #Calculate Error in Sources with exact solution and u
  u_exact = SVector(rho, rho, rho, rho*E, rho*qv, rho*ql) 

  du1, du2, du3, du4, du5, du6 = ( source_terms_moist_bubble(u, x, t, equations) -
                                   source_terms_moist_bubble(u_exact, x, t, equations))  
  #du1, du2, du3, du4, du5, du6 = zeros(Float64, 6)                              
  # Note that d/dt rho = -d/dx rho = -d/dy rho.
  du1 += rho_x 
  du2 += rho_x + dp
  du3 += du2 
  du4 += dE + 2*dp 
  du5 += qv * rho_x
  du6 += ql * rho_x 

  return SVector(du1, du2, du3, du4, du5, du6)
end


@inline function source_terms_convergence_test_moist(u, x, t, equations::CompressibleMoistEulerEquations2D)
  # Same settings as in `initial_condition`
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00, c_r, N_0r, rho_w, Rain, g = equations
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  
  x1, x2 = x
  si, co = sincos(ω * (x1 + x2 - t))
  rho = c + A * si
  rho_x = ω * A * co
  
  
  qv = 1/L_00
  ql = qv/10 

  mu = ((1 - qv - ql) * c_vd + qv * c_vv + ql * c_pl)
  xi = ((1 - qv - ql) * R_d + qv * R_v)

  T = (rho - 1) /mu + 10/c_vd
  dT = rho_x / mu
  
  E = (mu * T + qv * L_00 + 1)
  drhoE = E * rho_x + rho * mu * dT
  dp = xi * (T * rho_x + rho * dT)

  
    # Rain Term
    e_lp1 = c_pl * T + 1 

  #gm = exp(2.45374) # Gamma(4.5)
  #a = - c_r * gm * inv(6)
  #b = pi * rho_w * N_0r

  #Wf = a * (ql * rho / b)^(1/8)
  #dWf = a / (8 * b * (ql * rho / b)^(7/8)) *  ql * rho_x
  
  #Rflux = rho * ql * dWf + ql * rho_x * Wf
  #RfluxE = e_lp1 * Rflux + c_pl * dT * ql * rho * Wf


  Wf = fall_speed_rain(rho*ql, rho*ql, equations)  
  Rflux = 9/8 * Wf * ql * rho_x
    RfluxE = e_lp1 * Rflux + c_pl * dT * ql * rho * Wf

  #Calculate Error in Sources with exact solution and u
  u_exact = (rho, rho, rho, rho*E, rho*qv, rho*ql) 

  du1, du2, du3, du4, du5, du6 = ( source_terms_rain(u, x, t, equations) -
                                   source_terms_rain(u_exact, x, t, equations))  
  #du1, du2, du3, du4, du5, du6 = zeros(Float64, 6)                              
  # Note that d/dt rho = -d/dx rho = -d/dy rho.
  du1 += rho_x + Rflux
  du2 += rho_x + dp
  du3 += du2 + Rflux
  du4 += drhoE + 2*dp + RfluxE
  du5 += qv * rho_x
  du6 += ql * rho_x + Rflux

  return SVector(du1, du2, du3, du4, du5, du6)
end


function initial_condition_gravity_wave(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, kappa, g, c_pd, c_vd, R_d, R_v = equations
  z = x[2]
  theta_ref = 250
  T_0 = 250
  N = g / sqrt(c_pd * T_0)

  theta = T_0 * exp(N^2 *z / g)
  p = p_0(1 + g^2 * inv(c_pd * T_0 * N^2) * (exp(-z * N^2 / g) - 1))^(- kappa)
  rho = p_0 * inv(theta * R_d) * (p_0 / p)^(inv(gamma))
  T = p / (R_d * rho)

  v1 = 20
  v2 = 0
  rho_v1 = rho * v2
  rho_v2 = rho * v2
  rho_E = rho * c_vd * T + 1/2 * rho * (v1^2 + v2^2)
  rho_qv = 0
  rho_ql = 0
  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end


@inline function source_terms_geopotential(u, x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack g, Rain = equations
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  tmp = rho_v2
  if Rain
    W_f = fall_speed_rain(rho_ql, rho_ql, equations)
    tmp += rho_ql * W_f
  end

  return SVector(zero(eltype(u)), zero(eltype(u)),
                 -g * rho, -g * tmp, 
                 zero(eltype(u)), zero(eltype(u)))
end


@inline function source_terms_geopotential(u, equations::CompressibleMoistEulerEquations2D)
  @unpack g, Rain = equations
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  tmp = rho_v2
  if Rain
    W_f = fall_speed_rain(rho_ql, rho_ql, equations)
    tmp += rho_ql * W_f
  end

  return SVector(zero(eltype(u)), zero(eltype(u)),
                 -g * rho, -g * tmp, 
                 zero(eltype(u)), zero(eltype(u)))
end


@inline function source_terms_moist_bubble(u, x, t, equations::CompressibleMoistEulerEquations2D)

  return source_terms_geopotential(u, equations) + source_terms_phase_change(u, equations)
end


@inline function source_terms_rain(u, x, t, equations::CompressibleMoistEulerEquations2D)
  return source_terms_geopotential(u, equations) + 
         source_terms_phase_change(u, equations) +
         source_terms_ground_vapor(u, x, t, equations)
end


@inline function source_terms_phase_change(u, equations::CompressibleMoistEulerEquations2D)
  Q_ph = phase_change_term(u, equations)

  return SVector(zero(eltype(u)), zero(eltype(u)), zero(eltype(u)),
                 zero(eltype(u)) , Q_ph, -Q_ph)
end


@inline function source_terms_ground_vapor(u, x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pv, L_00, g = equations
  rho, rho_v1, rho_v2, rho_e, Rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  Q_v = ground_vapor_term(u, x, t, equations)
  _, T = get_current_condition(u, equations)
  K = 0.5 * (v1^2 + v2^2)
  h_v = c_pv * T + L_00

  return SVector(Q_v, Q_v * v1, Q_v * v2,
                 Q_v * (h_v  + K) , Q_v , zero(eltype(u)))
end


@inline function source_terms_sponge_layer(u, x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pv, L_00, g = equations
  rho, rho_v1, rho_v2, rho_e, Rho_qv, rho_ql = u
  z = x[end]
  z_s = 

  sponge = zero(eltype(u))
  zero = zero(eltype(u))
  if(z > z_s)
  gamma = 2
  z_top =
  alpha =
  tau_s = alpha * sin(1/2 *(z - z_s / (z_top - z_s)) )^gamma 
  end
  
  return SVector(zero, sponge, zero, zero, zero, zero)
end


@inline function get_current_condition(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_vd, R_d, c_vv, c_pv, R_v, c_pl, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  rho_qd = rho - rho_qv - rho_ql

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  
  # inner energy
  rho_e = (rho_E - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  # Absolute Temperature
  T = (rho_e - L_00 * rho_qv) / (rho_qd * c_vd + rho_qv * c_vv + rho_ql * c_pl)  
      
  # Pressure
  p = (rho_qd * R_d + rho_qv * R_v) * T

  return SVector(p, T)
end


# This source term models the phase chance between could water and vapor
@inline function phase_change_term(u, equations::CompressibleMoistEulerEquations2D)
  @unpack R_v = equations
  rho, _ , _, _, rho_qv, rho_ql = u
  _, T = get_current_condition(u, equations)
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
@inline function ground_vapor_term(u, x, t, equations::CompressibleMoistEulerEquations2D)
  xc, zc = (0, 2400)
  rho = u[1]
  f_t = (t-30) / 10
  #a = 0.25
  a = 1
  f_x = sqrt((a * x[1] - xc)^2 + (x[2] - zc)^2) / 200
  0.00025 * exp(-(f_t^2 + f_x^2)) * rho

  return 0.00025 * exp(-(f_t^2 + f_x^2)) * rho
end


# This source term models the condensed water falling down as rain
@inline function fall_speed_rain(rho_ql, rho_ql_speed, equations::CompressibleMoistEulerEquations2D)
  @unpack c_r, rho_w, N_0r = equations
 
  gm = exp(2.45374) # Gamma(4.5)
  # Parametrisation by Frisius and Wacker
  if (rho_ql < 0 || rho_ql_speed < 0)
    return 0
  end
  W_f = - c_r * gm * inv(6) * (rho_ql_speed * inv(pi * rho_w * N_0r))^(1/8)

  return W_f
end


@inline function flux_LMARS(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack a = equations
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  
  # Unpack left and right state
  p_ll, T_ll = get_current_condition(u_ll, equations)
  p_rr, T_rr = get_current_condition(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr

  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?

  

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll
      f4 += p_ll
    else
      f1, f2, f3, f4, f5, f6 = u_rr
      f4 += p_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 1, 0, 0, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)

    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll
      f4 += p_ll
    else
      f1, f2, f3, f4, f5, f6 = u_rr
      f4 += p_rr
    end
    #if (v_interface > 0)
    #  Q_v = fall_speed_rain(u_ll, equations)   
    #  f1 = rho_ll + Q_v
    #  f2 = rho_v1_ll
    #  f3 = rho_v2_ll+ Q_v * v2_ll
    #  f4 = rho_e_ll + p_ll + Q_v * (equations.c_pl * T_ll + 0.5*(v1_ll^2 + v2_ll^2))
    #  f5 = rho_qv_ll
    #  f6 = rho_ql_ll + Q_v
    #else
    #  Q_v = fall_speed_rain(u_rr, equations)
    #  f1 = rho_rr + Q_v
    #  f2 = rho_v1_rr
    #  f3 = rho_v2_rr+ Q_v * v2_rr
    #  f4 = rho_e_rr + p_rr + Q_v * (equations.c_pl * T_rr + 0.5*(v1_rr^2 + v2_rr^2))
    #  f5 = rho_qv_rr
    #  f6 = rho_ql_rr + Q_v
    #end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 0, 1, 0, 0 ,0 ) * p_interface
  end

  return flux
end

@inline function flux_LMARS_rain(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack a, c_pl = equations
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  
  # Unpack left and right state
  p_ll, T_ll = get_current_condition(u_ll, equations)
  p_rr, T_rr = get_current_condition(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr

  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?
  T_mean = 0.5 * (T_ll + T_rr)
  rho_ql_mean = 0.5 * (rho_ql_ll + rho_ql_rr)

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll
      f4 += p_ll
    else
      f1, f2, f3, f4, f5, f6 = u_rr
      f4 += p_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 1, 0, 0, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)
    W_f = fall_speed_rain(rho_mean, rho_mean, equations)
    
    
    if (v_interface > 0)
      Q_fall = rho_ql_rr * fall_speed_rain(rho_ql_rr, rho_ql_mean, equations) # rho_qv * W_f
      v_square_mean = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
      f1, f2, f3, f4, f5, f6 = u_ll * v_interface
      f4 += p_ll * v_interface + Q_fall * (c_pl * T_rr + v_square_mean)
      f1 += Q_fall
      f6 += Q_fall 
      f3 += Q_fall * v_interface
    else
      Q_fall = rho_ql_ll * fall_speed_rain(rho_ql_ll, rho_ql_mean, equations)
      v_square_mean = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
      f1, f2, f3, f4, f5, f6 = u_rr * v_interface
      f4 += p_rr * v_interface + Q_fall * (c_pl * T_ll + v_square_mean)
      f1 += Q_fall
      f6 += Q_fall 
      f3 += Q_fall * v_interface
    end


    flux = SVector(f1, f2, f3, f4, f5, f6) + SVector(0, 0, 1, 0, 0 ,0 ) * p_interface
  end

  return flux
end


@inline function flux_LMARS_rain1(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack a, c_pl = equations
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  
  # Unpack left and right state
  p_ll, T_ll = get_current_condition(u_ll, equations)
  p_rr, T_rr = get_current_condition(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr

  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?
  T_mean = 0.5 * (T_ll + T_rr)
  rho_ql_mean = 0.5 * (rho_ql_ll + rho_ql_rr)

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll
      f4 += p_ll
    else
      f1, f2, f3, f4, f5, f6 = u_rr
      f4 += p_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 1, 0, 0, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)
    
    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll * v_interface
      f4 += p_ll * v_interface 
    else
      f1, f2, f3, f4, f5, f6 = u_rr * v_interface
      f4 += p_rr * v_interface 
    end

    W_f = fall_speed_rain(rho_ql_mean, rho_ql_mean, equations)
    v_square_mean = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
    if(v_interface + W_f > 0)
      W_f = fall_speed_rain(rho_ql_ll, rho_ql_mean, equations)
      f4 += W_f * rho_ql_ll * (c_pl * T_ll + v_square_mean)
      f1 += W_f * rho_ql_ll
      f6 += W_f * rho_ql_ll 
      f3 += W_f * rho_ql_ll * v_interface
    else
      W_f = fall_speed_rain(rho_ql_rr, rho_ql_mean, equations)
      f4 += W_f * rho_ql_rr * (c_pl * T_rr + v_square_mean)
      f1 += W_f * rho_ql_rr
      f6 += W_f * rho_ql_rr 
      f3 += W_f * rho_ql_rr * v_interface
    end


    flux = SVector(f1, f2, f3, f4, f5, f6) + SVector(0, 0, 1, 0, 0 ,0 ) * p_interface
  end

  return flux
end


@inline function flux_LMARS_rain2(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack a, c_pl = equations
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  
  # Unpack left and right state
  p_ll, T_ll = get_current_condition(u_ll, equations)
  p_rr, T_rr = get_current_condition(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr

  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?
  T_mean = 0.5 * (T_ll + T_rr)
  rho_ql_mean = 0.5 * (rho_ql_ll + rho_ql_rr)

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll
      f4 += p_ll
    else
      f1, f2, f3, f4, f5, f6 = u_rr
      f4 += p_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 1, 0, 0, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)
    
    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll * v_interface
      f4 += p_ll * v_interface 
    else
      f1, f2, f3, f4, f5, f6 = u_rr * v_interface
      f4 += p_rr * v_interface 
    end

    W_f = fall_speed_rain(rho_ql_mean, rho_ql_mean, equations)      
    v_liquid_interface = v_interface + W_f
    v_square_mean = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
    if(v_liquid_interface > 0)
      f1 += W_f * rho_ql_ll
      f3 += W_f * rho_ql_ll * v_interface
      f4 += W_f * rho_ql_ll * (c_pl * T_ll + v_square_mean)
      f6 = (W_f + v_interface) * rho_ql_ll
    else
      f1 += W_f * rho_ql_rr 
      f3 += W_f * rho_ql_rr * v_interface
      f4 += W_f * rho_ql_rr * (c_pl * T_rr + v_square_mean)
      f6 = (W_f + v_interface) * rho_ql_rr 
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) + SVector(0, 0, 1, 0, 0 ,0 ) * p_interface
  end

  return flux
end


@inline function flux_LMARS_rain3(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack a, c_pl = equations
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  
  # Unpack left and right state
  p_ll, T_ll = get_current_condition(u_ll, equations)
  p_rr, T_rr = get_current_condition(u_rr, equations)

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr

  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?
  T_mean = 0.5 * (T_ll + T_rr)
  rho_ql_mean = 0.5 * (rho_ql_ll + rho_ql_rr)

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll
      f4 += p_ll
    else
      f1, f2, f3, f4, f5, f6 = u_rr
      f4 += p_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 1, 0, 0, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)
    
    if (v_interface > 0)
      f1, f2, f3, f4, f5, f6 = u_ll * v_interface
      f4 += p_ll * v_interface 
    else
      f1, f2, f3, f4, f5, f6 = u_rr * v_interface
      f4 += p_rr * v_interface 
    end

    W_f_ll = fall_speed_rain(rho_ql_ll, rho_ql_ll, equations) 
    W_f_rr = fall_speed_rain(rho_ql_rr, rho_ql_rr, equations)          
    v_square_mean = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
    W_f = 0.5 * (W_f_ll + W_f_rr)
    if (v_interface + W_f) > 0
      f1 += W_f * rho_ql_ll
      f3 += W_f * rho_ql_ll * v_interface
      f4 += W_f * rho_ql_ll * (c_pl * T_ll + v_square_mean)
      f6 = (W_f + v_interface) * rho_ql_ll
    else
      f1 += W_f * rho_ql_rr 
      f3 += W_f * rho_ql_rr * v_interface
      f4 += W_f * rho_ql_rr * (c_pl * T_rr + v_square_mean)
      f6 = (W_f + v_interface) * rho_ql_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) + SVector(0, 0, 1, 0, 0 ,0 ) * p_interface
  end

  return flux
end


@inline function flux_LMARS(u_ll, u_rr, normal_direction::AbstractVector ,equations::CompressibleMoistEulerEquations2D)
  @unpack a = equations
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  p_ll, T_ll = get_current_condition(u_ll, equations)
  p_rr, T_rr = get_current_condition(u_rr, equations)
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr

  v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] 
  v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  # diffusion parameter <= 1 
  beta = 1 

  # Compute the necessary interface flux components
  rho_mean = 0.5*(rho_ll + rho_rr) # TODO why choose the mean value here?
  norm_ = norm(normal_direction)

  rho = 0.5 * (rho_ll + rho_rr)
  p_interface = 0.5 * (p_ll + p_rr) - beta * 0.5 * a * rho * (v_rr - v_ll) / norm_
  v_interface = 0.5 * (v_ll + v_rr) - beta * 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

  if (v_interface > 0)
    f1, f2, f3, f4, f5, f6 = u_ll * v_interface
    f4 += p_ll * v_interface
  else
    f1, f2, f3, f4, f5, f6 = u_rr * v_interface
    f4 += p_rr * v_interface
  end

  return SVector(f1, 
                 f2 + p_interface * normal_direction[1], 
                 f3 + p_interface * normal_direction[2], 
                 f4, f5, f6)  
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = get_current_condition(u, equations)[1]
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

  #s = log(p) - equations.gamma*log(rho)
  s=0
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

@inline function cons2drypot(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho

  pot1 = rho
  pot2 = v1
  pot3 = v2
  pot4 = dry_pottemp_thermodynamic(u, equations)
  pot5 = qv
  pot6 = ql

  return SVector(pot1, pot2, pot3, pot4, pot5, pot6)
end

@inline function cons2moistpot(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho

  pot1 = rho
  pot2 = v1
  pot3 = v2
  pot4 = moist_pottemp_thermodynamic(u, equations)
  pot5 = qv
  pot6 = ql

  return SVector(pot1, pot2, pot3, pot4, pot5, pot6)
end

@inline function cons2moist(u, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_pd, c_pv, c_pl, p_0 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  ql = rho_ql / rho
  p, T = get_current_condition(u, equations)
 
  p_v = rho_qv * R_v * T
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  H = p_v * inv(p_vs) 

  rho_d = rho - (rho_qv + rho_ql)
  r_v = inv(rho_d) * rho_qv
  r_l = inv(rho_d) * rho_ql

  # Potential temperature
  R_m = R_d + r_v * R_v
  c_pml = c_pd + r_v * c_pv + r_l * c_pl
  kappa_m =  R_m * inv(c_pml)
  pot = T * (p_0 / p)^(kappa_m)

  pot1 = rho
  pot2 = p
  pot3 = T
  pot4 = pot
  pot5 = H
  pot6 = ql

  return SVector(pot1, pot2, pot3, pot4, pot5, pot6)
end


@inline function density(u, equations::CompressibleMoistEulerEquations2D)
 rho = u[1]
 return rho
end


@inline function density_dry(u, equations::CompressibleMoistEulerEquations2D)
  rho_qd = u[1] - (u[5] + u[6])
  return rho_qd
end


@inline function density_vapor(u, equations::CompressibleMoistEulerEquations2D)
  rho_qv = u[5] 
  return rho_qv
end


@inline function density_liquid(u, equations::CompressibleMoistEulerEquations2D)
  rho_ql = u[6]
  return rho_ql
end

@inline function ratio_liquid(u, equations::CompressibleMoistEulerEquations2D)
  rho = u[1]
  rho_ql = u[6]
  ql = rho_ql / rho
  return ql
end

@inline function ratio_vapor(u, equations::CompressibleMoistEulerEquations2D)
  rho = u[1]
  rho_qv = u[5]
  qv = rho_qv / rho
  return qv
end


@inline function pressure(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
 p = get_current_condition(u, equations)[1]
 return p
end


@inline function density_pressure(u, equations::CompressibleMoistEulerEquations2D)
  rho = u[1]
 rho_times_p = rho * get_current_condition(u, equations)[1]
 return rho_times_p
end

#TODO:
# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  # Pressure
  p = (equations.gamma - 1) * (cons[4] - 1/2 * (cons[2]^2 + cons[3]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


#TODO:
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

@inline function dry_pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, p_0, kappa = equations
  # Pressure
  p = get_current_condition(cons, equations)[1]
  # Potential temperature
  pot = p_0 * (p / p_0)^(1 - kappa) / (R_d * cons[1])

  return pot
end

@inline function moist_pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_pd, c_pv, c_pl, p_0 = equations
  # Pressure
  p, T = get_current_condition(cons, equations)
  rho_d = cons[1] - (cons[5] + cons[6])
  r_v = inv(rho_d) * cons[5]
  r_l = inv(rho_d) * cons[6]

  # Potential temperature
  R_m = R_d + r_v * R_v
  c_pml = c_pd + r_v * c_pv + r_l * c_pl
  kappa_m =  R_m * inv(c_pml)
  pot = T * (p_0 / p)^(kappa_m)
  return pot
end

@inline function temp_error(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_pd, c_pv, c_pl, p_0 = equations
  # Pressure
  p, T = get_current_condition(cons, equations)
  rho_d = cons[1] - (cons[5] + cons[6])
  r_v = inv(rho_d) * cons[5]
  r_l = inv(rho_d) * cons[6]

  # Potential temperature
  R_m = R_d + r_v * R_v
  c_pml = c_pd + r_v * c_pv + r_l * c_pl
  kappa_m =  R_m * inv(c_pml)
  pot = T * (p_0 / p)^(kappa_m)
  return abs(300 - pot) + 0.5 * sqrt(abs(energy_kinetic(cons, equations)))
end

@inline function aequivalent_pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, R_d, R_v, p_0, kappa, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = cons
  rho_d = rho - rho_qv - rho_ql
  p, T, _, _, _, _ = get_moist_profile(cons, equations)
  p_v = rho_qv * R_v * T
  p_d = p - p_v
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  H = p_v / p_vs
  r_v = rho_qv / rho_d

  #Aequivalentpotential temperature
  aeq_pot = T * (p_0 / p_d)^( kappa) * H^(- r_v * R_v /c_pd) * exp((L_00 + (c_pv - c_pl) * T ) * (r_v / c_pd * T))

  return aeq_pot
end

@inline function cons2aeqpot(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, R_d, R_v, p_0, kappa, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = cons
  rho_d = rho - rho_qv - rho_ql
  p, T = get_current_condition(cons, equations)
  p_v = rho_qv * R_v * T
  p_d = p - p_v
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  H = p_v / p_vs
  r_v = rho_qv / rho_d
  r_l = rho_ql / rho_d
  r_t = r_v + r_l
  L_v = L_00 + (c_pv - c_pl) * T
  c_p = c_pd + r_t * c_pl
  
  #Aequivalentpotential temperature
  aeq_pot = (T * (p_0 / p_d)^(R_d / c_p) * H^(- r_v * R_v / c_p) *
             exp(L_v * r_v * inv(c_p * T)))

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho

  pot1 = rho
  pot2 = v1
  pot3 = v2
  pot4 = aeq_pot
  pot5 = r_v
  pot6 = r_t
  return SVector(pot1, pot2, pot3, pot4, pot5, pot6)
end

@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, c_vd, c_vv = equations
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)
  qd_ll = 1 - qv_ll - ql_ll
  qd_rr = 1 - qv_rr - ql_rr  
  # Get the density and gas gamma
  gamma_ll = (qd_ll * c_pd + qv_ll * c_pv + ql_ll * c_pl) * inv(qd_ll * c_vd + qv_ll * c_vv + ql_ll * c_pl)
  gamma_rr = (qd_rr * c_pd + qv_rr * c_pv + ql_rr * c_pl) * inv(qd_rr * c_vd + qv_rr * c_vv + ql_rr * c_pl)


  # Compute the sound speeds on the left and right
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  c_ll = sqrt(gamma_ll * p_ll / rho_ll)
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  c_rr = sqrt(gamma_rr * p_rr / rho_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, c_vd, c_vv = equations
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)
  qd_ll = 1 - qv_ll - ql_ll
  qd_rr = 1 - qv_rr - ql_rr  
  # Get the density and gas gamma
  gamma_ll = (qd_ll * c_pd + qv_ll * c_pv ) * inv(qd_ll * c_vd + qv_ll * c_vv )
  gamma_rr = (qd_rr * c_pd + qv_rr * c_pv ) * inv(qd_rr * c_vd + qv_rr * c_vv )
  # Calculate normal velocities and sound speed
  # left
  v_ll = (  v1_ll * normal_direction[1]
          + v2_ll * normal_direction[2] )
  c_ll = sqrt(gamma_ll * p_ll / rho_ll)
  # right
  v_rr = (  v1_rr * normal_direction[1]
          + v2_rr * normal_direction[2] )
  c_rr = sqrt(gamma_rr * p_rr / rho_rr)

  return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction) 
end

@inline function max_abs_speeds(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, c_vd, c_vv = equations
  rho, v1, v2, p, qv, ql = cons2prim(u, equations)
  qd = 1 - qv - ql 

  gamma = (qd * c_pd + qv * c_pv ) * inv(qd * c_vd + qv * c_vv)
  c = sqrt(gamma * p / rho)

  return (abs(v1) + c, abs(v2) + c)
end

# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
# TODO: This doesn't really use the `orientation` - should it?
#@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
#  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
#  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
#  v1_ll = rho_v1_ll / rho_ll
#  v2_ll = rho_v2_ll / rho_ll
#  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
#  p_ll = get_current_condition(u_ll, equations)[1]
#  c_ll = sqrt(equations.gamma * p_ll / rho_ll)
#  v1_rr = rho_v1_rr / rho_rr
#  v2_rr = rho_v2_rr / rho_rr
#  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
#  p_rr = get_current_condition(u_rr, equations)[1]
#  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

#  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
#end


@inline function max_abs_speed_naive2(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  return max_abs_speed_naive(u_ll, u_rr, 0, equations) * norm(normal_direction)
end


@inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)

  # Compute the necessary mean values
  rho_mean = ln_mean(rho_ll, rho_rr)
  # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
  # in exact arithmetic since
  #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
  #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
  inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  qv_avg = 0.5 * (qv_ll + qv_rr)
  ql_avg = 0.5 * (ql_ll + ql_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
  qd_avg = (1 - qv_avg - ql_avg)
  e = (qv_avg * L_00 + 
       (qd_avg * c_vd + qv_avg * c_vv + ql_avg * c_pl) * 
       inv_rho_p_mean * inv(qd_avg * R_d + qv_avg * R_v))

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = f1 * ( velocity_square_avg + e ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)
    f5 = f1 * qv_avg
    f6 = f1 * ql_avg
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * ( velocity_square_avg + e ) + 0.5 * (p_ll*v2_rr + p_rr*v2_ll)
    f5 = f1 * qv_avg
    f6 = f1 * ql_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6)
end

@inline function flux_ranocha_rain(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)

  # Compute the necessary mean values
  rho_mean = ln_mean(rho_ll, rho_rr)
  # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
  # in exact arithmetic since
  #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
  #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
  inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  qv_avg = 0.5 * (qv_ll + qv_rr)
  ql_avg = 0.5 * (ql_ll + ql_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
  qd_avg = (1 - qv_avg - ql_avg)
  T_interface = inv_rho_p_mean * inv(qd_avg * R_d + qv_avg * R_v)
  e = (qv_avg * L_00 + 
       (qd_avg * c_vd + qv_avg * c_vv + ql_avg * c_pl) * T_interface)
  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = f1 * ( velocity_square_avg + e ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)
    f5 = f1 * qv_avg
    f6 = f1 * ql_avg
  else
    rho_ql = rho_mean * ql_avg 
    Q_fall = rho_ql * fall_speed_rain(rho_ql, rho_ql, equations)
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * ( velocity_square_avg + e ) + 0.5 * (p_ll*v2_rr + p_rr*v2_ll)
    f5 = f1 * qv_avg
    f6 = f1 * ql_avg
    f4 += Q_fall * (c_pl * T_interface + velocity_square_avg)
    f1 += Q_fall
    f6 += Q_fall 
    f3 += Q_fall * v2_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6)
end

@inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)
  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  # Compute the necessary mean values
  rho_mean = ln_mean(rho_ll, rho_rr)
  # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
  # in exact arithmetic since
  #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
  #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
  inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  qv_avg = 0.5 * (qv_ll + qv_rr)
  ql_avg = 0.5 * (ql_ll + ql_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)
  qd_avg = (1 - qv_avg - ql_avg)
  e = (qv_avg * L_00 + 
       (qd_avg * c_vd + qv_avg * c_vv + ql_avg * c_pl) * 
       inv_rho_p_mean * inv(qd_avg * R_d + qv_avg * R_v))


  # Calculate fluxes depending on normal_direction
  f1 = rho_mean * (v1_avg * normal_direction[1] + v2_avg * normal_direction[2])
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]
  f4 = ( f1 * ( velocity_square_avg + e )
        + 0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll) )
  f5 = f1 * qv_avg
  f6 = f1 * ql_avg

  return SVector(f1, f2, f3, f4, f5, f6)
end


@inline function flux_shima_etal(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  qv_avg  = 1/2 * ( qv_ll +  qv_rr)
  ql_avg  = 1/2 * ( ql_ll +  ql_rr)
  p_avg = 1/2 * (p_ll + p_rr)
  qd_avg = (1 - qv_avg - ql_avg)
  e = (qv_avg * L_00 + 
       (qd_avg * c_vd + qv_avg * c_vv + ql_avg * c_pl) * 
       (p_avg / rho_avg) * inv(qd_avg * R_d + qv_avg * R_v))
  kin_avg = 1/2 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    pv1_avg = 1/2 * (p_ll*v1_rr + p_rr*v1_ll)
    f1 = rho_avg * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = f1 * e  + rho_avg*v1_avg*kin_avg + pv1_avg
    f5 = f1 * qv_avg
    f6 = f1 * ql_avg
  else
    pv2_avg = 1/2 * (p_ll*v2_rr + p_rr*v2_ll)
    f1 = rho_avg * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * e  + rho_avg*v1_avg*kin_avg + pv2_avg
    f5 = f1 * qv_avg
    f6 = f1 * ql_avg
  end

  return SVector(f1, f2, f3, f4, f5, f6)
end


@inline function flux_shima_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  qv_avg  = 1/2 * ( qv_ll +  qv_rr)
  ql_avg  = 1/2 * ( ql_ll +  ql_rr)
  p_avg = 1/2 * (p_ll + p_rr)
  qd_avg = (1 - qv_avg - ql_avg)
  e = (qv_avg * L_00 + 
       (qd_avg * c_vd + qv_avg * c_vv + ql_avg * c_pl) * 
       (p_avg / rho_avg) * inv(qd_avg * R_d + qv_avg * R_v))

  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]
  
  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  qv_avg  = 1/2 * ( qv_ll +  qv_rr)
  ql_avg  = 1/2 * ( ql_ll +  ql_rr)
  p_avg = 1/2 * (p_ll + p_rr)
  v_dot_n_avg = 1/2 * (v_dot_n_ll + v_dot_n_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)

   # Calculate fluxes depending on normal_direction
   f1 = rho_avg * v_dot_n_avg
   f2 = f1 * v1_avg + p_avg * normal_direction[1]
   f3 = f1 * v2_avg + p_avg * normal_direction[2]
   f4 = ( f1 * (velocity_square_avg + e)
         + 0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll))
   f5 = f1 * qv_avg
   f6 = f1 * ql_avg

  return SVector(f1, f2, f3, f4, f5, f6)
end


end # @muladd
