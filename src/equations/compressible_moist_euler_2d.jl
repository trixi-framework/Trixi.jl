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
  L_00::RealT # latent heat of evaporation  at 0 K
  a::RealT
end

function CompressibleMoistEulerEquations2D(;g= 9.81, RealT=Float64)
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
   L_00 = 3147620.0
   a = 360.0
   return CompressibleMoistEulerEquations2D{RealT}(p_0, c_pd, c_vd, R_d, c_pv, c_vv, R_v, c_pl,  g, kappa, gamma, L_00, a)
  end


varnames(::typeof(cons2cons), ::CompressibleMoistEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_E", "rho_qv", "rho_ql")
varnames(::typeof(cons2prim), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "p", "qv", "ql")
varnames(::typeof(cons2temp), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "T", "qv", "ql")
varnames(::typeof(cons2drypot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "drypottemp", "qv", "ql")
varnames(::typeof(cons2moistpot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "moistpottemp", "qv", "ql")
varnames(::typeof(cons2moist), ::CompressibleMoistEulerEquations2D) = ("qv", "ql", "rt", "T", "H", "aeqpottemp")
varnames(::typeof(cons2aeqpot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "aeqpottemp", "rv", "rt")


# Calculate 1D flux for a single point.
@inline function flux(u, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
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

# Calculate 1D flux for a single point in the normal direction.
# Note, this directional vector is not normalized.
@inline function flux(u, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
  ql = rho_ql / rho
  p, T = get_current_condition(u, equations)
  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  rho_v_normal = rho * v_normal 
  f1 = rho_v_normal 
  f2 = (rho_v_normal ) * v1 + p * normal_direction[1]
  f3 = (rho_v_normal ) * v2 + p * normal_direction[2]
  f4 = (rho_e + p) * v_normal 
  f5 = rho_v_normal * qv 
  f6 = (rho_v_normal ) * ql 
  return SVector(f1, f2, f3, f4, f5, f6)
end

# Slip-wall boundary condition 
# Determine the boundary numerical surface flux for a slip wall condition.
# Imposes a zero normal velocity at the wall.
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
  surface_flux_function, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, c_vd, c_vv = equations
  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_
  
  # rotate the internal solution state
  u_local = rotate_to_x(u_inner, normal, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent, p_local, qv_local, ql_local = cons2prim(u_local, equations)
  qd_local =  1 - qv_local - ql_local
  gamma = (qd_local * c_pd + qv_local * c_pv + ql_local * c_pl ) * inv(qd_local * c_vd + qv_local * c_vv + ql_local * c_pl)
  # Get the solution of the pressure Riemann problem
  # See Section 6.3.3 of
  # Eleuterio F. Toro (2009)
  # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
  if v_normal <= 0.0
  sound_speed = sqrt(gamma * p_local / rho_local) # local sound speed
  p_star = p_local * (1.0 + 0.5 * (gamma - 1) * v_normal / sound_speed)^(2.0 * gamma * inv(gamma-1))
  else # v_normal > 0.0
  A = 2.0 / ((gamma + 1) * rho_local)
  B = p_local * (gamma - 1) / (gamma + 1)
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

# Fix sign for structured mesh. 
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

# Rotate momentum flux. The same as in compressible Euler.
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


# Recreates the convergence test initial condition from compressible euler 2D.
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

# Recreates the convergence test source term from compressible euler 2D.
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

# Initial condition for the convergence analysis.
# Extends the convergence test to the system with gravity and phase change.
function initial_condition_convergence_test_moist(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations

  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  qv = 100/L_00
  ql = qv / 10

  mu = ((1 - qv - ql)*c_vd + qv*c_vv + ql*c_pl)

  T = (ini - 1) /mu + 10/c_vd + 40
  E = (mu*T + qv*L_00 + 1)

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_e = E * ini
  rho_qv = qv * ini 
  rho_ql = ql * ini
  
  return SVector(rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql)
end

# Source term for the convergence analysis.
# Extends the convergence test to the system with gravity and phase change.
@inline function source_terms_convergence_test_moist(u, x, t, equations::CompressibleMoistEulerEquations2D)
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
  
  qv = 100/L_00
  ql = qv / 10
  mu = ((1 - qv - ql) * c_vd + qv * c_vv + ql * c_pl)
  xi = ((1 - qv - ql) * R_d + qv * R_v)

  T = (rho - 1) /mu + 10/c_vd + 40
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
  du3 += rho_x + dp 
  du4 += dE + 2*dp 
  du5 += qv * rho_x
  du6 += ql * rho_x 

  return SVector(du1, du2, du3, du4, du5, du6)
end

# Gravity source term
@inline function source_terms_geopotential(u, x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack g = equations
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  tmp = rho_v2

  return SVector(zero(eltype(u)), zero(eltype(u)),
                 -g * rho, -g * tmp, 
                 zero(eltype(u)), zero(eltype(u)))
end


@inline function source_terms_geopotential(u, equations::CompressibleMoistEulerEquations2D)
  @unpack g = equations
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  tmp = rho_v2

  return SVector(zero(eltype(u)), zero(eltype(u)),
                 -g * rho, -g * tmp, 
                 zero(eltype(u)), zero(eltype(u)))
end

# Raylight damping sponge source term form A. Sridhar et al., 
#Large-eddy simulations with ClimateMachine: a new open-sourcecode for
#atmospheric simulations on GPUs and CPUs, 2 Oct 2021, doi: 10.5194/gmd-15-6259-2022,
#https://arxiv.org/abs/2110.00853 [physics.ao-ph] .
@inline function source_terms_nonhydrostatic_raylight_sponge(u, x, t, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  z = x[2]

  # relaxed background velocity
  vr1, vr2 = (10.0, 0.0)
  # damping threshold
  z_s = 9000.0
  # boundary top
  z_top = 16000.0
  # positive even power with default value 2
  gamma = 2.0
  #relaxation coefficient > 0
  alpha = 0.5

  tau_s = zero(eltype(u))
  if z > z_s 
    tau_s = alpha * sin(0.5 * (z-z_s) * inv(z_top - z_s))^(gamma)
  end

  return SVector(zero(eltype(u)), 
                 -tau_s * rho *(v1-vr1),
                 -tau_s * rho *(v2-vr2),
                 zero(eltype(u)), zero(eltype(u)), zero(eltype(u)))
end


# Source term with gravity and phase change
@inline function source_terms_moist_bubble(u, x, t, equations::CompressibleMoistEulerEquations2D)

  return source_terms_geopotential(u, equations) + source_terms_phase_change(u, equations)
end


# Calculate pressure and temperature from a state u.
@inline function get_current_condition(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_vd, R_d, c_vv, c_pv, R_v, c_pl, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  rho_qd = rho - rho_qv - rho_ql

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  
  # Inner energy
  rho_e = (rho_E - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  # Absolute temperature
  T = (rho_e - L_00 * rho_qv) / (rho_qd * c_vd + rho_qv * c_vv + rho_ql * c_pl)  
      
  # Pressure
  p = (rho_qd * R_d + rho_qv * R_v) * T

  return SVector(p, T)
end


# Calculate Q_ph for a state u.
# This source term models the phase chance between could water and vapor.
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


# Add the source containing Q_ph
@inline function source_terms_phase_change(u, equations::CompressibleMoistEulerEquations2D)

  Q_ph = phase_change_term(u, equations)
  
  return SVector(zero(eltype(u)), zero(eltype(u)), zero(eltype(u)),
                 zero(eltype(u)) , Q_ph, -Q_ph)
end


# Low Mach number approximate Riemann solver (LMARS) from 
# X. Chen, N. Andronova, B. Van Leer, J. E. Penner, J. P. Boyd, C. Jablonowski, S.
# Lin, A Control-Volume Model of the Compressible Euler Equations with a Vertical Lagrangian
# Coordinate Monthly Weather Review Vol. 141.7, pages 2526–2544, 2013, 
# https://journals.ametsoc.org/view/journals/mwre/141/7/mwr-d-12-00129.1.xml.
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


# Convert conservative variables to primitive.
@inline function cons2prim(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = get_current_condition(u, equations)[1]
  qv = rho_qv / rho
  ql = rho_ql / rho

  return SVector(rho, v1, v2, p, qv, ql)
end


# Convert conservative variables to primitive with 
# temperature instead of pressure.
@inline function cons2temp(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T = get_current_condition(u, equations)[2]
  qv = rho_qv / rho
  ql = rho_ql / rho

  return SVector(rho, v1, v2, T, qv, ql)
end

# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_pd, c_pv, c_pl, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p, T = get_current_condition(u, equations)
  v_square = v1^2 + v2^2
  rho_qd =rho-rho_qv-rho_ql

  # Work around if an individual density is zero
  # Thermodynamic entropy
  s_d = 0
  s_v = 0
  s_l = 0

  # Thermodynamic entropy
  if(rho_qd > 0.0)
    s_d = c_pd*log(T) - R_d*log(rho_qd*R_d*T)
  end
  if(rho_qv > 0.0)
    s_v = c_pv*log(T) - R_v*log(rho_qv*R_v*T)
  end
  if(rho_ql > 0.0)
    s_l = c_pl*log(T)
  end

  g_d = (c_pd - s_d)*T
  g_v = L_00 + (c_pv - s_v)*T
  g_l = (c_pl - s_l)*T


  w1 = g_d - 0.5 * v_square
  w2 = v1
  w3 = v2
  w4 = -1
  w5 = g_v-g_d
  w6 = g_l-g_d

  return inv(T) * SVector(w1, w2, w3, w4, w5, w6)
end


# Convert primitive to conservative variables.
@inline function prim2cons(prim, equations::CompressibleMoistEulerEquations2D)
  rho, v1, v2, p, qv, ql = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_qv = rho * qv
  rho_ql = rho * ql
  rho_E  = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end


# Convert conservative variables to primitive with 
# potential temperature instead of pressure.
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


# Convert conservative variables to primitive with 
# moist potential temperature instead of pressure.
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


# Convert conservative variables to moisture related variables.
@inline function cons2moist(u, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_pd, c_pv, c_pl, p_0 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  qv = rho_qv / rho
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

  pot1 = qv
  pot2 = ql
  pot3 = r_v + r_l
  pot4 = T
  pot5 = H
  pot6 = aequivalent_pottemp_thermodynamic(u, equations)

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


@inline function temperature(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_vd, R_d, c_vv, c_pv, R_v, c_pl, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  rho_qd = rho - rho_qv - rho_ql

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  
  # inner energy
  rho_e = (rho_E - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  # Absolute Temperature
  T = (rho_e - L_00 * rho_qv) / (rho_qd * c_vd + rho_qv * c_vv + rho_ql * c_pl)  
  return T
end


@inline function saturation_pressure(u, equations::CompressibleMoistEulerEquations2D)
  @unpack R_v = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  T = get_current_condition(u, equations)[2]
  p_v = rho_qv * R_v * T
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  H = p_v / p_vs 
  return H
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
 return p
end


@inline function density_pressure(u, equations::CompressibleMoistEulerEquations2D)
  rho = u[1]
 rho_times_p = rho * get_current_condition(u, equations)[1]
 return rho_times_p
end


# Calculate thermodynamic entropy for a conservative state `cons`.
# This is the dryspecific entropy multiplied by the dry air density.
@inline function entropy_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack c_vd, c_vv, c_pl, R_d, R_v = equations
  # Pressure
  p, T = get_current_condition(cons, equations)
  rho_qd =cons[1]-cons[5]-cons[6]
  rho_qv =cons[5]
  rho_ql =cons[6]
  # Thermodynamic entropy
  s_d = c_vd*log(T) - R_d*log(rho_qd*R_d)

  s_v = c_vv*log(T) - R_v*log(rho_qv*R_v)

  s_l = c_pl*log(T)
  
  return rho_qd*s_d + rho_qv*s_v + rho_ql*s_l
end


# Calculate mathematical entropy for a conservative state `cons`.
@inline function entropy_math(cons, equations::CompressibleMoistEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) 

  return S
end


# Default entropy is the mathematical entropy.
@inline entropy(cons, equations::CompressibleMoistEulerEquations2D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`.
@inline energy_total(cons, ::CompressibleMoistEulerEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`.
@inline function energy_kinetic(u, equations::CompressibleMoistEulerEquations2D)
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  return (rho_v1^2 + rho_v2^2) / (2 * rho)
end


# Calculate internal energy for a conservative state `cons`.
@inline function energy_internal(cons, equations::CompressibleMoistEulerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


# Calculate the dry potential temperature for a conservative state `cons`.
@inline function dry_pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, p_0, kappa = equations
  # Pressure
  p = get_current_condition(cons, equations)[1]
  # Potential temperature
  pot = p_0 * (p / p_0)^(1 - kappa) / (R_d * cons[1])

  return pot
end


# Calculate the moist potential temperature for a conservative state `cons`.
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


# Calculate the aequivalent potential temperature for a conservative state `cons`.
@inline function aequivalent_pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
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

  return aeq_pot
end


# Convert conservative variables to primitive varuables with 
# aequivalent potential temperature instead of pressure
# and mixing ratios innstead of specific contend.
@inline function cons2aeqpot(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, R_d, R_v, p_0, L_00 = equations
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


# Adjusted version of LLF dissipation from compressible euler.
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, c_vd, c_vv = equations
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)
  qd_ll = 1 - qv_ll - ql_ll
  qd_rr = 1 - qv_rr - ql_rr  
  # Get the density and gas gamma
  gamma_ll = (qd_ll * c_pd + qv_ll * c_pv + ql_ll *c_pl) * inv(qd_ll * c_vd + qv_ll * c_vv + ql_ll *c_pl)
  gamma_rr = (qd_rr * c_pd + qv_rr * c_pv + ql_rr *c_pl) * inv(qd_rr * c_vd + qv_rr * c_vv + ql_rr *c_pl)
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


# Adjusted version of lambda_max from compressible euler.
@inline function max_abs_speeds(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_pd, c_pv, c_pl, c_vd, c_vv = equations
  rho, v1, v2, p, qv, ql = cons2prim(u, equations)
  qd = 1 - qv - ql 

  gamma = (qd * c_pd + qv * c_pv + ql * c_pl ) * inv(qd * c_vd + qv * c_vv + ql * c_pl)
  c = sqrt(gamma * p / rho)

  return (abs(v1) + c, abs(v2) + c)
end


# Adjusted EC flux in a normal direction with R_q=0. This is based on
# A. Gouasmi, K. Duraisamy, S. M. Murman, Formulation of Entropy-Stable schemes for the
# multicomponent compressible Euler equations, 4 Feb 2020, doi:10.1016/j.cma.2020.112912,
# https://arxiv.org/abs/1904.00972 [math.NA].
@inline function flux_chandrashekar(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, R_v, c_vd, c_vv, c_pl, L_00 = equations
  R_q = 0
  # Unpack left and right state
  rho_ll, rho_v1_ll, rho_v2_ll, rho_E_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_E_rr, rho_qv_rr, rho_ql_rr = u_rr

  rho_qd_ll = rho_ll - rho_qv_ll - rho_ql_ll
  rho_qd_rr = rho_rr - rho_qv_rr - rho_ql_rr
  v1_ll = rho_v1_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_ll = rho_v2_ll / rho_ll
  v2_rr = rho_v2_rr / rho_rr

  # inner energy
  rho_e_ll = (rho_E_ll - 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll))
  rho_e_rr = (rho_E_rr - 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr))

  # Absolute Temperature
  T_ll = (rho_e_ll - L_00 * rho_qv_ll) / (rho_qd_ll * c_vd + rho_qv_ll * c_vv + rho_ql_ll * c_pl)
  T_rr = (rho_e_rr - L_00 * rho_qv_rr) / (rho_qd_rr * c_vd + rho_qv_rr * c_vv + rho_ql_rr * c_pl)

  # Compute the necessary mean values
  rho_qd_mean = 0
  rho_qv_mean = 0
  rho_ql_mean = 0
  inv_T_mean = 0
  if(!(rho_qd_ll==0.0) && !(rho_qd_rr==0.0))
    rho_qd_mean = ln_mean(rho_qd_ll, rho_qd_rr)
  end
  if(!(rho_qv_ll==0.0) && !(rho_qv_rr==0.0))
    rho_qv_mean = ln_mean(rho_qv_ll, rho_qv_rr)
  end
  if(!(rho_ql_ll==0.0) && !(rho_ql_rr==0.0))
    rho_ql_mean = ln_mean(rho_ql_ll, rho_ql_rr)
  end
  if(!(inv(T_ll)==0.0) && !(inv(T_rr)==0.0))
    inv_T_mean = inv_ln_mean(inv(T_ll), inv(T_rr))
  end
  


  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  v1_square_avg = 0.5 * (v1_ll^2 + v1_rr^2)
  v2_square_avg = 0.5 * (v2_ll^2 + v2_rr^2)
  rho_qd_avg = 0.5 * (rho_qd_ll + rho_qd_rr)
  rho_qv_avg = 0.5 * (rho_qv_ll + rho_qv_rr)
  rho_ql_avg = 0.5 * (rho_ql_ll + rho_ql_rr)
  inv_T_avg  = 0.5 * (inv(T_ll) + inv(T_rr))
  v_dot_n_avg = normal_direction[1]*v1_avg + normal_direction[2]*v2_avg

  p_int = inv(inv_T_avg) * (R_d*rho_qd_avg + R_v*rho_qv_avg + R_q*rho_ql_avg) 
  K_avg = 0.5 *(v1_square_avg + v2_square_avg)

  f_1d = rho_qd_mean * v_dot_n_avg
  f_1v = rho_qv_mean * v_dot_n_avg
  f_1l = rho_ql_mean * v_dot_n_avg
  f1 = f_1d + f_1v + f_1l
  f2 = f1*v1_avg + normal_direction[1]*p_int
  f3 = f1*v2_avg + normal_direction[2]*p_int
  f4 = ((c_vd*inv_T_mean - K_avg) * f_1d + (L_00 + c_vv*inv_T_mean - K_avg) * f_1v +
          (c_pl*inv_T_mean - K_avg) * f_1l + v1_avg*f2 + v2_avg*f3 )

  return SVector(f1, f2, f3, f4, f_1v, f_1l)
end


end # @muladd
