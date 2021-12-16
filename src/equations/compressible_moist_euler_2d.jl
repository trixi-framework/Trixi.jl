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
varnames(::typeof(cons2aeqpot), ::CompressibleMoistEulerEquations2D) = ("rho", "v1", "v2", "aeqpottemp", "rv", "rt")

struct AtmossphereLayers{RealT<:Real}
  equations::CompressibleMoistEulerEquations2D
  # structure:  1--> i-layer (z = total_hight/precision *(i-1)),  2--> rho, rho_theta, rho_qv, rho_ql
  LayerData::Matrix{RealT}
  total_hight::RealT
  preciseness::Int
  layers::Int
  ground_state::NTuple{2, RealT}
  equivalentpotential_temperature::RealT
  mixing_ratios::NTuple{2, RealT}
end

function AtmossphereLayers(equations ; total_hight=10000.0, preciseness=10, ground_state=(1.4, 100000.0), equivalentpotential_temperature=320, mixing_ratios=(0.02, 0.02), RealT=Float64)
  @unpack kappa, p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl = equations
  rho0, p0 = ground_state
  r_t0, r_v0 = mixing_ratios
  theta_e0 = equivalentpotential_temperature

  rho_qv0 = rho0 * r_v0
  T0 = theta_e0
  y0 = [p0, rho0, T0, r_t0, r_v0, rho_qv0, theta_e0]

  n = convert(Int, total_hight / preciseness)
  dz = 0.01
  LayerData = zeros(RealT, n+1, 4)

  F = generate_function_of_y(dz, y0, r_t0, theta_e0, equations)
  sol = nlsolve(F, y0)
  p, rho, T, r_t, r_v, rho_qv, theta_e = sol.zero
  
  rho_d = rho / (1 + r_t)
  rho_ql = rho - rho_d - rho_qv
  kappa_M=(R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
  rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 

  LayerData[1, :] = [rho, rho_theta, rho_qv, rho_ql]
   for i in (1:n)
    y0 = deepcopy(sol.zero)
    dz = preciseness
    F = generate_function_of_y(dz, y0, r_t0, theta_e0, equations)
    sol = nlsolve(F, y0)
    p, rho, T, r_t, r_v, rho_qv, theta_e = sol.zero
    
    rho_d = rho / (1 + r_t)
    rho_ql = rho - rho_d - rho_qv
    kappa_M=(R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
    rho_theta = rho * (p0 / p)^kappa_M * T * (1 + (R_v / R_d) *r_v) / (1 + r_t) 

    LayerData[i+1, :] = [rho, rho_theta, rho_qv, rho_ql]
   end
  
  return AtmossphereLayers{RealT}(equations, LayerData, total_hight, dz, n, ground_state, theta_e0, mixing_ratios)
end

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
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = get_current_condition(u, equations)[1]

  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = (rho_E + p) * v1
    f5 = rho_qv * v1
    f6 = rho_ql * v1
  else
    f1 = rho_v2 
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = (rho_E + p) * v2 
    f5 = rho_qv * v2
    f6 = rho_ql * v2
  end
  return SVector(f1, f2, f3, f4, f5, f6)
end


@inline function flux(u, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
  rho_e = u[4]
  rho, v1, v2, p, qv, ql = cons2prim(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  rho_v_normal = rho * v_normal
  f1 = rho_v_normal
  f2 = rho_v_normal * v1 + p * normal_direction[1]
  f3 = rho_v_normal * v2 + p * normal_direction[2]
  f4 = (rho_e + p) * v_normal
  f5 = rho_v_normal * qv 
  f6 = rho_v_normal * ql
  return SVector(f1, f2, f3, f4, f5, f6)
end


function boundary_condition_slip_wall(u_inner, orientation::Integer, direction, x, t,
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

function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, x, t,
  surface_flux_function, equations::CompressibleMoistEulerEquations2D)

  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_

  # rotate the internal solution state
  u_local = rotate_to_x(u_inner, normal, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent, p_local, rho_qv_local, rho_ql_local = cons2prim(u_local, equations)

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


function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector, direction, x, t,
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


function initial_condition_moist_bubble(x, t, equations::CompressibleMoistEulerEquations2D, AtmosphereLayers)
  @unpack LayerData, preciseness, total_hight = AtmosphereLayers
  dz = preciseness
  z = x[2] 
  if (z > total_hight && !(isapprox(z, total_hight)))
    error("The atmossphere does not match the simulation domain")
  end
  n = convert(Int, floor(z/dz)) + 1
  z_l = (n-1) * dz
  (rho_l, rho_theta_l, rho_qv_l, rho_ql_l) = LayerData[n, :]
  z_r = n * dz
  if (z_l == total_hight)
    z_r = z_l + dz 
    n = n-1
  end
  (rho_r, rho_theta_r, rho_qv_r, rho_ql_r) = LayerData[n+1, :]
  rho = (rho_r * (z - z_l) + rho_l * (z_r - z)) / dz
  rho_theta = rho * (rho_theta_r / rho_r * (z - z_l) + rho_theta_l / rho_l * (z_r - z)) / dz
  rho_qv = rho * (rho_qv_r / rho_r * (z - z_l) + rho_qv_l / rho_l * (z_r - z)) / dz
  rho_ql = rho * (rho_ql_r / rho_r * (z - z_l) + rho_ql_l / rho_l * (z_r - z)) / dz

  rho, rho_e, rho_qv, rho_ql = PerturbMoistProfile(x, rho, rho_theta, rho_qv, rho_ql, equations::CompressibleMoistEulerEquations2D)

  v1 = 0
  v2 = 0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_E = rho_e + 1/2 * rho *(v1^2 + v2^2)


  return SVector(rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql)
end


function PerturbMoistProfile(x, rho, rho_theta, rho_qv, rho_ql, equations::CompressibleMoistEulerEquations2D)
  @unpack kappa, p_0, c_pd, c_vd, c_pv, c_vv, R_d, R_v, c_pl, L_00 = equations
  xc = 0
  zc = 2000
  rc = 2000
  Δθ = 2
  
  r = sqrt((x[1] - xc)^2 + (x[2] - zc)^2)
  rho_d = rho - rho_qv - rho_ql
  kappa_M = (R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
  p_loc = p_0 *(R_d * rho_theta / p_0)^(1/(1-kappa_M))
  T_loc = p_loc / (R_d * rho_d + R_v * rho_qv)
  rho_e = (c_vd * rho_d + c_vv * rho_qv + c_pl * rho_ql) * T_loc + L_00 * rho_qv

  p_v = rho_qv * R_v * T_loc
  p_d = p_loc - p_v
  T_C = T_loc - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  H = p_v / p_vs
  r_v = rho_qv / rho_d
  r_l = rho_ql / rho_d
  r_t = r_v + r_l

  # Aequivalentpotential temperature
  a=T_loc * (p_0 / p_d)^(R_d / (c_pd + r_t * c_pl))
  b=H^(- r_v * R_v /c_pd)
  L_v = L_00 + (c_pv - c_pl) * T_loc
  c=exp(L_v * r_v / ((c_pd + r_t * c_pl) * T_loc))
  aeq_pot = (a * b *c)

  # Assume pressure stays constant
  if (r < rc && Δθ > 0) 
    θ_dens = rho_theta / rho * (p_loc / p_0)^(kappa_M - kappa)
    θ_dens_new = θ_dens * (1 + Δθ * cospi(0.5*r/rc)^2 / 300)
    rt =(rho_qv + rho_ql) / rho_d 
    rv = rho_qv / rho_d
    θ_loc = θ_dens_new * (1 + rt)/(1 + (R_v / R_d) * rv)
    if rt > 0 
      while true 
        T_loc = θ_loc * (p_loc / p_0)^kappa
        T_C = T_loc - 273.15
        # SaturVapor
        pvs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
        rho_d_new = (p_loc - pvs) / (R_d * T_loc)
        rvs = pvs / (R_v * rho_d_new * T_loc)
        θ_new = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rvs)
        if abs(θ_new-θ_loc) <= θ_loc * 1.0e-12
          break
        else
          θ_loc=θ_new
        end
      end
    else
      rvs = 0
      T_loc = θ_loc * (p_loc / p_0)^kappa
      rho_d_new = p_loc / (R_d * T_loc)
      θ_new = θ_dens_new * (1 + rt) / (1 + (R_v / R_d) * rvs)
    end
    rho_qv = rvs * rho_d_new
    rho_ql = (rt - rvs) * rho_d_new
    rho = rho_d_new * (1 + rt)
    rho_d = rho - rho_qv - rho_ql
    kappa_M = (R_d * rho_d + R_v * rho_qv) / (c_pd * rho_d + c_pv * rho_qv + c_pl * rho_ql)
    rho_theta = rho * θ_dens_new * (p_loc / p_0)^(kappa - kappa_M)
    rho_e = (c_vd * rho_d + c_vv * rho_qv + c_pl * rho_ql) * T_loc + L_00 * rho_qv
  end
  return SVector(rho, rho_e, rho_qv, rho_ql)
end


function moist_state(y, dz, y0, r_t0, theta_e0, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, g, c_pd, c_pv, c_vd, c_vv, R_d, R_v, c_pl, L_00 = equations
  (p, rho, T, r_t, r_v, rho_qv, theta_e) = y
  p0 = y0[1]

  F = zeros(7,1)
  rho_d = rho / (1 + r_t)
  p_d = R_d * rho_d * T
  T_C = T - 273.15
  p_vs = 611.2 * exp(17.62 * T_C / (243.12 + T_C))
  L = L_00 - (c_pl - c_pv) * T

  F[1] = (p - p0) / dz + g * rho 
  F[2] = p - (R_d * rho_d + R_v * rho_qv) * T
  # H = 1 is assumed
  F[3] = (theta_e - T * (p_d / p_0)^(-R_d / (c_pd + c_pl * r_t)) *
         exp(L * r_v / ((c_pd + c_pl * r_t) * T)))
  F[4] = r_t - r_t0
  F[5] = rho_qv - rho_d * r_v
  F[6] = theta_e - theta_e0
  a = p_vs / (R_v * T) - rho_qv
  b = rho - rho_qv - rho_d
  F[7] = a+b-sqrt(a*a+b*b)

  return F
end


function generate_function_of_y(dz, y0, r_t0, theta_e0, equations::CompressibleMoistEulerEquations2D)
  function function_of_y(y)
    return moist_state(y, dz, y0, r_t0, theta_e0, equations)
  end
end


function initial_density_current(x, t, equations::CompressibleMoistEulerEquations2D)
  @unpack p_0, kappa, g, c_pd, c_vd, R_d, R_v = equations
  xc = 0
  zc = 3000
  rx = 4000
  rz = 2000
  θ_ref = 300
  θ_c = -15
  Δθ = 0

  r = sqrt((x[1] - xc) * rx^(-2) +(x[2] - zc) * rz^(-2))

  if(r <= 1)
  Δθ = 0.5 * θ_c * (1 + cos( pi * r))
  end

  return SVector(rho, rho_v1, rho_v2, rho_E,  rho_qv, rho_ql)
end


# Warm bubble test from paper:
# Wicker, L. J., and W. C. Skamarock, 1998: A time-splitting scheme
# for the elastic equations incorporating second-order Runge–Kutta
# time differencing. Mon. Wea. Rev., 126, 1992–1999.
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
  rho = p_0 / (R_d * θ) * (π_exner)^(c_vd / R_d) # density
  p = p_0 * (1-kappa * g * x[2] / (R_d * θ_ref))^(c_pd / R_d)
  T = p / (R_d * rho)

  v1 = 0
  v2 = 0
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_E = rho * c_vd * T + 1/2 * rho * (v1^2 + v2^2)  
  return SVector(rho, rho_v1, rho_v2, rho_E, zero(eltype(g)) ,zero(eltype(g)))
end


@inline function source_terms_warm_bubble(u, x, t, equations::CompressibleMoistEulerEquations2D)
  return source_terms_geopotential(u, equations)
end


@inline function source_terms_geopotential(u, equations::CompressibleMoistEulerEquations2D)
  du3 = -equations.g * u[1]
  du4 = -equations.g * u[3]
  
  return SVector(zero(eltype(u)), zero(eltype(u)), du3, 
                 du4, zero(eltype(u)), zero(eltype(u)))
end

function source_terms_moist_air(du, u, equations::CompressibleMoistEulerEquations2D, dg, cache)
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_local = u[:, i, j ,element]
      #x_local = get_node_coords(cache.elements.node_coordinates, equations, dg, i, j, element)
      #z = x_local[2]
      _, _, W_f , _, _, E_v = get_moist_profile(u, equations)
      Q_v = ground_vapor_source_term(u, equations)
      Q_ph = phase_change_term(u, equations)
      du[3, i, j, element] += -equations.g * u_local[1]
      du[4, i, j, element] += -equations.g * (u_local[3] - u_local[1] * u_local[6] * W_f) + E_v * Q_v 
      du[5, i, j, element] += Q_v + Q_ph
      du[6, i, j, element] += -Q_ph
    end
  end
  return nothing
end

@inline function source_terms_moist_bubble(u, x, t, equations::CompressibleMoistEulerEquations2D)
  return source_terms_geopotential(u, equations) + source_terms_phase_change(u, equations)
end

@inline function source_terms_phase_change(u, equations::CompressibleMoistEulerEquations2D)
  Q_ph = phase_change_term(u, equations)

  return SVector(zero(eltype(u)), zero(eltype(u)), zero(eltype(u)),
                 zero(eltype(u)) , Q_ph , -Q_ph)
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


  # inner energy
  rho_e = (rho_E - 0.5 * (rho_v1*v1 + rho_v2*v2))
  e = rho_e / rho
  # Absolute Temperature
  T = inv(rho_qd * c_vd + rho_qv * c_vv + rho_ql * c_pl) * (rho_e - L_00 * rho_qv)
      

  # Pressure
  p = (rho_qd * R_d + rho_qv * R_v) * T

  # Parametrisation by Frisius and Wacker
  W_f = - c_r * gm * inv(6) * (rho_ql * inv(pi * rho_w * N_0r))^(1/8)

  # Energy factors for source terms
  e_l = equations.c_pl * T
  E_l = (rho_E - rho_e) / rho + e_l

  h_v = c_pv * T + L_00
  E_v = (rho_E - rho_e) / rho + h_v


  return SVector(p, T, W_f, e, E_l, E_v)
end


@inline function get_current_condition(u, equations::CompressibleMoistEulerEquations2D)
  @unpack c_vd, R_d, c_vv, c_pv, R_v, c_pl, L_00 = equations
  rho, rho_v1, rho_v2, rho_E, rho_qv, rho_ql = u
  rho_qd = rho - rho_qv - rho_ql

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  
  # inner energy
  rho_e = (rho_E - 0.5 * (rho_v1*v1 + rho_v2*v2))

  # Absolute Temperature
  T = inv(rho_qd * c_vd + rho_qv * c_vv + rho_ql * c_pl) * (rho_e - L_00 * rho_qv)
      
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
function ground_vapor_source_term(u, equations::CompressibleMoistEulerEquations2D)

  return zero(eltype(u))
end


# This source term models the condensed water falling down as rain
function condensation_source_term(u, equations::CompressibleMoistEulerEquations2D)

  return zero(eltype(u))
end


@inline function FluxLMARS(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  @unpack a = equations
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll, rho_qv_ll, rho_ql_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr, rho_qv_rr, rho_ql_rr = u_rr
  
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr = cons2prim(u_rr, equations)
  
  # Compute the necessary interface flux components

  rho_mean = 0.5 * (rho_ll + rho_rr) # TODO why choose the mean value here?

  if orientation == 1

    beta = 1 # diffusion parameter <= 1 

    v_interface = 0.5 * (v1_rr + v1_ll) - beta * inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - beta * 0.5 * rho_mean * a * (v1_rr - v1_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = rho_e_ll + p_ll
      f5 = rho_qv_ll
      f6 = rho_ql_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = rho_e_rr + p_rr
      f5 = rho_qv_rr
      f6 = rho_ql_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 1, 0, 0, 0, 0) * p_interface

  else # orientation = 2

    v_interface = 0.5 * (v2_rr + v2_ll) - inv(2 * rho_mean * a) * (p_rr - p_ll)
    p_interface = 0.5 * (p_rr + p_ll) - 0.5 * rho_mean * a * (v2_rr - v2_ll)

    if (v_interface > 0)
      f1 = rho_ll
      f2 = rho_v1_ll
      f3 = rho_v2_ll
      f4 = rho_e_ll + p_ll
      f5 = rho_qv_ll
      f6 = rho_ql_ll
    else
      f1 = rho_rr
      f2 = rho_v1_rr
      f3 = rho_v2_rr
      f4 = rho_e_rr + p_rr
      f5 = rho_qv_rr
      f6 = rho_ql_rr
    end

    flux = SVector(f1, f2, f3, f4, f5, f6) * v_interface + SVector(0, 0, 1, 0, 0 ,0 ) * p_interface
  end

  return flux
end


@inline function FluxLMARS(u_ll, u_rr, normal_direction::AbstractVector , equations::CompressibleMoistEulerEquations2D)
  @unpack a = equations
  # Unpack left and right state
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll, qv_ll, ql_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr, qv_rr, ql_rr  = cons2prim(u_rr, equations)

  v_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] 
  v_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  # diffusion parameter <= 1 
  beta = 1 

  # Compute the necessary interface flux components
  rho_mean = 0.5*(rho_ll + rho_rr) # TODO why choose the mean value here?
  norm_ = norm(normal_direction)

  rho = 0.5 * (rho_ll + rho_rr)
  p_interface = 0.5 * (p_ll + p_rr) - 0.5 * a * rho * (v_rr - v_ll) / norm_
  v_interface = 0.5 * (v_ll + v_rr) - 1 / (2 * a * rho) * (p_rr - p_ll) * norm_

  if (v_interface > 0)
    f1, f2, f3, f4, f5, f6 = u_ll
    f5 += p_ll
  else
    f1, f2, f3, f4, f5, f6 = u_rr
    f5 += p_rr
  end

  flux = (SVector(f1, f2, f3, f4, f5, f6) * v_interface + 
          SVector(0, normal_direction[1], normal_direction[2], 0, 0, 0) * p_interface)

  return flux
end


@inline function max_abs_speeds(u, equations::CompressibleMoistEulerEquations2D)
  rho, v1, v2, p, qv, ql = cons2prim(u, equations)
  #@info(u)
  #@info(cons2prim(u, equations))

  c = sqrt(equations.gamma * p / rho)
  return abs(v1) + c, abs(v2) + c
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


@inline function density_dry(u, equations::CompressibleMoistEulerEquations2D)
  rho_qd = u[1] - (u[5] + u[6])
  return rho_qd
 end


 @inline function density_vapor(u, equations::CompressibleMoistEulerEquations2D)
  rho_qv = u[5] 
  return rho_qv
 end


 @inline function density_liqid(u, equations::CompressibleMoistEulerEquations2D)
  rho_ql = u[6]
  return rho_ql
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

@inline function pottemp_thermodynamic(cons, equations::CompressibleMoistEulerEquations2D)
  @unpack R_d, p_0, kappa = equations
  # Pressure
  p = get_moist_profile(cons, equations)[1]

  # Potential temperature
  pot = p_0 * (p / p_0)^(1 - kappa) / (R_d * cons[1])

  return pot
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


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
# TODO: This doesn't really use the `orientation` - should it?
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleMoistEulerEquations2D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2)
  p_ll = get_current_condition(u_ll, equations)[1]
  c_ll = sqrt(equations.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2)
  p_rr = get_current_condition(u_rr, equations)[1]
  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end


@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleMoistEulerEquations2D)
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


"""
    function flux_shima_etal(u_ll, u_rr, orientation, equation::CompressibleMoistEquations2D)
This flux is is a modification of the original kinetic energy preserving two-point flux by
Kuya, Totani and Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
  [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)
The modification is in the energy flux to guarantee pressure equilibrium and was developed by
Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
  Preventing spurious pressure oscillations in split convective form discretizations for
  compressible flows
"""
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
