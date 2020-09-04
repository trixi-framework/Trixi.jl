
@doc raw"""
    IdealGlmMhdEquations3D

The ideal compressible GLM-MHD equations in three space dimensions.
"""
mutable struct IdealGlmMhdEquations3D <: AbstractIdealGlmMhdEquations{3, 9}
  gamma::Float64
  c_h::Float64 # GLM cleaning speed
end

function IdealGlmMhdEquations3D()
  gamma = parameter("gamma", 1.4)
  c_h = 0.0   # GLM cleaning wave speed
  IdealGlmMhdEquations3D(gamma, c_h)
end


get_name(::IdealGlmMhdEquations3D) = "IdealGlmMhdEquations3D"
have_nonconservative_terms(::IdealGlmMhdEquations3D) = Val(true)
varnames_cons(::IdealGlmMhdEquations3D) = @SVector ["rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi"]
varnames_prim(::IdealGlmMhdEquations3D) = @SVector ["rho", "v1", "v2", "v3", "p", "B1", "B2", "B3", "psi"]
default_analysis_quantities(::IdealGlmMhdEquations3D) = (:l2_error, :linf_error, :dsdu_ut,
                                                         :l2_divb, :linf_divb)


# Set initial conditions at physical location `x` for time `t`
function initial_conditions_constant(x, t, equation::IdealGlmMhdEquations3D)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_v3 = -0.5
  rho_e = 50.0
  B1 = 3.0
  B2 = -1.2
  B3 = 0.5
  psi = 0.0
  return @SVector [rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi]
end

function initial_conditions_convergence_test(x, t, equation::IdealGlmMhdEquations3D)
  # Alfvén wave in three space dimensions
  # Altmann thesis http://dx.doi.org/10.18419/opus-3895
  # domain must be set to [-1, 1]^3, γ = 5/3
  p = 1
  omega = 2*pi # may be multiplied by frequency
  # r: length-variable = length of computational domain
  r = 2
  # e: epsilon = 0.2
  e = 0.2
  nx  = 1 / sqrt(r^2 + 1)
  ny  = r / sqrt(r^2 + 1)
  sqr = 1
  Va  = omega / (ny * sqr)
  phi_alv = omega / ny * (nx * (x[1] - 0.5*r) + ny * (x[2] - 0.5*r)) - Va * t

  rho = 1.
  v1  = -e*ny*cos(phi_alv) / rho
  v2  =  e*nx*cos(phi_alv) / rho
  v3  =  e *  sin(phi_alv) / rho
  B1  = nx -rho*v1*sqr
  B2  = ny -rho*v2*sqr
  B3  =    -rho*v3*sqr
  psi = 0

  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equation)
end

function initial_conditions_ec_test(x, t, equation::IdealGlmMhdEquations3D)
  # Adapted MHD version of the weak blast wave from Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Same discontinuity in the velocities but with magnetic fields
  # Set up polar coordinates
  inicenter = (0, 0, 0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  z_norm = x[3] - inicenter[3]
  r = sqrt(x_norm^2 + y_norm^2 + z_norm^2)
  phi = atan(y_norm, x_norm)
  theta = iszero(r) ? 0.0 : acos(z_norm / r)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos(phi) * sin(theta)
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin(phi) * sin(theta)
  v3  = r > 0.5 ? 0.0 : 0.1882 * cos(theta)
  p   = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, v2, v3, p, 1.0, 1.0, 1.0, 0.0), equation)
end

function initial_conditions_orszag_tang(x, t, equation::IdealGlmMhdEquations3D)
  # setup taken from Table 4 of Bohm et al. JCP article (2018) DOI: 10.1016/j.jcp.2018.06.027
  # domain must be [0, 1]^3 , γ = 5/3
  rho = 25.0 / (36.0 * pi)
  v1 = -sin(2.0*pi*x[3])
  v2 =  sin(2.0*pi*x[1])
  v3 =  sin(2.0*pi*x[2])
  p = 5.0 / (12.0 * pi)
  B1 = -sin(2.0*pi*x[3]) / (4.0*pi)
  B2 =  sin(4.0*pi*x[1]) / (4.0*pi)
  B3 =  sin(4.0*pi*x[2]) / (4.0*pi)
  psi = 0.0
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equation)
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(ut, u, x, element_id, t, n_nodes, equation::IdealGlmMhdEquations3D)


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equation::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  p = (equation.gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - mag_en - 0.5*psi^2)
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1*v1 + p + mag_en - B1^2
    f3 = rho_v1*v2 - B1*B2
    f4 = rho_v1*v3 - B1*B3
    f5 = (rho_e + p + mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B1
    f6 = equation.c_h*psi
    f7 = v1*B2 - v2*B1
    f8 = v1*B3 - v3*B1
    f9 = equation.c_h*B1
  elseif orientation == 2
    f1 = rho_v2
    f2 = rho_v2*v1 - B2*B1
    f3 = rho_v2*v2 + p + mag_en - B2^2
    f4 = rho_v2*v3 - B2*B3
    f5 = (rho_e + p + mag_en)*v2 - B2*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B2
    f6 = v2*B1 - v1*B2
    f7 = equation.c_h*psi
    f8 = v2*B3 - v3*B2
    f9 = equation.c_h*B2
  else
    f1 = rho_v3
    f2 = rho_v3*v1 - B3*B1
    f3 = rho_v3*v2 - B3*B2
    f4 = rho_v3*v3 + p + mag_en - B3^2
    f5 = (rho_e + p + mag_en)*v3 - B3*(v1*B1 + v2*B2 + v3*B3) + equation.c_h*psi*B3
    f6 = v3*B1 - v1*B3
    f7 = v3*B2 - v2*B3
    f8 = equation.c_h*psi
    f9 = equation.c_h*B3
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate the nonconservative terms from Powell and Galilean invariance
# OBS! This is scaled by 1/2 becuase it will cancel later with the factor of 2 in dsplit_transposed
@inline function calcflux_twopoint_nonconservative!(f1, f2, f3, u, element_id, equation::IdealGlmMhdEquations3D, dg)
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = get_node_vars(u, dg, i, j, k, element_id)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho

    # Powell nonconservative term: Φ^Pow = (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    phi_pow = 0.5 * SVector(0, B1, B2, B3, v1*B1 + v2*B2 + v3*B3, v1, v2, v3, 0)

    # Galilean nonconservative term: Φ^Gal_{1,2} = (0, 0, 0, 0, ψ v_{1,2}, 0, 0, 0, v_{1,2})
    # x-direction
    phi_gal_x = 0.5 * SVector(0, 0, 0, 0, v1*psi, 0, 0, 0, v1)
    # y-direction
    phi_gal_y = 0.5 * SVector(0, 0, 0, 0, v2*psi, 0, 0, 0, v2)
    # z-direction
    phi_gal_z = 0.5 * SVector(0, 0, 0, 0, v3*psi, 0, 0, 0, v3)

    # add both nonconservative terms into the volume
    for l in 1:nnodes(dg)
      _, _, _, _, _, B1, _, _, psi = get_node_vars(u, dg, l, j, k, element_id)
      for v in 1:nvariables(dg)
        f1[v, l, i, j, k] += phi_pow[v] * B1 + phi_gal_x[v] * psi
      end
      _, _, _, _, _, _, B2, _, psi = get_node_vars(u, dg, i, l, k, element_id)
      for v in 1:nvariables(dg)
        f2[v, l, i, j, k] += phi_pow[v] * B2 + phi_gal_y[v] * psi
      end
      _, _, _, _, _, _, _, B3, psi = get_node_vars(u, dg, i, j, l, element_id)
      for v in 1:nvariables(dg)
        f3[v, l, i, j, k] += phi_pow[v] * B3 + phi_gal_z[v] * psi
      end
    end
  end
end


"""
    flux_derigs_etal(u_ll, u_rr, orientation, equation::IdealGlmMhdEquations3D)

Entropy conserving two-point flux by Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations
[DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(u_ll, u_rr, orientation, equation::IdealGlmMhdEquations3D)
  # Unpack left and right states to get velocities, pressure, and inverse temperature (called beta)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  p_ll = (equation.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)
  p_rr = (equation.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)
  beta_ll = 0.5*rho_ll/p_ll
  beta_rr = 0.5*rho_rr/p_rr
  # for convenience store v⋅B
  vel_dot_mag_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  vel_dot_mag_rr = v1_rr*B1_rr + v2_rr*B2_rr + v3_rr*B3_rr

  # Compute the necessary mean values needed for either direction
  rho_avg  = 0.5*(rho_ll+rho_rr)
  rho_mean = ln_mean(rho_ll,rho_rr)
  beta_mean = ln_mean(beta_ll,beta_rr)
  beta_avg = 0.5*(beta_ll+beta_rr)
  v1_avg = 0.5*(v1_ll+v1_rr)
  v2_avg = 0.5*(v2_ll+v2_rr)
  v3_avg = 0.5*(v3_ll+v3_rr)
  p_mean = 0.5*rho_avg/beta_avg
  B1_avg = 0.5*(B1_ll+B1_rr)
  B2_avg = 0.5*(B2_ll+B2_rr)
  B3_avg = 0.5*(B3_ll+B3_rr)
  psi_avg = 0.5*(psi_ll+psi_rr)
  vel_norm_avg = 0.5*(vel_norm_ll+vel_norm_rr)
  mag_norm_avg = 0.5*(mag_norm_ll+mag_norm_rr)
  vel_dot_mag_avg = 0.5*(vel_dot_mag_ll+vel_dot_mag_rr)

  # Calculate fluxes depending on orientation with specific direction averages
  if orientation == 1
    f1 = rho_mean*v1_avg
    f2 = f1*v1_avg + p_mean + 0.5*mag_norm_avg - B1_avg*B1_avg
    f3 = f1*v2_avg - B1_avg*B2_avg
    f4 = f1*v3_avg - B1_avg*B3_avg
    f6 = equation.c_h*psi_avg
    f7 = v1_avg*B2_avg - v2_avg*B1_avg
    f8 = v1_avg*B3_avg - v3_avg*B1_avg
    f9 = equation.c_h*B1_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B1_avg = 0.5*(B1_ll*psi_ll + B1_rr*psi_rr)
    v1_mag_avg = 0.5*(v1_ll*mag_norm_ll + v1_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equation.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v1_mag_avg +
          B1_avg*vel_dot_mag_avg - equation.c_h*psi_B1_avg)
  elseif orientation == 2
    f1 = rho_mean*v2_avg
    f2 = f1*v1_avg - B2_avg*B1_avg
    f3 = f1*v2_avg + p_mean + 0.5*mag_norm_avg - B2_avg*B2_avg
    f4 = f1*v3_avg - B2_avg*B3_avg
    f6 = v2_avg*B1_avg - v1_avg*B2_avg
    f7 = equation.c_h*psi_avg
    f8 = v2_avg*B3_avg - v3_avg*B2_avg
    f9 = equation.c_h*B2_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B2_avg = 0.5*(B2_ll*psi_ll + B2_rr*psi_rr)
    v2_mag_avg = 0.5*(v2_ll*mag_norm_ll + v2_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equation.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v2_mag_avg +
          B2_avg*vel_dot_mag_avg - equation.c_h*psi_B2_avg)
  else
    f1 = rho_mean*v3_avg
    f2 = f1*v1_avg - B3_avg*B1_avg
    f3 = f1*v2_avg - B3_avg*B2_avg
    f4 = f1*v3_avg + p_mean + 0.5*mag_norm_avg - B3_avg*B3_avg
    f6 = v3_avg*B1_avg - v1_avg*B3_avg
    f7 = v3_avg*B2_avg - v2_avg*B3_avg
    f8 = equation.c_h*psi_avg
    f9 = equation.c_h*B3_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B3_avg = 0.5*(B3_ll*psi_ll + B3_rr*psi_rr)
    v3_mag_avg = 0.5*(v3_ll*mag_norm_ll + v3_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equation.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v3_mag_avg +
          B3_avg*vel_dot_mag_avg - equation.c_h*psi_B3_avg)
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


function flux_lax_friedrichs(u_ll, u_rr, orientation, equation::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate velocities and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  cf_ll = calc_fast_wavespeed(u_ll, orientation, equation)
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  cf_rr = calc_fast_wavespeed(u_rr, orientation, equation)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  λ_max = max(v_mag_ll, v_mag_rr) + max(cf_ll, cf_rr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (rho_rr    - rho_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (rho_v1_rr - rho_v1_ll)
  f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (rho_v2_rr - rho_v2_ll)
  f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (rho_v3_rr - rho_v3_ll)
  f5 = 1/2 * (f_ll[5] + f_rr[5]) - 1/2 * λ_max * (rho_e_rr  - rho_e_ll)
  f6 = 1/2 * (f_ll[6] + f_rr[6]) - 1/2 * λ_max * (B1_rr     - B1_ll)
  f7 = 1/2 * (f_ll[7] + f_rr[7]) - 1/2 * λ_max * (B2_rr     - B2_ll)
  f8 = 1/2 * (f_ll[8] + f_rr[8]) - 1/2 * λ_max * (B3_rr     - B3_ll)
  f9 = 1/2 * (f_ll[9] + f_rr[9]) - 1/2 * λ_max * (psi_rr    - psi_ll)

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


"""
    flux_hll(u_ll, u_rr, orientation, equation::IdealGlmMhdEquations3D)

HLL flux for ideal GLM-MHD equations like that by Li (2005)
  An HLLC Riemann solver for magneto-hydrodynamics
[DOI: 10.1016/j.jcp.2004.08.020](https://doi.org/10.1016/j.jcp.2004.08.020)
"""
function flux_hll(u_ll, u_rr, orientation, equation::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  p_ll = (equation.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)

  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  p_rr = (equation.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equation)
  f_rr = calcflux(u_rr, orientation, equation)

  # Approximate the left-most and right-most eigenvalues in the Riemann fan
  if orientation == 1 # x-direction
    c_f_ll = calc_fast_wavespeed(u_ll, orientation, equation)
    c_f_rr = calc_fast_wavespeed(u_rr, orientation, equation)
    vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equation)
    Ssl = min(v1_ll - c_f_ll, vel_roe - c_f_roe)
    Ssr = max(v1_rr + c_f_rr, vel_roe + c_f_roe)
  elseif orientation == 2 # y-direction
    c_f_ll = calc_fast_wavespeed(u_ll, orientation, equation)
    c_f_rr = calc_fast_wavespeed(u_rr, orientation, equation)
    vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equation)
    Ssl = min(v2_ll - c_f_ll, vel_roe - c_f_roe)
    Ssr = max(v2_rr + c_f_rr, vel_roe + c_f_roe)
  else # z-direction
    c_f_ll = calc_fast_wavespeed(u_ll, orientation, equation)
    c_f_rr = calc_fast_wavespeed(u_rr, orientation, equation)
    vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equation)
    Ssl = min(v2_ll - c_f_ll, vel_roe - c_f_roe)
    Ssr = max(v2_rr + c_f_rr, vel_roe + c_f_roe)
  end

  if Ssl >= 0.0 && Ssr > 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
    f4 = f_ll[4]
    f5 = f_ll[5]
    f6 = f_ll[6]
    f7 = f_ll[7]
    f8 = f_ll[8]
    f9 = f_ll[9]
  elseif Ssr <= 0.0 && Ssl < 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
    f4 = f_rr[4]
    f5 = f_rr[5]
    f6 = f_rr[6]
    f7 = f_rr[7]
    f8 = f_rr[8]
    f9 = f_rr[9]
  else
    f1 = (Ssr * f_ll[1] - Ssl * f_rr[1] + Ssl * Ssr * (rho_rr[1]    - rho_ll[1]))    / (Ssr - Ssl)
    f2 = (Ssr * f_ll[2] - Ssl * f_rr[2] + Ssl * Ssr * (rho_v1_rr[1] - rho_v1_ll[1])) / (Ssr - Ssl)
    f3 = (Ssr * f_ll[3] - Ssl * f_rr[3] + Ssl * Ssr * (rho_v2_rr[1] - rho_v2_ll[1])) / (Ssr - Ssl)
    f4 = (Ssr * f_ll[4] - Ssl * f_rr[4] + Ssl * Ssr * (rho_v3_rr[1] - rho_v3_ll[1])) / (Ssr - Ssl)
    f5 = (Ssr * f_ll[5] - Ssl * f_rr[5] + Ssl * Ssr * (rho_e_rr[1]  - rho_e_ll[1]))  / (Ssr - Ssl)
    f6 = (Ssr * f_ll[6] - Ssl * f_rr[6] + Ssl * Ssr * (B1_rr[1]     - B1_ll[1]))     / (Ssr - Ssl)
    f7 = (Ssr * f_ll[7] - Ssl * f_rr[7] + Ssl * Ssr * (B2_rr[1]     - B2_ll[1]))     / (Ssr - Ssl)
    f8 = (Ssr * f_ll[8] - Ssl * f_rr[8] + Ssl * Ssr * (B3_rr[1]     - B3_ll[1]))     / (Ssr - Ssl)
    f9 = (Ssr * f_ll[9] - Ssl * f_rr[9] + Ssl * Ssr * (psi_rr[1]    - psi_ll[1]))    / (Ssr - Ssl)
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# strong form of nonconservative flux on a side, e.g., the Powell term
#     phi^L 1/2 (B^L+B^R) normal - phi^L B^L normal = phi^L 1/2 (B^R-B^L) normal
# OBS! 1) "weak" formulation of split DG already includes the contribution -1/2(phi^L B^L normal)
#         so this routine only adds 1/2(phi^L B^R nvec)
#         analogously for the Galilean nonconservative term
#      2) this is non-unique along an interface! normal direction is super important
function noncons_interface_flux(u_left, u_right, orientation, equation::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _, B1_ll, B2_ll, B3_ll, psi_ll = u_left
  _, _, _, _, _, B1_rr, B2_rr, B3_rr, psi_rr = u_right

  # extract velocites from the left
  v1_ll  = rho_v1_ll / rho_ll
  v2_ll  = rho_v2_ll / rho_ll
  v3_ll  = rho_v3_ll / rho_ll
  v_dot_B_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  # extract magnetic field variable from the right and set the normal velocity
  # Note, both depend upon the orientation and need psi_rr
  if orientation == 1 # x-direction
    v_normal = v1_ll
    B_normal = B1_rr
  elseif orientation == 2 # y-direction
    v_normal = v2_ll
    B_normal = B2_rr
  else # z-direction
    v_normal = v3_ll
    B_normal = B3_rr
  end
  # compute the nonconservative flux: Powell (with B_normal) and Galilean (with v_normal)
  noncons2 = 0.5 * B_normal * B1_ll
  noncons3 = 0.5 * B_normal * B2_ll
  noncons4 = 0.5 * B_normal * B3_ll
  noncons5 = 0.5 * B_normal * v_dot_B_ll + 0.5 * v_normal * psi_ll * psi_rr
  noncons6 = 0.5 * B_normal * v1_ll
  noncons7 = 0.5 * B_normal * v2_ll
  noncons8 = 0.5 * B_normal * v3_ll
  noncons9 = 0.5 * v_normal * psi_rr

  return SVector(0, noncons2, noncons3, noncons4, noncons5, noncons6, noncons7, noncons8, noncons9)
end


# 1) Determine maximum stable time step based on polynomial degree and CFL number
# 2) Update the GLM cleaning wave speed c_h to be the largest value of the fast
#    magnetoacoustic over the entire domain (note this routine is called in a loop
#    over all elements in dg.jl)
function calc_max_dt(u, element_id, invjacobian, cfl,
                     equation::IdealGlmMhdEquations3D, dg)
  λ_max = 0.0
  equation.c_h = 0.0
  for k in 1:nnodes(dg), j in 1:nnodes(dg), i in 1:nnodes(dg)
    u_node = get_node_vars(u, dg, i, j, k, element_id)
    rho, rho_v1, rho_v2, rho_v3, _ = u_node
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho
    v_mag = sqrt(v1^2 + v2^2 + v3^2)
    cf_x_direction = calc_fast_wavespeed(u_node, 1, equation)
    cf_y_direction = calc_fast_wavespeed(u_node, 2, equation)
    cf_z_direction = calc_fast_wavespeed(u_node, 3, equation)
    cf_max = max(cf_x_direction, cf_y_direction, cf_z_direction)
    equation.c_h = max(equation.c_h, cf_max) # GLM cleaning speed = c_f
    λ_max = max(λ_max, v_mag + cf_max)
  end

  dt = cfl * 2 / (nnodes(dg) * invjacobian * λ_max)

  return dt
end


# Convert conservative variables to primitive
function cons2prim(cons, equation::IdealGlmMhdEquations3D)
  prim = similar(cons)
  @. prim[1, :, :, :, :] = cons[1, :, :, :, :]
  @. prim[2, :, :, :, :] = cons[2, :, :, :, :] / cons[1, :, :, :, :]
  @. prim[3, :, :, :, :] = cons[3, :, :, :, :] / cons[1, :, :, :, :]
  @. prim[4, :, :, :, :] = cons[4, :, :, :, :] / cons[1, :, :, :, :]
  @. prim[5, :, :, :, :] = ((equation.gamma - 1)
                           *(cons[5, :, :, :, :] - 0.5 * (cons[2, :, :, :, :] * prim[2, :, :, :, :] +
                                                          cons[3, :, :, :, :] * prim[3, :, :, :, :] +
                                                          cons[4, :, :, :, :] * prim[4, :, :, :, :])
                                                 - 0.5 * (cons[6, :, :, :, :] * cons[6, :, :, :, :] +
                                                          cons[7, :, :, :, :] * cons[7, :, :, :, :] +
                                                          cons[8, :, :, :, :] * cons[8, :, :, :, :])
                                                 - 0.5 * cons[9, :, :, :, :] * cons[9, :, :, :, :]))
  @. prim[6, :, :, :, :] = cons[6, :, :, :, :]
  @. prim[7, :, :, :, :] = cons[7, :, :, :, :]
  @. prim[8, :, :, :, :] = cons[8, :, :, :, :]
  @. prim[9, :, :, :, :] = cons[9, :, :, :, :]
  return prim
end


# Convert conservative variables to entropy
function cons2entropy(cons, n_nodes, n_elements, equation::IdealGlmMhdEquations3D)
  entropy = similar(cons)
  v = zeros(3, n_nodes, n_nodes, n_nodes, n_elements)
  B = zeros(3, n_nodes, n_nodes, n_nodes, n_elements)
  v_square = zeros(n_nodes, n_nodes, n_nodes, n_elements)
  p = zeros(n_nodes, n_nodes, n_nodes, n_elements)
  s = zeros(n_nodes, n_nodes, n_nodes, n_elements)
  rho_p = zeros(n_nodes, n_nodes, n_nodes, n_elements)
  # velocities
  @. v[1, :, :, :, :] = cons[2, :, :, :, :] / cons[1, :, :, :, :]
  @. v[2, :, :, :, :] = cons[3, :, :, :, :] / cons[1, :, :, :, :]
  @. v[3, :, :, :, :] = cons[4, :, :, :, :] / cons[1, :, :, :, :]
  # magnetic fields
  @. B[1, :, :, :, :] = cons[6, :, :, :, :]
  @. B[2, :, :, :, :] = cons[7, :, :, :, :]
  @. B[3, :, :, :, :] = cons[8, :, :, :, :]
  # kinetic energy, pressure, entropy
  @. v_square[ :, :, :, :] = (v[1, :, :, :, :]*v[1, :, :, :, :] +
                              v[2, :, :, :, :]*v[2, :, :, :, :] +
                              v[3, :, :, :, :]*v[3, :, :, :, :])
  @. p[ :, :, :, :] = ((equation.gamma - 1)*(cons[5, :, :, :, :] -
                                             0.5*cons[1, :, :, :, :]*v_square[:,:,:, :] -
                    0.5*(B[1, :, :, :, :]*B[1, :, :, :, :] + B[2, :, :, :, :]*B[2, :, :, :, :] +
                         B[3, :, :, :, :]*B[3, :, :, :, :])
                    - 0.5*cons[9, :, :, :, :]*cons[9, :, :, :, :]))
  @. s[ :, :, :, :] = log(p[:, :, :, :]) - equation.gamma*log(cons[1, :, :, :, :])
  @. rho_p[ :, :, :, :] = cons[1, :, :, :, :] / p[ :, :, :, :]

  @. entropy[1, :, :, :, :] = ((equation.gamma - s[:,:,:,:])/(equation.gamma-1) -
                               0.5*rho_p[:,:,:,:]*v_square[:,:,:,:])
  @. entropy[2, :, :, :, :] =  rho_p[:,:,:,:]*v[1,:,:,:,:]
  @. entropy[3, :, :, :, :] =  rho_p[:,:,:,:]*v[2,:,:,:,:]
  @. entropy[4, :, :, :, :] =  rho_p[:,:,:,:]*v[3,:,:,:,:]
  @. entropy[5, :, :, :, :] = -rho_p[:,:,:,:]
  @. entropy[6, :, :, :, :] =  rho_p[:,:,:,:]*B[1,:,:,:,:]
  @. entropy[7, :, :, :, :] =  rho_p[:,:,:,:]*B[2,:,:,:,:]
  @. entropy[8, :, :, :, :] =  rho_p[:,:,:,:]*B[3,:,:,:,:]
  @. entropy[9, :, :, :, :] =  rho_p[:,:,:,:]*cons[9,:,:,:,:]

  return entropy
end

# Convert primitive to conservative variables
function prim2cons(prim, equation::IdealGlmMhdEquations3D)
  cons = similar(prim)
  cons[1] = prim[1]
  cons[2] = prim[2] * prim[1]
  cons[3] = prim[3] * prim[1]
  cons[4] = prim[4] * prim[1]
  cons[5] = prim[5]/(equation.gamma-1)+0.5*(cons[2]*prim[2] + cons[3]*prim[3] + cons[4]*prim[4])+
            0.5*(prim[6]*prim[6] + prim[7]*prim[7] + prim[8]*prim[8] + 0.5*prim[9]*prim[9])
  cons[6] = prim[6]
  cons[7] = prim[7]
  cons[8] = prim[8]
  cons[9] = prim[9]
  return cons
end


# Convert conservative variables to indicator variable for discontinuities (elementwise version)
@inline function cons2indicator!(indicator, cons, element_id, n_nodes, indicator_variable,
                                 equation::IdealGlmMhdEquations3D)
  for k in 1:n_nodes, j in 1:n_nodes, i in 1:n_nodes
    indicator[1, i, j, k] = cons2indicator(cons[1, i, j, k, element_id],
                                           cons[2, i, j, k, element_id],
                                           cons[3, i, j, k, element_id],
                                           cons[4, i, j, k, element_id],
                                           cons[5, i, j, k, element_id],
                                           cons[6, i, j, k, element_id],
                                           cons[7, i, j, k, element_id],
                                           cons[8, i, j, k, element_id],
                                           cons[9, i, j, k, element_id],
                                           indicator_variable, equation)
  end
end



# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi,
                                ::Val{:density}, equation::IdealGlmMhdEquations3D)
  # Indicator variable is rho
  return rho
end



# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi,
                                ::Val{:pressure}, equation::IdealGlmMhdEquations3D)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  # Indicator variable is p
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2)
                                  - 0.5*(B1^2 + B2^2 + B3^2)
                                  - 0.5*psi^2)
  return p
end


# Convert conservative variables to indicator variable for discontinuities (pointwise version)
@inline function cons2indicator(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi,
                                ::Val{:density_pressure}, equation::IdealGlmMhdEquations3D)
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  # Calculate pressure
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2)
                                  - 0.5*(B1^2 + B2^2 + B3^2)
                                  - 0.5*psi^2)
  # Indicator variable is rho * p
  return rho * p
end


# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(cons, direction, equation::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  v_mag = sqrt(v1^2 + v2^2 + v3^2)
  p = (equation.gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  a_square = equation.gamma * p / rho
  b1 = B1/sqrt(rho)
  b2 = B2/sqrt(rho)
  b3 = B3/sqrt(rho)
  b_square = b1^2 + b2^2 + b3^2
  if direction == 1 # x-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2))
  elseif direction == 2 # y-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b2^2))
  else # z-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b3^2))
  end
  return c_f
end


"""
    calc_fast_wavespeed_roe(u_ll, u_rr, direction, equation::IdealGlmMhdEquations3D)

Compute the fast magnetoacoustic wave speed using Roe averages
as given by Cargo and Gallice (1997)
  Roe Matrices for Ideal MHD and Systematic Construction
  of Roe Matrices for Systems of Conservation Laws
[DOI: 10.1006/jcph.1997.5773](https://doi.org/10.1006/jcph.1997.5773)
"""
@inline function calc_fast_wavespeed_roe(u_ll, u_rr, direction, equation::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate primitive variables
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  p_ll = (equation.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)

  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  p_rr = (equation.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

  # compute total pressure which is thermal + magnetic pressures
  p_total_ll = p_ll + 0.5*mag_norm_ll
  p_total_rr = p_rr + 0.5*mag_norm_rr

  # compute the Roe density averages
  sqrt_rho_ll = sqrt(rho_ll)
  sqrt_rho_rr = sqrt(rho_rr)
  inv_sqrt_rho_add  = 1.0 / (sqrt_rho_ll + sqrt_rho_rr)
  inv_sqrt_rho_prod = 1.0 / (sqrt_rho_ll * sqrt_rho_rr)
  rho_ll_roe =  sqrt_rho_ll * inv_sqrt_rho_add
  rho_rr_roe =  sqrt_rho_rr * inv_sqrt_rho_add
  # Roe averages
  # velocities and magnetic fields
  v1_roe = v1_ll * rho_ll_roe + v1_rr * rho_rr_roe
  v2_roe = v2_ll * rho_ll_roe + v2_rr * rho_rr_roe
  v3_roe = v3_ll * rho_ll_roe + v3_rr * rho_rr_roe
  B1_roe = B1_ll * rho_ll_roe + B1_rr * rho_rr_roe
  B2_roe = B2_ll * rho_ll_roe + B2_rr * rho_rr_roe
  B3_roe = B3_ll * rho_ll_roe + B3_rr * rho_rr_roe
  # enthalpy
  H_ll  = (rho_e_ll + p_total_ll) / rho_ll
  H_rr  = (rho_e_rr + p_total_rr) / rho_rr
  H_roe = H_ll * rho_ll_roe + H_rr * rho_rr_roe
  # temporary vairable see equation (4.12) in Cargo and Gallice
  X = 0.5 * ( (B1_ll - B1_rr)^2 + (B2_ll - B2_rr)^2 + (B3_ll - B3_rr)^2 ) * inv_sqrt_rho_add^2
  # averaged components needed to compute c_f, the fast magnetoacoustic wave speed
  b_square_roe = (B1_roe^2 + B2_roe^2 + B3_roe^2) * inv_sqrt_rho_prod # scaled magnectic sum
  a_square_roe = ((2.0 - equation.gamma) * X +
                 (equation.gamma -1.0) * (H_roe - 0.5*(v1_roe^2 + v2_roe^2 + v3_roe^2) -
                                          b_square_roe)) # acoustic speed
  # finally compute the average wave speed and set the output velocity (depends on orientation)
  if direction == 1 # x-direction
    c_a_roe = B1_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
    a_star_roe = sqrt( (a_square_roe + b_square_roe)^2 - 4.0 * a_square_roe * c_a_roe )
    c_f_roe = sqrt( 0.5 * (a_square_roe + b_square_roe + a_star_roe) )
    vel_out_roe = v1_roe
  elseif direction == 2 # y-direction
    c_a_roe = B2_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
    a_star_roe = sqrt( (a_square_roe + b_square_roe)^2 - 4.0 * a_square_roe * c_a_roe )
    c_f_roe = sqrt( 0.5 * (a_square_roe + b_square_roe + a_star_roe) )
    vel_out_roe = v2_roe
  else # z-direction
    c_a_roe = B3_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
    a_star_roe = sqrt( (a_square_roe + b_square_roe)^2 - 4.0 * a_square_roe * c_a_roe )
    c_f_roe = sqrt( 0.5 * (a_square_roe + b_square_roe + a_star_roe) )
    vel_out_roe = v3_roe
  end

  return vel_out_roe, c_f_roe
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equation::IdealGlmMhdEquations3D)
  # Pressure
  p = (equation.gamma - 1) * (cons[5] - 1/2 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1]
                                      - 1/2 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
                                      - 1/2 * cons[9]^2)

  # Thermodynamic entropy
  s = log(p) - equation.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equation::IdealGlmMhdEquations3D)
  S = -entropy_thermodynamic(cons, equation) * cons[1] / (equation.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equation::IdealGlmMhdEquations3D) = entropy_math(cons, equation)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::IdealGlmMhdEquations3D) = cons[5]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equation::IdealGlmMhdEquations3D)
  return 0.5 * (cons[2]^2 + cons[3]^2 + cons[4]^2)/cons[1]
end


# Calculate the magnetic energy for a conservative state `cons'.
#  OBS! For non-dinmensional form of the ideal MHD magnetic pressure ≡ magnetic energy
@inline function energy_magnetic(cons, ::IdealGlmMhdEquations3D)
  return 0.5 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equation::IdealGlmMhdEquations3D)
  return (energy_total(cons, equation)
          - energy_kinetic(cons, equation)
          - energy_magnetic(cons, equation)
          - cons[9]^2 / 2)
end


# Calcluate the cross helicity (\vec{v}⋅\vec{B}) for a conservative state `cons'
@inline function cross_helicity(cons, ::IdealGlmMhdEquations3D)
  return (cons[2]*cons[6] + cons[3]*cons[7] + cons[4]*cons[8]) / cons[1]
end
