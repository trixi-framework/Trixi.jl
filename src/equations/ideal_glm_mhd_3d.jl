
@doc raw"""
    IdealGlmMhdEquations3D

The ideal compressible GLM-MHD equations in three space dimensions.
"""
mutable struct IdealGlmMhdEquations3D{RealT<:Real} <: AbstractIdealGlmMhdEquations{3, 9}
  gamma::RealT
  c_h::RealT # GLM cleaning speed
end

function IdealGlmMhdEquations3D(gamma; initial_c_h=convert(typeof(gamma), NaN))
  # Use `promote` to ensure that `gamma` and `initial_c_h` have the same type
  IdealGlmMhdEquations3D(promote(gamma, initial_c_h)...)
end


have_nonconservative_terms(::IdealGlmMhdEquations3D) = Val(true)
varnames(::typeof(cons2cons), ::IdealGlmMhdEquations3D) = ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi")
varnames(::typeof(cons2prim), ::IdealGlmMhdEquations3D) = ("rho", "v1", "v2", "v3", "p", "B1", "B2", "B3", "psi")
default_analysis_integrals(::IdealGlmMhdEquations3D)  = (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))


# Set initial conditions at physical location `x` for time `t`
"""
initial_condition_constant(x, t, equations::IdealGlmMhdEquations3D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::IdealGlmMhdEquations3D)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_v3 = -0.5
  rho_e = 50.0
  B1 = 3.0
  B2 = -1.2
  B3 = 0.5
  psi = 0.0
  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end


"""
    initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations3D)

An Alfvén wave as smooth initial condition used for convergence tests.
"""
function initial_condition_convergence_test(x, t, equations::IdealGlmMhdEquations3D)
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

  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdEquations3D)

A weak blast wave adapted from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::IdealGlmMhdEquations3D)
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

  return prim2cons(SVector(rho, v1, v2, v3, p, 1.0, 1.0, 1.0, 0.0), equations)
end


"""
    initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations3D)

The classical Orszag-Tang vortex test case. Here, the setup is taken from
- Dominik Derigs, Gregor J. Gassner, Stefanie Walch & Andrew R. Winters (2018)
  Entropy Stable Finite Volume Approximations for Ideal Magnetohydrodynamics
  [doi: 10.1365/s13291-018-0178-9](https://doi.org/10.1365/s13291-018-0178-9)
"""
function initial_condition_orszag_tang(x, t, equations::IdealGlmMhdEquations3D)
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
  return prim2cons(SVector(rho, v1, v2, v3, p, B1, B2, B3, psi), equations)
end


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::IdealGlmMhdEquations3D)


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)
  mag_en = 0.5*(B1^2 + B2^2 + B3^2)
  p = (equations.gamma - 1) * (rho_e - kin_en - mag_en - 0.5*psi^2)
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1*v1 + p + mag_en - B1^2
    f3 = rho_v1*v2 - B1*B2
    f4 = rho_v1*v3 - B1*B3
    f5 = (kin_en + equations.gamma*p/(equations.gamma - 1) + 2*mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3) + equations.c_h*psi*B1
    f6 = equations.c_h*psi
    f7 = v1*B2 - v2*B1
    f8 = v1*B3 - v3*B1
    f9 = equations.c_h*B1
  elseif orientation == 2
    f1 = rho_v2
    f2 = rho_v2*v1 - B2*B1
    f3 = rho_v2*v2 + p + mag_en - B2^2
    f4 = rho_v2*v3 - B2*B3
    f5 = (kin_en + equations.gamma*p/(equations.gamma - 1) + 2*mag_en)*v2 - B2*(v1*B1 + v2*B2 + v3*B3) + equations.c_h*psi*B2
    f6 = v2*B1 - v1*B2
    f7 = equations.c_h*psi
    f8 = v2*B3 - v3*B2
    f9 = equations.c_h*B2
  else
    f1 = rho_v3
    f2 = rho_v3*v1 - B3*B1
    f3 = rho_v3*v2 - B3*B2
    f4 = rho_v3*v3 + p + mag_en - B3^2
    f5 = (kin_en + equations.gamma*p/(equations.gamma - 1) + 2*mag_en)*v3 - B3*(v1*B1 + v2*B2 + v3*B3) + equations.c_h*psi*B3
    f6 = v3*B1 - v1*B3
    f7 = v3*B2 - v2*B3
    f8 = equations.c_h*psi
    f9 = equations.c_h*B3
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  kin_en = 0.5 * rho * (v1^2 + v2^2 + v3^2)
  mag_en = 0.5 * (B1^2 + B2^2 + B3^2)
  p = (equations.gamma - 1) * (rho_e - kin_en - mag_en - 0.5*psi^2)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2] + v3 * normal_direction[3]
  B_normal = B1 * normal_direction[1] + B2 * normal_direction[2] + B3 * normal_direction[3]
  rho_v_normal = rho * v_normal

  f1 = rho_v_normal
  f2 = rho_v_normal * v1 - B1 * B_normal + (p + mag_en) * normal_direction[1]
  f3 = rho_v_normal * v2 - B2 * B_normal + (p + mag_en) * normal_direction[2]
  f4 = rho_v_normal * v3 - B3 * B_normal + (p + mag_en) * normal_direction[3]
  f5 = ( (kin_en + equations.gamma*p/(equations.gamma - 1) + 2*mag_en) * v_normal
        - B_normal * (v1*B1 + v2*B2 + v3*B3) + equations.c_h * psi * B_normal )
  f6 = ( equations.c_h * psi * normal_direction[1] + (v2 * B1 - v1 * B2) * normal_direction[2] +
         (v3 * B1 - v1 * B3) * normal_direction[3] )
  f7 = ( (v1 * B2 - v2 * B1) * normal_direction[1] + equations.c_h * psi * normal_direction[2] +
         (v3 * B2 - v2 * B3) * normal_direction[3] )
  f8 = ( (v1 * B3 - v3 * B1) * normal_direction[1] + (v2 * B3 - v3 * B2) * normal_direction[2] +
         equations.c_h * psi * normal_direction[3] )
  f9 = equations.c_h * B_normal

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate the nonconservative terms from Powell and Galilean invariance
# OBS! This is scaled by 1/2 becuase it will cancel later with the factor of 2 in dsplit_transposed
@inline function calcflux_twopoint_nonconservative!(f1, f2, f3, u, element,
                                                    equations::IdealGlmMhdEquations3D, dg, cache)
  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = get_node_vars(u, equations, dg, i, j, k, element)
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
    for l in eachnode(dg)
      _, _, _, _, _, B1, _, _, psi = get_node_vars(u, equations, dg, l, j, k, element)
      for v in eachvariable(equations)
        f1[v, l, i, j, k] += phi_pow[v] * B1 + phi_gal_x[v] * psi
      end
      _, _, _, _, _, _, B2, _, psi = get_node_vars(u, equations, dg, i, l, k, element)
      for v in eachvariable(equations)
        f2[v, l, i, j, k] += phi_pow[v] * B2 + phi_gal_y[v] * psi
      end
      _, _, _, _, _, _, _, B3, psi = get_node_vars(u, equations, dg, i, j, l, element)
      for v in eachvariable(equations)
        f3[v, l, i, j, k] += phi_pow[v] * B3 + phi_gal_z[v] * psi
      end
    end
  end

  return nothing
end


# Calculate the nonconservative terms from Powell and Galilean invariance for CurvedMesh{3}
# OBS! This is scaled by 1/2 becuase it will cancel later with the factor of 2 in dsplit_transposed
@inline function calcflux_twopoint_nonconservative!(f1, f2, f3, u, element, contravariant_vectors,
                                                    equations::IdealGlmMhdEquations3D, dg, cache)
   for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = get_node_vars(u, equations, dg, i, j, k, element)
    v1 = rho_v1 / rho
    v2 = rho_v2 / rho
    v3 = rho_v3 / rho

    # Powell nonconservative term: Φ^Pow = (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
    phi_pow = 0.5 * SVector(0, B1, B2, B3, v1*B1 + v2*B2 + v3*B3, v1, v2, v3, 0)

    # Galilean nonconservative term: Φ^Gal_{1,2,3} = (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
    # first direction
    Ja11_ijk, Ja12_ijk, Ja13_ijk = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
    phi_gal_x = 0.5*(Ja11_ijk*v1 + Ja12_ijk*v2 + Ja13_ijk*v3).*SVector(0, 0, 0, 0, psi, 0, 0, 0, 1)
    # second direction
    Ja21_ijk, Ja22_ijk, Ja23_ijk = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
    phi_gal_y = 0.5*(Ja21_ijk*v1 + Ja22_ijk*v2 + Ja23_ijk*v3).*SVector(0, 0, 0, 0, psi, 0, 0, 0, 1)
    # third direction
    Ja31_ijk, Ja32_ijk, Ja33_ijk = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
    phi_gal_z = 0.5*(Ja31_ijk*v1 + Ja32_ijk*v2 + Ja33_ijk*v3).*SVector(0, 0, 0, 0, psi, 0, 0, 0, 1)

    # add both nonconservative terms into the volume
    for l in eachnode(dg)
      _, _, _, _, _, B1, B2, B3, psi = get_node_vars(u, equations, dg, l, j, k, element)
      Ja11_ljk, Ja12_ljk, Ja13_ljk = get_contravariant_vector(1, contravariant_vectors, l, j, k, element)
      Ja11_avg = 0.5 * (Ja11_ijk + Ja11_ljk)
      Ja12_avg = 0.5 * (Ja12_ijk + Ja12_ljk)
      Ja13_avg = 0.5 * (Ja13_ijk + Ja13_ljk)
      for v in eachvariable(equations)
        f1[v, l, i, j, k] += ( phi_pow[v] * (Ja11_avg * B1 + Ja12_avg * B2 + Ja13_avg * B3) +
                               phi_gal_x[v] * psi )
      end
      _, _, _, _, _, B1, B2, B3, psi = get_node_vars(u, equations, dg, i, l, k, element)
      Ja21_ilk, Ja22_ilk, Ja23_ilk = get_contravariant_vector(2, contravariant_vectors, i, l, k, element)
      Ja21_avg = 0.5 * (Ja21_ijk + Ja21_ilk)
      Ja22_avg = 0.5 * (Ja22_ijk + Ja22_ilk)
      Ja23_avg = 0.5 * (Ja23_ijk + Ja23_ilk)
      for v in eachvariable(equations)
        f2[v, l, i, j, k] += ( phi_pow[v] * (Ja21_avg * B1 + Ja22_avg * B2 + Ja23_avg * B3) +
                               phi_gal_y[v] * psi )
      end
      _, _, _, _, _, B1, B2, B3, psi = get_node_vars(u, equations, dg, i, j, l, element)
      Ja31_ijl, Ja32_ijl, Ja33_ijl = get_contravariant_vector(3, contravariant_vectors, i, j, l, element)
      Ja31_avg = 0.5 * (Ja31_ijk + Ja31_ijl)
      Ja32_avg = 0.5 * (Ja32_ijk + Ja32_ijl)
      Ja33_avg = 0.5 * (Ja33_ijk + Ja33_ijl)
      for v in eachvariable(equations)
        f3[v, l, i, j, k] += ( phi_pow[v] * (Ja31_avg * B1 + Ja32_avg * B2 + Ja33_avg * B3) +
                               phi_gal_z[v] * psi )
      end
    end
  end

  return nothing
end


"""
    flux_derigs_etal(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations3D)

Entropy conserving two-point flux by
- Derigs et al. (2018)
  Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field
  divergence diminishing ideal magnetohydrodynamics equations
  [DOI: 10.1016/j.jcp.2018.03.002](https://doi.org/10.1016/j.jcp.2018.03.002)
"""
function flux_derigs_etal(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
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
  p_ll = (equations.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)
  p_rr = (equations.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)
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
    f6 = equations.c_h*psi_avg
    f7 = v1_avg*B2_avg - v2_avg*B1_avg
    f8 = v1_avg*B3_avg - v3_avg*B1_avg
    f9 = equations.c_h*B1_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B1_avg = 0.5*(B1_ll*psi_ll + B1_rr*psi_rr)
    v1_mag_avg = 0.5*(v1_ll*mag_norm_ll + v1_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equations.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v1_mag_avg +
          B1_avg*vel_dot_mag_avg - equations.c_h*psi_B1_avg)
  elseif orientation == 2
    f1 = rho_mean*v2_avg
    f2 = f1*v1_avg - B2_avg*B1_avg
    f3 = f1*v2_avg + p_mean + 0.5*mag_norm_avg - B2_avg*B2_avg
    f4 = f1*v3_avg - B2_avg*B3_avg
    f6 = v2_avg*B1_avg - v1_avg*B2_avg
    f7 = equations.c_h*psi_avg
    f8 = v2_avg*B3_avg - v3_avg*B2_avg
    f9 = equations.c_h*B2_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B2_avg = 0.5*(B2_ll*psi_ll + B2_rr*psi_rr)
    v2_mag_avg = 0.5*(v2_ll*mag_norm_ll + v2_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equations.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v2_mag_avg +
          B2_avg*vel_dot_mag_avg - equations.c_h*psi_B2_avg)
  else
    f1 = rho_mean*v3_avg
    f2 = f1*v1_avg - B3_avg*B1_avg
    f3 = f1*v2_avg - B3_avg*B2_avg
    f4 = f1*v3_avg + p_mean + 0.5*mag_norm_avg - B3_avg*B3_avg
    f6 = v3_avg*B1_avg - v1_avg*B3_avg
    f7 = v3_avg*B2_avg - v2_avg*B3_avg
    f8 = equations.c_h*psi_avg
    f9 = equations.c_h*B3_avg
    # total energy flux is complicated and involves the previous eight components
    psi_B3_avg = 0.5*(B3_ll*psi_ll + B3_rr*psi_rr)
    v3_mag_avg = 0.5*(v3_ll*mag_norm_ll + v3_rr*mag_norm_rr)
    f5 = (f1*0.5*(1/(equations.gamma-1)/beta_mean - vel_norm_avg) + f2*v1_avg + f3*v2_avg +
          f4*v3_avg + f6*B1_avg + f7*B2_avg + f8*B3_avg + f9*psi_avg - 0.5*v3_mag_avg +
          B3_avg*vel_dot_mag_avg - equations.c_h*psi_B3_avg)
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


"""
    flux_hindenlang(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations3D)

Entropy conserving and kinetic energy preserving two-point flux of
Hindenlang (2019), extending [`flux_ranocha`](@ref) to the MHD equations.

## References
- Hindenlang (2019)
  A new entropy conservative two-point flux for ideal MHD equations derived from
  first principles.
  Presented at HONOM 2019: European workshop on high order numerical methods
  for evolutionary PDEs, theory and applications
- Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
  [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
- Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
  [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_hindenlang(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  # Unpack left and right states to get velocities and pressure
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  p_ll = (equations.gamma - 1) * (rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)
  p_rr = (equations.gamma - 1) * (rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

  # Compute the necessary mean values needed for either direction
  rho_mean   = ln_mean(rho_ll, rho_rr)
  rho_p_mean = ln_mean(rho_ll / p_ll, rho_rr / p_rr)
  v1_avg  = 0.5 * ( v1_ll +  v1_rr)
  v2_avg  = 0.5 * ( v2_ll +  v2_rr)
  v3_avg  = 0.5 * ( v3_ll +  v3_rr)
  p_avg   = 0.5 * (  p_ll +   p_rr)
  psi_avg = 0.5 * (psi_ll + psi_rr)
  velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
  magnetic_square_avg = 0.5 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

  # Calculate fluxes depending on orientation with specific direction averages
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg + magnetic_square_avg - 0.5 * (B1_ll * B1_rr + B1_rr * B1_ll)
    f3 = f1 * v2_avg                               - 0.5 * (B1_ll * B2_rr + B1_rr * B2_ll)
    f4 = f1 * v3_avg                               - 0.5 * (B1_ll * B3_rr + B1_rr * B3_ll)
    #f5 below
    f6 = equations.c_h * psi_avg
    f7 = 0.5 * (v1_ll * B2_ll - v2_ll * B1_ll + v1_rr * B2_rr - v2_rr * B1_rr)
    f8 = 0.5 * (v1_ll * B3_ll - v3_ll * B1_ll + v1_rr * B3_rr - v3_rr * B1_rr)
    f9 = equations.c_h * 0.5 * (B1_ll + B1_rr)
    # total energy flux is complicated and involves the previous components
    f5 = ( f1 * ( velocity_square_avg + 1 / ((equations.gamma-1) * rho_p_mean) )
          + 0.5 * ( p_ll * v1_rr +  p_rr * v1_ll)
          + 0.5 * (v1_ll * B2_ll * B2_rr + v1_rr * B2_rr * B2_ll)
          + 0.5 * (v1_ll * B3_ll * B3_rr + v1_rr * B3_rr * B3_ll)
          - 0.5 * (v2_ll * B1_ll * B2_rr + v2_rr * B1_rr * B2_ll)
          - 0.5 * (v3_ll * B1_ll * B3_rr + v3_rr * B1_rr * B3_ll)
          + 0.5 * equations.c_h * (B1_ll * psi_rr + B1_rr * psi_ll) )
  elseif orientation == 2
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg                               - 0.5 * (B2_ll * B1_rr + B2_rr * B1_ll)
    f3 = f1 * v2_avg + p_avg + magnetic_square_avg - 0.5 * (B2_ll * B2_rr + B2_rr * B2_ll)
    f4 = f1 * v3_avg                               - 0.5 * (B2_ll * B3_rr + B2_rr * B3_ll)
    #f5 below
    f6 = 0.5 * (v2_ll * B1_ll - v1_ll * B2_ll + v2_rr * B1_rr - v1_rr * B2_rr)
    f7 = equations.c_h * psi_avg
    f8 = 0.5 * (v2_ll * B3_ll - v3_ll * B2_ll + v2_rr * B3_rr - v3_rr * B2_rr)
    f9 = equations.c_h * 0.5 * (B2_ll + B2_rr)
    # total energy flux is complicated and involves the previous components
    f5 = ( f1 * ( velocity_square_avg + 1 / ((equations.gamma-1) * rho_p_mean) )
          + 0.5 * ( p_ll * v2_rr +  p_rr * v2_ll)
          + 0.5 * (v2_ll * B1_ll * B1_rr + v2_rr * B1_rr * B1_ll)
          + 0.5 * (v2_ll * B3_ll * B3_rr + v2_rr * B3_rr * B3_ll)
          - 0.5 * (v1_ll * B2_ll * B1_rr + v1_rr * B2_rr * B1_ll)
          - 0.5 * (v3_ll * B2_ll * B3_rr + v3_rr * B2_rr * B3_ll)
          + 0.5 * equations.c_h * (B2_ll * psi_rr + B2_rr * psi_ll) )
  else # orientation == 3
    f1 = rho_mean * v3_avg
    f2 = f1 * v1_avg                               - 0.5 * (B3_ll * B1_rr + B3_rr * B1_ll)
    f3 = f1 * v2_avg                               - 0.5 * (B3_ll * B2_rr + B3_rr * B2_ll)
    f4 = f1 * v3_avg + p_avg + magnetic_square_avg - 0.5 * (B3_ll * B3_rr + B3_rr * B3_ll)
    #f5 below
    f6 = 0.5 * (v3_ll * B1_ll - v1_ll * B3_ll + v3_rr * B1_rr - v1_rr * B3_rr)
    f7 = 0.5 * (v3_ll * B2_ll - v2_ll * B3_ll + v3_rr * B2_rr - v2_rr * B3_rr)
    f8 = equations.c_h * psi_avg
    f9 = equations.c_h * 0.5 * (B3_ll + B3_rr)
    # total energy flux is complicated and involves the previous components
    f5 = ( f1 * ( velocity_square_avg + 1 / ((equations.gamma-1) * rho_p_mean) )
          + 0.5 * ( p_ll * v3_rr +  p_rr * v3_ll)
          + 0.5 * (v3_ll * B1_ll * B1_rr + v3_rr * B1_rr * B1_ll)
          + 0.5 * (v3_ll * B2_ll * B2_rr + v3_rr * B2_rr * B2_ll)
          - 0.5 * (v1_ll * B3_ll * B1_rr + v1_rr * B3_rr * B1_ll)
          - 0.5 * (v2_ll * B3_ll * B2_rr + v2_rr * B3_rr * B2_ll)
          + 0.5 * equations.c_h * (B3_ll * psi_rr + B3_rr * psi_ll) )

  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate velocities and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_mag_ll = sqrt(v1_ll^2 + v2_ll^2 + v3_ll^2)
  cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_mag_rr = sqrt(v1_rr^2 + v2_rr^2 + v3_rr^2)
  cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

  λ_max = max(v_mag_ll, v_mag_rr) + max(cf_ll, cf_rr)
end


@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::IdealGlmMhdEquations3D)
    # Compute wave speed estimates in each direction. Requires rotation because
    # the fast magnetoacoustic wave speed has a nonlinear dependence on the direction
    norm_ = norm(normal_direction)
    # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
    normal_vector = normal_direction / norm_
    # Some vector that can't be identical to normal_vector (unless normal_vector == 0)
    tangent1 = SVector(normal_direction[2], normal_direction[3], -normal_direction[1])
    # Orthogonal projection
    tangent1 -= dot(normal_vector, tangent1) * normal_vector
    tangent1 = normalize(tangent1)
    # Third orthogonal vector
    tangent2 = normalize(cross(normal_direction, tangent1))
    # rotate the solution states
    u_ll_rotated = rotate_to_x(u_ll, normal_vector, tangent1, tangent2, equations)
    u_rr_rotated = rotate_to_x(u_rr, normal_vector, tangent1, tangent2, equations)
  return max_abs_speed_naive(u_ll_rotated, u_rr_rotated, 1, equations) * norm(normal_direction)
end


"""
    min_max_speed_naive(u_ll, u_rr, orientation, equations::IdealGlmMhdEquations3D)

Calculate minimum and maximum wave speeds for HLL-type fluxes as in
- Li (2005)
  An HLLC Riemann solver for magneto-hydrodynamics
  [DOI: 10.1016/j.jcp.2004.08.020](https://doi.org/10.1016/j.jcp.2004.08.020)
"""
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  p_ll = (equations.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)

  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  p_rr = (equations.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

  # Approximate the left-most and right-most eigenvalues in the Riemann fan
  if orientation == 1 # x-direction
    c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)
    vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equations)
    λ_min = min(v1_ll - c_f_ll, vel_roe - c_f_roe)
    λ_max = max(v1_rr + c_f_rr, vel_roe + c_f_roe)
  elseif orientation == 2 # y-direction
    c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)
    vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equations)
    λ_min = min(v2_ll - c_f_ll, vel_roe - c_f_roe)
    λ_max = max(v2_rr + c_f_rr, vel_roe + c_f_roe)
  else # z-direction
    c_f_ll = calc_fast_wavespeed(u_ll, orientation, equations)
    c_f_rr = calc_fast_wavespeed(u_rr, orientation, equations)
    vel_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, orientation, equations)
    λ_min = min(v2_ll - c_f_ll, vel_roe - c_f_roe)
    λ_max = max(v2_rr + c_f_rr, vel_roe + c_f_roe)
  end

  return λ_min, λ_max
end


# Very naive way to approximate the edges of the Riemann fan in the normal direction
@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _, _, _, _, _ = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, _, _, _, _, _ = u_rr

  # Calculate primitive velocity variables
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll

  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr

  # Compute wave speed estimates in each direction. Requires rotation because
  # the fast magnetoacoustic wave speed has a nonlinear dependence on the direction
  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal_vector = normal_direction / norm_
  # Some vector that can't be identical to normal_vector (unless normal_vector == 0)
  tangent1 = SVector(normal_direction[2], normal_direction[3], -normal_direction[1])
  # Orthogonal projection
  tangent1 -= dot(normal_vector, tangent1) * normal_vector
  tangent1 = normalize(tangent1)

  # Third orthogonal vector
  tangent2 = normalize(cross(normal_direction, tangent1))

  # Compute the rotated velocities and wave speeds
  v_normal_ll = v1_ll*normal_vector[1] + v2_ll*normal_vector[2] + v3_ll*normal_vector[3]
  v_normal_rr = v1_rr*normal_vector[1] + v2_rr*normal_vector[2] + v3_rr*normal_vector[3]

  u_ll_rotated = rotate_to_x(u_ll, normal_vector, tangent1, tangent2, equations)
  u_rr_rotated = rotate_to_x(u_rr, normal_vector, tangent1, tangent2, equations)

  c_f_ll_rotated = calc_fast_wavespeed(u_ll_rotated, 1, equations)
  c_f_rr_rotated = calc_fast_wavespeed(u_rr_rotated, 1, equations)
  v_roe_rotated, c_f_roe_rotated = calc_fast_wavespeed_roe(u_ll_rotated, u_rr_rotated, 1, equations)

  # Estimate the min/max eigenvalues in the normal direction
  λ_min = min(v_normal_ll - c_f_ll_rotated, v_roe_rotated - c_f_roe_rotated) * norm_
  λ_max = max(v_normal_rr + c_f_rr_rotated, v_roe_rotated + c_f_roe_rotated) * norm_

  return λ_min, λ_max
end


# Rotate normal vector to x-axis; normal, tangent1 and tangent2 need to be orthonormal
# Called inside `FluxRotated` in `numerical_fluxes.jl` so the directions
# has been normalized prior to this rotation of the state vector
# Note, for ideal GLM-MHD only the velocities and magnetic field variables rotate
@inline function rotate_to_x(u, normal_vector, tangent1, tangent2, equations::IdealGlmMhdEquations3D)
  # Multiply with [ 1   0        0       0   0   0        0       0   0;
  #                 0   ―  normal_vector ―   0   0        0       0   0;
  #                 0   ―    tangent1    ―   0   0        0       0   0;
  #                 0   ―    tangent2    ―   0   0        0       0   0;
  #                 0   0        0       0   1   0        0       0   0;
  #                 0   0        0       0   0   ―  normal_vector ―   0;
  #                 0   0        0       0   0   ―    tangent1    ―   0;
  #                 0   0        0       0   0   ―    tangent2    ―   0;
  #                 0   0        0       0   0   0        0       0   1 ]
  return SVector(u[1],
                 normal_vector[1] * u[2] + normal_vector[2] * u[3] + normal_vector[3] * u[4],
                 tangent1[1] * u[2] + tangent1[2] * u[3] + tangent1[3] * u[4],
                 tangent2[1] * u[2] + tangent2[2] * u[3] + tangent2[3] * u[4],
                 u[5],
                 normal_vector[1] * u[6] + normal_vector[2] * u[7] + normal_vector[3] * u[8],
                 tangent1[1] * u[6] + tangent1[2] * u[7] + tangent1[3] * u[8],
                 tangent2[1] * u[6] + tangent2[2] * u[7] + tangent2[3] * u[8],
                 u[9])

end


# Rotate x-axis to normal vector; normal, tangent1 and tangent2 need to be orthonormal
# Called inside `FluxRotated` in `numerical_fluxes.jl` so the directions
# has been normalized prior to this back-rotation of the state vector
# Note, for ideal GLM-MHD only the velocities and magnetic field variables back-rotate
@inline function rotate_from_x(u, normal_vector, tangent1, tangent2, equations::IdealGlmMhdEquations3D)
  # Multiply with [ 1        0          0        0      0        0          0        0      0;
  #                 0        |          |        |      0        0          0        0      0;
  #                 0  normal_vector tangent1 tangent2  0        0          0        0      0;
  #                 0        |          |        |      0        0          0        0      0;
  #                 0        0          0        0      1        0          0        0      0;
  #                 0        0          0        0      0        |          |        |      0;
  #                 0        0          0        0      0  normal_vector tangent1 tangent2  0;
  #                 0        0          0        0      0        |          |        |      0;
  #                 0        0          0        0      0        0          0        0      1 ]
  return SVector(u[1],
                 normal_vector[1] * u[2] + tangent1[1] * u[3] + tangent2[1] * u[4],
                 normal_vector[2] * u[2] + tangent1[2] * u[3] + tangent2[2] * u[4],
                 normal_vector[3] * u[2] + tangent1[3] * u[3] + tangent2[3] * u[4],
                 u[5],
                 normal_vector[1] * u[6] + tangent1[1] * u[7] + tangent2[1] * u[8],
                 normal_vector[2] * u[6] + tangent1[2] * u[7] + tangent2[2] * u[8],
                 normal_vector[3] * u[6] + tangent1[3] * u[7] + tangent2[3] * u[8],
                 u[9])
end


# strong form of nonconservative flux on a side, e.g., the Powell term
#     phi^L 1/2 (B^L+B^R) normal - phi^L B^L normal = phi^L 1/2 (B^R-B^L) normal
# OBS! 1) "weak" formulation of split DG already includes the contribution -1/2(phi^L B^L normal)
#         so this routine only adds 1/2(phi^L B^R nvec)
#         analogously for the Galilean nonconservative term
#      2) this is non-unique along an interface! normal direction is super important
function noncons_interface_flux(u_left, u_right, orientation::Integer, equations::IdealGlmMhdEquations3D)
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


# Compute surface nonconservative "flux" computation in the normal direction (3D version)
# Note, due to the non-uniqueness of this term we cannot use any fancy rotation tricks.
@inline function noncons_interface_flux(u_left, u_right, normal_direction::AbstractVector, mode,
                                        equations::IdealGlmMhdEquations3D)
  @assert mode === :weak "only :weak version of nonconservative coupling is available for curved MHD"

  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _, B1_ll, B2_ll, B3_ll, psi_ll = u_left
  _, _, _, _, _, B1_rr, B2_rr, B3_rr, psi_rr = u_right

  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal_vector = normal_direction / norm_

  # extract velocites from the left
  v1_ll  = rho_v1_ll / rho_ll
  v2_ll  = rho_v2_ll / rho_ll
  v3_ll  = rho_v3_ll / rho_ll
  v_dot_B_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  # extract magnetic field variable from the right and set the normal velocity
  v_normal = normal_vector[1] * v1_ll + normal_vector[2] * v2_ll + normal_vector[3] * v3_ll
  B_normal = normal_vector[1] * B1_rr + normal_vector[2] * B2_rr + normal_vector[3] * B3_rr
  # compute the nonconservative flux: Powell (with B_normal) and Galilean (with v_normal)
  noncons2 = 0.5 * B_normal * B1_ll
  noncons3 = 0.5 * B_normal * B2_ll
  noncons4 = 0.5 * B_normal * B3_ll
  noncons5 = 0.5 * B_normal * v_dot_B_ll + 0.5 * v_normal * psi_ll * psi_rr
  noncons6 = 0.5 * B_normal * v1_ll
  noncons7 = 0.5 * B_normal * v2_ll
  noncons8 = 0.5 * B_normal * v3_ll
  noncons9 = 0.5 * v_normal * psi_rr

  return SVector(0, noncons2, noncons3, noncons4, noncons5, noncons6, noncons7, noncons8, noncons9) * norm_
end


@inline function max_abs_speeds(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, _ = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  cf_x_direction = calc_fast_wavespeed(u, 1, equations)
  cf_y_direction = calc_fast_wavespeed(u, 2, equations)
  cf_z_direction = calc_fast_wavespeed(u, 3, equations)

  return abs(v1) + cf_x_direction, abs(v2) + cf_y_direction, abs(v3) + cf_z_direction
end


# Convert conservative variables to primitive
function cons2prim(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5*rho*(v1^2 + v2^2 + v3^2) - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)

  return SVector(rho, v1, v2, v3, p, B1, B2, B3, psi)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  v_square = v1^2 + v2^2 + v3^2
  p = (equations.gamma - 1) * (rho_e - 0.5*rho*v_square - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  s = log(p) - equations.gamma*log(rho)
  rho_p = rho / p

  w1 = (equations.gamma - s) / (equations.gamma-1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = rho_p * v3
  w5 = -rho_p
  w6 = rho_p * B1
  w7 = rho_p * B2
  w8 = rho_p * B3
  w9 = rho_p * psi

  return SVector(w1, w2, w3, w4, w5, w6, w7, w8, w9)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::IdealGlmMhdEquations3D)
  rho, v1, v2, v3, p, B1, B2, B3, psi = prim

  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_v3 = rho * v3
  rho_e = p/(equations.gamma-1) + 0.5 * (rho_v1*v1 + rho_v2*v2 + rho_v3*v3) +
                                  0.5 * (B1^2 + B2^2 + B3^2) + 0.5 * psi^2

  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end


@inline function density(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  return rho
end

@inline function pressure(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  p = (equations.gamma - 1)*(rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
                                   - 0.5 * (B1^2 + B2^2 + B3^2)
                                   - 0.5 * psi^2)
  return p
end

@inline function density_pressure(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  p = (equations.gamma - 1)*(rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho
                                   - 0.5 * (B1^2 + B2^2 + B3^2)
                                   - 0.5 * psi^2)
  return rho * p
end


# Compute the fastest wave speed for ideal MHD equations: c_f, the fast magnetoacoustic eigenvalue
@inline function calc_fast_wavespeed(cons, direction, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
  v1 = rho_v1/rho
  v2 = rho_v2/rho
  v3 = rho_v3/rho
  v_mag = sqrt(v1^2 + v2^2 + v3^2)
  p = (equations.gamma - 1)*(rho_e - 0.5*rho*v_mag^2 - 0.5*(B1^2 + B2^2 + B3^2) - 0.5*psi^2)
  a_square = equations.gamma * p / rho
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
    calc_fast_wavespeed_roe(u_ll, u_rr, direction, equations::IdealGlmMhdEquations3D)

Compute the fast magnetoacoustic wave speed using Roe averages as given by
- Cargo and Gallice (1997)
  Roe Matrices for Ideal MHD and Systematic Construction
  of Roe Matrices for Systems of Conservation Laws
  [DOI: 10.1006/jcph.1997.5773](https://doi.org/10.1006/jcph.1997.5773)
"""
@inline function calc_fast_wavespeed_roe(u_ll, u_rr, direction, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate primitive variables
  v1_ll = rho_v1_ll/rho_ll
  v2_ll = rho_v2_ll/rho_ll
  v3_ll = rho_v3_ll/rho_ll
  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  p_ll = (equations.gamma - 1)*(rho_e_ll - 0.5*rho_ll*vel_norm_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)

  v1_rr = rho_v1_rr/rho_rr
  v2_rr = rho_v2_rr/rho_rr
  v3_rr = rho_v3_rr/rho_rr
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  p_rr = (equations.gamma - 1)*(rho_e_rr - 0.5*rho_rr*vel_norm_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

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
  a_square_roe = ((2.0 - equations.gamma) * X +
                 (equations.gamma -1.0) * (H_roe - 0.5*(v1_roe^2 + v2_roe^2 + v3_roe^2) -
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
@inline function entropy_thermodynamic(cons, equations::IdealGlmMhdEquations3D)
  # Pressure
  p = (equations.gamma - 1) * (cons[5] - 1/2 * (cons[2]^2 + cons[3]^2 + cons[4]^2) / cons[1]
                                       - 1/2 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
                                       - 1/2 * cons[9]^2)

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::IdealGlmMhdEquations3D)
  S = -entropy_thermodynamic(cons, equations) * cons[1] / (equations.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::IdealGlmMhdEquations3D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::IdealGlmMhdEquations3D) = cons[5]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equations::IdealGlmMhdEquations3D)
  return 0.5 * (cons[2]^2 + cons[3]^2 + cons[4]^2)/cons[1]
end


# Calculate the magnetic energy for a conservative state `cons'.
#  OBS! For non-dinmensional form of the ideal MHD magnetic pressure ≡ magnetic energy
@inline function energy_magnetic(cons, ::IdealGlmMhdEquations3D)
  return 0.5 * (cons[6]^2 + cons[7]^2 + cons[8]^2)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::IdealGlmMhdEquations3D)
  return (energy_total(cons, equations)
          - energy_kinetic(cons, equations)
          - energy_magnetic(cons, equations)
          - cons[9]^2 / 2)
end


# Calcluate the cross helicity (\vec{v}⋅\vec{B}) for a conservative state `cons'
@inline function cross_helicity(cons, ::IdealGlmMhdEquations3D)
  return (cons[2]*cons[6] + cons[3]*cons[7] + cons[4]*cons[8]) / cons[1]
end
