# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    IdealGlmMhdEquations3D(gamma)

The ideal compressible GLM-MHD equations for an ideal gas with ratio of
specific heats `gamma` in three space dimensions.
"""
mutable struct IdealGlmMhdEquations3D{RealT<:Real} <: AbstractIdealGlmMhdEquations{3, 9}
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
  c_h::RealT                 # GLM cleaning speed

  function IdealGlmMhdEquations3D(gamma, c_h)
    γ, inv_gamma_minus_one, c_h = promote(gamma, inv(gamma - 1), c_h)
    new{typeof(γ)}(γ, inv_gamma_minus_one, c_h)
  end
end

function IdealGlmMhdEquations3D(gamma; initial_c_h=convert(typeof(gamma), NaN))
  # Use `promote` to ensure that `gamma` and `initial_c_h` have the same type
  IdealGlmMhdEquations3D(promote(gamma, initial_c_h)...)
end


have_nonconservative_terms(::IdealGlmMhdEquations3D) = True()
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


# Pre-defined source terms should be implemented as
# function source_terms_WHATEVER(u, x, t, equations::IdealGlmMhdEquations3D)


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
  mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
  p_over_gamma_minus_one = (rho_e - kin_en - mag_en - 0.5 * psi^2)
  p = (equations.gamma - 1) * p_over_gamma_minus_one
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1*v1 + p + mag_en - B1^2
    f3 = rho_v1*v2 - B1*B2
    f4 = rho_v1*v3 - B1*B3
    f5 = (kin_en + equations.gamma * p_over_gamma_minus_one + 2*mag_en)*v1 - B1*(v1*B1 + v2*B2 + v3*B3) + equations.c_h*psi*B1
    f6 = equations.c_h*psi
    f7 = v1*B2 - v2*B1
    f8 = v1*B3 - v3*B1
    f9 = equations.c_h*B1
  elseif orientation == 2
    f1 = rho_v2
    f2 = rho_v2*v1 - B2*B1
    f3 = rho_v2*v2 + p + mag_en - B2^2
    f4 = rho_v2*v3 - B2*B3
    f5 = (kin_en + equations.gamma * p_over_gamma_minus_one + 2*mag_en)*v2 - B2*(v1*B1 + v2*B2 + v3*B3) + equations.c_h*psi*B2
    f6 = v2*B1 - v1*B2
    f7 = equations.c_h*psi
    f8 = v2*B3 - v3*B2
    f9 = equations.c_h*B2
  else
    f1 = rho_v3
    f2 = rho_v3*v1 - B3*B1
    f3 = rho_v3*v2 - B3*B2
    f4 = rho_v3*v3 + p + mag_en - B3^2
    f5 = (kin_en + equations.gamma * p_over_gamma_minus_one + 2*mag_en)*v3 - B3*(v1*B1 + v2*B2 + v3*B3) + equations.c_h*psi*B3
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
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
  mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
  p_over_gamma_minus_one = (rho_e - kin_en - mag_en - 0.5 * psi^2)
  p = (equations.gamma - 1) * p_over_gamma_minus_one

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2] + v3 * normal_direction[3]
  B_normal = B1 * normal_direction[1] + B2 * normal_direction[2] + B3 * normal_direction[3]
  rho_v_normal = rho * v_normal

  f1 = rho_v_normal
  f2 = rho_v_normal * v1 - B1 * B_normal + (p + mag_en) * normal_direction[1]
  f3 = rho_v_normal * v2 - B2 * B_normal + (p + mag_en) * normal_direction[2]
  f4 = rho_v_normal * v3 - B3 * B_normal + (p + mag_en) * normal_direction[3]
  f5 = ( (kin_en + equations.gamma * p_over_gamma_minus_one + 2*mag_en) * v_normal
        - B_normal * (v1*B1 + v2*B2 + v3*B3) + equations.c_h * psi * B_normal )
  f6 = ( equations.c_h * psi * normal_direction[1] +
         (v2 * B1 - v1 * B2) * normal_direction[2] +
         (v3 * B1 - v1 * B3) * normal_direction[3] )
  f7 = ( (v1 * B2 - v2 * B1) * normal_direction[1] +
         equations.c_h * psi * normal_direction[2] +
         (v3 * B2 - v2 * B3) * normal_direction[3] )
  f8 = ( (v1 * B3 - v3 * B1) * normal_direction[1] +
         (v2 * B3 - v3 * B2) * normal_direction[2] +
         equations.c_h * psi * normal_direction[3] )
  f9 = equations.c_h * B_normal

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end



"""
    flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
                                equations::IdealGlmMhdEquations3D)
    flux_nonconservative_powell(u_ll, u_rr,
                                normal_direction_ll     ::AbstractVector,
                                normal_direction_average::AbstractVector,
                                equations::IdealGlmMhdEquations3D)

Non-symmetric two-point flux discretizing the nonconservative (source) term of
Powell and the Galilean nonconservative term associated with the GLM multiplier
of the [`IdealGlmMhdEquations3D`](@ref).

On curvilinear meshes, this nonconservative flux depends on both the
contravariant vector (normal direction) at the current node and the averaged
one. This is different from numerical fluxes used to discretize conservative
terms.

## References
- Marvin Bohm, Andrew R.Winters, Gregor J. Gassner, Dominik Derigs,
  Florian Hindenlang, Joachim Saur
  An entropy stable nodal discontinuous Galerkin method for the resistive MHD
  equations. Part I: Theory and numerical verification
  [DOI: 10.1016/j.jcp.2018.06.027](https://doi.org/10.1016/j.jcp.2018.06.027)
"""
@inline function flux_nonconservative_powell(u_ll, u_rr, orientation::Integer,
                                             equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

  # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
  # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
  if orientation == 1
    f = SVector(0,
                B1_ll      * B1_rr,
                B2_ll      * B1_rr,
                B3_ll      * B1_rr,
                v_dot_B_ll * B1_rr + v1_ll * psi_ll * psi_rr,
                v1_ll      * B1_rr,
                v2_ll      * B1_rr,
                v3_ll      * B1_rr,
                                     v1_ll * psi_rr)
  elseif orientation == 2
    f = SVector(0,
                B1_ll      * B2_rr,
                B2_ll      * B2_rr,
                B3_ll      * B2_rr,
                v_dot_B_ll * B2_rr + v2_ll * psi_ll * psi_rr,
                v1_ll      * B2_rr,
                v2_ll      * B2_rr,
                v3_ll      * B2_rr,
                                     v2_ll * psi_rr)
  else # orientation == 3
    f = SVector(0,
                B1_ll      * B3_rr,
                B2_ll      * B3_rr,
                B3_ll      * B3_rr,
                v_dot_B_ll * B3_rr + v3_ll * psi_ll * psi_rr,
                v1_ll      * B3_rr,
                v2_ll      * B3_rr,
                v3_ll      * B3_rr,
                                     v3_ll * psi_rr)
  end

  return f
end

@inline function flux_nonconservative_powell(u_ll, u_rr,
                                             normal_direction_ll::AbstractVector,
                                             normal_direction_average::AbstractVector,
                                             equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_dot_B_ll = v1_ll * B1_ll + v2_ll * B2_ll + v3_ll * B3_ll

  # Note that `v_dot_n_ll` uses the `normal_direction_ll` (contravariant vector
  # at the same node location) while `B_dot_n_rr` uses the averaged normal
  # direction. The reason for this is that `v_dot_n_ll` depends only on the left
  # state and multiplies some gradient while `B_dot_n_rr` is used to compute
  # the divergence of B.
  v_dot_n_ll = v1_ll * normal_direction_ll[1]      + v2_ll * normal_direction_ll[2]      + v3_ll * normal_direction_ll[3]
  B_dot_n_rr = B1_rr * normal_direction_average[1] + B2_rr * normal_direction_average[2] + B3_rr * normal_direction_average[3]

  # Powell nonconservative term:   (0, B_1, B_2, B_3, v⋅B, v_1, v_2, v_3, 0)
  # Galilean nonconservative term: (0, 0, 0, 0, ψ v_{1,2,3}, 0, 0, 0, v_{1,2,3})
  f = SVector(0,
              B1_ll      * B_dot_n_rr,
              B2_ll      * B_dot_n_rr,
              B3_ll      * B_dot_n_rr,
              v_dot_B_ll * B_dot_n_rr + v_dot_n_ll * psi_ll * psi_rr,
              v1_ll      * B_dot_n_rr,
              v2_ll      * B_dot_n_rr,
              v3_ll      * B_dot_n_rr,
                                        v_dot_n_ll * psi_rr)

  return f
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
  rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr, equations)

  vel_norm_ll = v1_ll^2 + v2_ll^2 + v3_ll^2
  vel_norm_rr = v1_rr^2 + v2_rr^2 + v3_rr^2
  mag_norm_ll = B1_ll^2 + B2_ll^2 + B3_ll^2
  mag_norm_rr = B1_rr^2 + B2_rr^2 + B3_rr^2
  beta_ll = 0.5 * rho_ll / p_ll
  beta_rr = 0.5 * rho_rr / p_rr
  # for convenience store v⋅B
  vel_dot_mag_ll = v1_ll*B1_ll + v2_ll*B2_ll + v3_ll*B3_ll
  vel_dot_mag_rr = v1_rr*B1_rr + v2_rr*B2_rr + v3_rr*B3_rr

  # Compute the necessary mean values needed for either direction
  rho_avg = 0.5 * (rho_ll + rho_rr)
  rho_mean  = ln_mean(rho_ll, rho_rr)
  beta_mean = ln_mean(beta_ll, beta_rr)
  beta_avg = 0.5 * (beta_ll + beta_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  v3_avg = 0.5 * (v3_ll + v3_rr)
  p_mean = 0.5 * rho_avg / beta_avg
  B1_avg = 0.5 * (B1_ll + B1_rr)
  B2_avg = 0.5 * (B2_ll + B2_rr)
  B3_avg = 0.5 * (B3_ll + B3_rr)
  psi_avg = 0.5 * (psi_ll + psi_rr)
  vel_norm_avg = 0.5 * (vel_norm_ll + vel_norm_rr)
  mag_norm_avg = 0.5 * (mag_norm_ll + mag_norm_rr)
  vel_dot_mag_avg = 0.5 * (vel_dot_mag_ll + vel_dot_mag_rr)

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
    flux_hindenlang_gassner(u_ll, u_rr, orientation_or_normal_direction,
                            equations::IdealGlmMhdEquations3D)

Entropy conserving and kinetic energy preserving two-point flux of
Hindenlang and Gassner (2019), extending [`flux_ranocha`](@ref) to the MHD equations.

## References
- Florian Hindenlang, Gregor Gassner (2019)
  A new entropy conservative two-point flux for ideal MHD equations derived from
  first principles.
  Presented at HONOM 2019: European workshop on high order numerical methods
  for evolutionary PDEs, theory and applications
- Hendrik Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
  [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
- Hendrik Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
  [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_hindenlang_gassner(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  # Unpack left and right states
  rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr, equations)

  # Compute the necessary mean values needed for either direction
  rho_mean = ln_mean(rho_ll, rho_rr)
  # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
  # in exact arithmetic since
  #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
  #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
  inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
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
    f5 = ( f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one )
          + 0.5 * (
            +   p_ll * v1_rr +  p_rr * v1_ll
            + (v1_ll * B2_ll * B2_rr + v1_rr * B2_rr * B2_ll)
            + (v1_ll * B3_ll * B3_rr + v1_rr * B3_rr * B3_ll)
            - (v2_ll * B1_ll * B2_rr + v2_rr * B1_rr * B2_ll)
            - (v3_ll * B1_ll * B3_rr + v3_rr * B1_rr * B3_ll)
            + equations.c_h * (B1_ll * psi_rr + B1_rr * psi_ll) ) )
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
    f5 = ( f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one )
          + 0.5 * (
            +   p_ll * v2_rr +  p_rr * v2_ll
            + (v2_ll * B1_ll * B1_rr + v2_rr * B1_rr * B1_ll)
            + (v2_ll * B3_ll * B3_rr + v2_rr * B3_rr * B3_ll)
            - (v1_ll * B2_ll * B1_rr + v1_rr * B2_rr * B1_ll)
            - (v3_ll * B2_ll * B3_rr + v3_rr * B2_rr * B3_ll)
            + equations.c_h * (B2_ll * psi_rr + B2_rr * psi_ll) ) )
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
    f5 = ( f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one )
          + 0.5 * (
            +   p_ll * v3_rr +  p_rr * v3_ll
            + (v3_ll * B1_ll * B1_rr + v3_rr * B1_rr * B1_ll)
            + (v3_ll * B2_ll * B2_rr + v3_rr * B2_rr * B2_ll)
            - (v1_ll * B3_ll * B1_rr + v1_rr * B3_rr * B1_ll)
            - (v2_ll * B3_ll * B2_rr + v2_rr * B3_rr * B2_ll)
            + equations.c_h * (B3_ll * psi_rr + B3_rr * psi_ll) ) )
  end

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

@inline function flux_hindenlang_gassner(u_ll, u_rr, normal_direction::AbstractVector, equations::IdealGlmMhdEquations3D)
  # Unpack left and right states
  rho_ll, v1_ll, v2_ll, v3_ll, p_ll, B1_ll, B2_ll, B3_ll, psi_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, v3_rr, p_rr, B1_rr, B2_rr, B3_rr, psi_rr = cons2prim(u_rr, equations)
  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2] + v3_ll * normal_direction[3]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2] + v3_rr * normal_direction[3]
  B_dot_n_ll = B1_ll * normal_direction[1] + B2_ll * normal_direction[2] + B3_ll * normal_direction[3]
  B_dot_n_rr = B1_rr * normal_direction[1] + B2_rr * normal_direction[2] + B3_rr * normal_direction[3]

  # Compute the necessary mean values needed for either direction
  rho_mean = ln_mean(rho_ll, rho_rr)
  # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
  # in exact arithmetic since
  #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
  #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
  inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
  v1_avg  = 0.5 * ( v1_ll +  v1_rr)
  v2_avg  = 0.5 * ( v2_ll +  v2_rr)
  v3_avg  = 0.5 * ( v3_ll +  v3_rr)
  p_avg   = 0.5 * (  p_ll +   p_rr)
  psi_avg = 0.5 * (psi_ll + psi_rr)
  velocity_square_avg = 0.5 * (v1_ll * v1_rr + v2_ll * v2_rr + v3_ll * v3_rr)
  magnetic_square_avg = 0.5 * (B1_ll * B1_rr + B2_ll * B2_rr + B3_ll * B3_rr)

  # Calculate fluxes depending on normal_direction
  f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
  f2 = ( f1 * v1_avg + (p_avg + magnetic_square_avg) * normal_direction[1]
        - 0.5 * (B_dot_n_ll * B1_rr + B_dot_n_rr * B1_ll) )
  f3 = ( f1 * v2_avg + (p_avg + magnetic_square_avg) * normal_direction[2]
        - 0.5 * (B_dot_n_ll * B2_rr + B_dot_n_rr * B2_ll) )
  f4 = ( f1 * v3_avg + (p_avg + magnetic_square_avg) * normal_direction[3]
        - 0.5 * (B_dot_n_ll * B3_rr + B_dot_n_rr * B3_ll) )
  #f5 below
  f6 = ( equations.c_h * psi_avg * normal_direction[1]
        + 0.5 * (v_dot_n_ll * B1_ll - v1_ll * B_dot_n_ll +
                 v_dot_n_rr * B1_rr - v1_rr * B_dot_n_rr) )
  f7 = ( equations.c_h * psi_avg * normal_direction[2]
        + 0.5 * (v_dot_n_ll * B2_ll - v2_ll * B_dot_n_ll +
                 v_dot_n_rr * B2_rr - v2_rr * B_dot_n_rr) )
  f8 = ( equations.c_h * psi_avg * normal_direction[3]
        + 0.5 * (v_dot_n_ll * B3_ll - v3_ll * B_dot_n_ll +
                 v_dot_n_rr * B3_rr - v3_rr * B_dot_n_rr) )
  f9 = equations.c_h * 0.5 * (B_dot_n_ll + B_dot_n_rr)
  # total energy flux is complicated and involves the previous components
  f5 = ( f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one )
        + 0.5 * (
          +   p_ll * v_dot_n_rr +  p_rr * v_dot_n_ll
          + (v_dot_n_ll * B1_ll * B1_rr + v_dot_n_rr * B1_rr * B1_ll)
          + (v_dot_n_ll * B2_ll * B2_rr + v_dot_n_rr * B2_rr * B2_ll)
          + (v_dot_n_ll * B3_ll * B3_rr + v_dot_n_rr * B3_rr * B3_ll)
          - (v1_ll * B_dot_n_ll * B1_rr + v1_rr * B_dot_n_rr * B1_ll)
          - (v2_ll * B_dot_n_ll * B2_rr + v2_rr * B_dot_n_rr * B2_ll)
          - (v3_ll * B_dot_n_ll * B3_rr + v3_rr * B_dot_n_rr * B3_ll)
          + equations.c_h * (B_dot_n_ll * psi_rr + B_dot_n_rr * psi_ll) ) )

  return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _ = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, _ = u_rr

  # Calculate the left/right velocities and fast magnetoacoustic wave speeds
  if orientation == 1
    v_ll = rho_v1_ll / rho_ll
    v_rr = rho_v1_rr / rho_rr
  elseif orientation == 2
    v_ll = rho_v2_ll / rho_ll
    v_rr = rho_v2_rr / rho_rr
  else # orientation == 3
    v_ll = rho_v3_ll / rho_ll
    v_rr = rho_v3_rr / rho_rr
  end
  cf_ll = calc_fast_wavespeed(u_ll, orientation, equations)
  cf_rr = calc_fast_wavespeed(u_rr, orientation, equations)

  return max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end

@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _ = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, _ = u_rr

  # Calculate normal velocities and fast magnetoacoustic wave speeds
  # left
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  v_ll = (  v1_ll * normal_direction[1]
          + v2_ll * normal_direction[2]
          + v3_ll * normal_direction[3] )
  cf_ll = calc_fast_wavespeed(u_ll, normal_direction, equations)
  # right
  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  v_rr = (  v1_rr * normal_direction[1]
          + v2_rr * normal_direction[2]
          + v3_rr * normal_direction[3] )
  cf_rr = calc_fast_wavespeed(u_rr, normal_direction, equations)

  # wave speeds already scaled by norm(normal_direction) in [`calc_fast_wavespeed`](@ref)
  return max(abs(v_ll), abs(v_rr)) + max(cf_ll, cf_rr)
end


"""
    min_max_speed_naive(u_ll, u_rr, orientation_or_normal_direction, equations::IdealGlmMhdEquations3D)

Calculate minimum and maximum wave speeds for HLL-type fluxes as in
- Li (2005)
  An HLLC Riemann solver for magneto-hydrodynamics
  [DOI: 10.1016/j.jcp.2004.08.020](https://doi.org/10.1016/j.jcp.2004.08.020)
"""
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _ = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, _ = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr

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
    λ_min = min(v3_ll - c_f_ll, vel_roe - c_f_roe)
    λ_max = max(v3_rr + c_f_rr, vel_roe + c_f_roe)
  end

  return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, _ = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, _ = u_rr

  # Calculate primitive velocity variables
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr

  v_normal_ll = (v1_ll * normal_direction[1] +
                 v2_ll * normal_direction[2] +
                 v3_ll * normal_direction[3])
  v_normal_rr = (v1_rr * normal_direction[1] +
                 v2_rr * normal_direction[2] +
                 v3_rr * normal_direction[3])

  c_f_ll = calc_fast_wavespeed(u_ll, normal_direction, equations)
  c_f_rr = calc_fast_wavespeed(u_rr, normal_direction, equations)
  v_roe, c_f_roe = calc_fast_wavespeed_roe(u_ll, u_rr, normal_direction, equations)

  # Estimate the min/max eigenvalues in the normal direction
  λ_min = min(v_normal_ll - c_f_ll, v_roe - c_f_roe)
  λ_max = max(v_normal_rr + c_f_rr, v_roe + c_f_roe)

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
@inline function cons2prim(u, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3
                                              + B1 * B1 + B2 * B2 + B3 * B3
                                              + psi * psi))

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

  w1 = (equations.gamma - s) * equations.inv_gamma_minus_one - 0.5 * rho_p * v_square
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
  rho_e = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1*v1 + rho_v2*v2 + rho_v3*v3) +
                                  0.5 * (B1^2 + B2^2 + B3^2) + 0.5 * psi^2

  return SVector(rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi)
end


@inline function density(u, equations::IdealGlmMhdEquations3D)
  return u[1]
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
@inline function calc_fast_wavespeed(cons, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
  mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
  p = (equations.gamma - 1) * (rho_e - kin_en - mag_en - 0.5 * psi^2)
  a_square = equations.gamma * p / rho
  sqrt_rho = sqrt(rho)
  b1 = B1 / sqrt_rho
  b2 = B2 / sqrt_rho
  b3 = B3 / sqrt_rho
  b_square = b1 * b1 + b2 * b2 + b3 * b3
  if orientation == 1 # x-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b1^2))
  elseif orientation == 2 # y-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b2^2))
  else # z-direction
    c_f = sqrt(0.5*(a_square + b_square) + 0.5*sqrt((a_square + b_square)^2 - 4.0*a_square*b3^2))
  end
  return c_f
end

@inline function calc_fast_wavespeed(cons, normal_direction::AbstractVector, equations::IdealGlmMhdEquations3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, B1, B2, B3, psi = cons
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  kin_en = 0.5 * (rho_v1 * v1 + rho_v2 * v2 + rho_v3 * v3)
  mag_en = 0.5 * (B1 * B1 + B2 * B2 + B3 * B3)
  p = (equations.gamma - 1) * (rho_e - kin_en - mag_en - 0.5 * psi^2)
  a_square = equations.gamma * p / rho
  sqrt_rho = sqrt(rho)
  b1 = B1 / sqrt_rho
  b2 = B2 / sqrt_rho
  b3 = B3 / sqrt_rho
  b_square = b1 * b1 + b2 * b2 + b3 * b3
  norm_squared = (normal_direction[1] * normal_direction[1] +
                  normal_direction[2] * normal_direction[2] +
                  normal_direction[3] * normal_direction[3])
  b_dot_n_squared = (b1 * normal_direction[1] +
                     b2 * normal_direction[2] +
                     b3 * normal_direction[3])^2 / norm_squared

  c_f = sqrt(
    (0.5 * (a_square + b_square) +
     0.5 * sqrt((a_square + b_square)^2 - 4 * a_square * b_dot_n_squared)) * norm_squared)
  return c_f
end


"""
    calc_fast_wavespeed_roe(u_ll, u_rr, orientation_or_normal_direction, equations::IdealGlmMhdEquations3D)

Compute the fast magnetoacoustic wave speed using Roe averages as given by
- Cargo and Gallice (1997)
  Roe Matrices for Ideal MHD and Systematic Construction
  of Roe Matrices for Systems of Conservation Laws
  [DOI: 10.1006/jcph.1997.5773](https://doi.org/10.1006/jcph.1997.5773)
"""
@inline function calc_fast_wavespeed_roe(u_ll, u_rr, orientation::Integer, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate primitive variables
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  kin_en_ll = 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll + rho_v3_ll * v3_ll)
  mag_norm_ll = B1_ll * B1_ll + B2_ll * B2_ll + B3_ll * B3_ll
  p_ll = (equations.gamma - 1)*(rho_e_ll - kin_en_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  kin_en_rr = 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr + rho_v3_rr * v3_rr)
  mag_norm_rr = B1_rr * B1_rr + B2_rr * B2_rr + B3_rr * B3_rr
  p_rr = (equations.gamma - 1)*(rho_e_rr - kin_en_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

  # compute total pressure which is thermal + magnetic pressures
  p_total_ll = p_ll + 0.5 * mag_norm_ll
  p_total_rr = p_rr + 0.5 * mag_norm_rr

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
  # temporary variable see equation (4.12) in Cargo and Gallice
  X = 0.5 * ( (B1_ll - B1_rr)^2 + (B2_ll - B2_rr)^2 + (B3_ll - B3_rr)^2 ) * inv_sqrt_rho_add^2
  # averaged components needed to compute c_f, the fast magnetoacoustic wave speed
  b_square_roe = (B1_roe^2 + B2_roe^2 + B3_roe^2) * inv_sqrt_rho_prod # scaled magnectic sum
  a_square_roe = ((2.0 - equations.gamma) * X +
                 (equations.gamma -1.0) * (H_roe - 0.5*(v1_roe^2 + v2_roe^2 + v3_roe^2) -
                                          b_square_roe)) # acoustic speed
  # finally compute the average wave speed and set the output velocity (depends on orientation)
  if orientation == 1 # x-direction
    c_a_roe = B1_roe^2 * inv_sqrt_rho_prod # (squared) Alfvén wave speed
    a_star_roe = sqrt( (a_square_roe + b_square_roe)^2 - 4.0 * a_square_roe * c_a_roe )
    c_f_roe = sqrt( 0.5 * (a_square_roe + b_square_roe + a_star_roe) )
    vel_out_roe = v1_roe
  elseif orientation == 2 # y-direction
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

@inline function calc_fast_wavespeed_roe(u_ll, u_rr, normal_direction::AbstractVector, equations::IdealGlmMhdEquations3D)
  rho_ll, rho_v1_ll, rho_v2_ll, rho_v3_ll, rho_e_ll, B1_ll, B2_ll, B3_ll, psi_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_v3_rr, rho_e_rr, B1_rr, B2_rr, B3_rr, psi_rr = u_rr

  # Calculate primitive variables
  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  v3_ll = rho_v3_ll / rho_ll
  kin_en_ll = 0.5 * (rho_v1_ll * v1_ll + rho_v2_ll * v2_ll + rho_v3_ll * v3_ll)
  mag_norm_ll = B1_ll * B1_ll + B2_ll * B2_ll + B3_ll * B3_ll
  p_ll = (equations.gamma - 1)*(rho_e_ll - kin_en_ll - 0.5*mag_norm_ll - 0.5*psi_ll^2)

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  v3_rr = rho_v3_rr / rho_rr
  kin_en_rr = 0.5 * (rho_v1_rr * v1_rr + rho_v2_rr * v2_rr + rho_v3_rr * v3_rr)
  mag_norm_rr = B1_rr * B1_rr + B2_rr * B2_rr + B3_rr * B3_rr
  p_rr = (equations.gamma - 1)*(rho_e_rr - kin_en_rr - 0.5*mag_norm_rr - 0.5*psi_rr^2)

  # compute total pressure which is thermal + magnetic pressures
  p_total_ll = p_ll + 0.5 * mag_norm_ll
  p_total_rr = p_rr + 0.5 * mag_norm_rr

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
  # temporary variable see equation (4.12) in Cargo and Gallice
  X = 0.5 * ( (B1_ll - B1_rr)^2 + (B2_ll - B2_rr)^2 + (B3_ll - B3_rr)^2 ) * inv_sqrt_rho_add^2
  # averaged components needed to compute c_f, the fast magnetoacoustic wave speed
  b_square_roe = (B1_roe^2 + B2_roe^2 + B3_roe^2) * inv_sqrt_rho_prod # scaled magnectic sum
  a_square_roe = ((2.0 - equations.gamma) * X +
                 (equations.gamma -1.0) * (H_roe - 0.5*(v1_roe^2 + v2_roe^2 + v3_roe^2) -
                                          b_square_roe)) # acoustic speed

  # finally compute the average wave speed and set the output velocity (depends on orientation)
  norm_squared = (normal_direction[1] * normal_direction[1] +
                  normal_direction[2] * normal_direction[2] +
                  normal_direction[3] * normal_direction[3])
  B_roe_dot_n_squared = (B1_roe * normal_direction[1] +
                         B2_roe * normal_direction[2] +
                         B3_roe * normal_direction[3])^2 / norm_squared

  c_a_roe = B_roe_dot_n_squared * inv_sqrt_rho_prod # (squared) Alfvén wave speed
  a_star_roe = sqrt((a_square_roe + b_square_roe)^2 - 4 * a_square_roe * c_a_roe)
  c_f_roe = sqrt(0.5 * (a_square_roe + b_square_roe + a_star_roe) * norm_squared)
  vel_out_roe = (v1_roe * normal_direction[1] +
                 v2_roe * normal_direction[2] +
                 v3_roe * normal_direction[3])

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
  S = -entropy_thermodynamic(cons, equations) * cons[1] * equations.inv_gamma_minus_one

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


# Calculate the cross helicity (\vec{v}⋅\vec{B}) for a conservative state `cons'
@inline function cross_helicity(cons, ::IdealGlmMhdEquations3D)
  return (cons[2]*cons[6] + cons[3]*cons[7] + cons[4]*cons[8]) / cons[1]
end


end # @muladd
