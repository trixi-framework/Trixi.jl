# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    CompressibleEulerEquations1D(gamma)

The compressible Euler equations
```math
\partial t
\begin{pmatrix}
\rho \\ \rho v_1 \\ \rho e
\end{pmatrix}
+
\partial x
\begin{pmatrix}
\rho v_1 \\ \rho v_1^2 + p \\ (\rho e +p) v_1
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0
\end{pmatrix}
```
for an ideal gas with ratio of specific heats `gamma` in one space dimension.
Here, ``\rho`` is the density, ``v_1`` the velocity, ``e`` the specific total energy **rather than** specific internal energy, and
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho v_1^2 \right)
```
the pressure.
"""
struct CompressibleEulerEquations1D{RealT<:Real} <: AbstractCompressibleEulerEquations{1, 3}
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

  function CompressibleEulerEquations1D(gamma)
    γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
    new{typeof(γ)}(γ, inv_gamma_minus_one)
  end
end


varnames(::typeof(cons2cons), ::CompressibleEulerEquations1D) = ("rho", "rho_v1", "rho_e")
varnames(::typeof(cons2prim), ::CompressibleEulerEquations1D) = ("rho", "v1", "p")


"""
    initial_condition_constant(x, t, equations::CompressibleEulerEquations1D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::CompressibleEulerEquations1D)
  rho = 1.0
  rho_v1 = 0.1
  rho_e = 10.0
  return SVector(rho, rho_v1, rho_e)
end


"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations1D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations1D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] - t))

  rho = ini
  rho_v1 = ini
  rho_e = ini^2

  return SVector(rho, rho_v1, rho_e)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations1D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations1D)
  # Same settings as in `initial_condition`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equations.gamma

  x1, = x

  si, co = sincos(ω * (x1 - t))
  rho = c + A * si
  rho_x = ω * A * co

  # Note that d/dt rho = -d/dx rho.
  # This yields du2 = du3 = d/dx p (derivative of pressure).
  # Other terms vanish because of v = 1.
  du1 = zero(eltype(u))
  du2 = rho_x * (2 * rho - 0.5) * (γ - 1)
  du3 = du2

  return SVector(du1, du2, du3)
end


"""
    initial_condition_density_wave(x, t, equations::CompressibleEulerEquations1D)

A sine wave in the density with constant velocity and pressure; reduces the
compressible Euler equations to the linear advection equations.
This setup is the test case for stability of EC fluxes from paper
- Gregor J. Gassner, Magnus Svärd, Florian J. Hindenlang (2020)
  Stability issues of entropy-stable and/or split-form high-order schemes
  [arXiv: 2007.09026](https://arxiv.org/abs/2007.09026)
with the following parameters
- domain [-1, 1]
- mesh = 4x4
- polydeg = 5
"""
function initial_condition_density_wave(x, t, equations::CompressibleEulerEquations1D)
  v1 = 0.1
  rho = 1 + 0.98 * sinpi(2 * (x[1] - t * v1))
  rho_v1 = rho * v1
  p = 20
  rho_e = p / (equations.gamma - 1) + 1/2 * rho * v1^2
  return SVector(rho, rho_v1, rho_e)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations1D)

A weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations1D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter = SVector(0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)
  # The following code is equivalent to
  # phi = atan(0.0, x_norm)
  # cos_phi = cos(phi)
  # in 1D but faster
  cos_phi = x_norm > 0 ? one(x_norm) : -one(x_norm)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  p   = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, p), equations)
end


"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::CompressibleEulerEquations1D)

One dimensional variant of the setup used for convergence tests of the Euler equations
with self-gravity from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
!!! note
    There is no additional source term necessary for the manufactured solution in one
    spatial dimension. Thus, [`source_terms_eoc_test_coupled_euler_gravity`](@ref) is not
    present there.
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::CompressibleEulerEquations1D)
  # OBS! this assumes that γ = 2 other manufactured source terms are incorrect
  if equations.gamma != 2.0
    error("adiabatic constant must be 2 for the coupling convergence test")
  end
  c = 2.0
  A = 0.1
  ini = c + A * sinpi(x[1] - t)
  G = 1.0 # gravitational constant

  rho = ini
  v1 = 1.0
  p = 2 * ini^2 * G / pi # * 2 / ndims, but ndims==1 here

  return prim2cons(SVector(rho, v1, p), equations)
end


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u
  v1 = rho_v1 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho_v1 * v1)
  # Ignore orientation since it is always "1" in 1D
  f1 = rho_v1
  f2 = rho_v1 * v1 + p
  f3 = (rho_e + p) * v1
  return SVector(f1, f2, f3)
end


"""
    flux_shima_etal(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)

This flux is is a modification of the original kinetic energy preserving two-point flux by
- Yuichi Kuya, Kosuke Totani and Soshi Kawai (2018)
  Kinetic energy and entropy preserving schemes for compressible flows
  by split convective forms
  [DOI: 10.1016/j.jcp.2018.08.058](https://doi.org/10.1016/j.jcp.2018.08.058)

The modification is in the energy flux to guarantee pressure equilibrium and was developed by
- Nao Shima, Yuichi Kuya, Yoshiharu Tamaki, Soshi Kawai (JCP 2020)
  Preventing spurious pressure oscillations in split convective form discretizations for
  compressible flows
  [DOI: 10.1016/j.jcp.2020.110060](https://doi.org/10.1016/j.jcp.2020.110060)
"""
@inline function flux_shima_etal(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  kin_avg = 1/2 * (v1_ll * v1_rr)

  # Calculate fluxes
  # Ignore orientation since it is always "1" in 1D
  pv1_avg = 1/2 * (p_ll*v1_rr + p_rr*v1_ll)
  f1 = rho_avg * v1_avg
  f2 = rho_avg * v1_avg * v1_avg + p_avg
  f3 = p_avg*v1_avg * equations.inv_gamma_minus_one + rho_avg*v1_avg*kin_avg + pv1_avg

  return SVector(f1, f2, f3)
end


"""
    flux_kennedy_gruber(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
  [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_e_ll = last(u_ll)
  rho_e_rr = last(u_rr)
  rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  e_avg   = 1/2 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

  # Ignore orientation since it is always "1" in 1D
  f1 = rho_avg * v1_avg
  f2 = rho_avg * v1_avg * v1_avg + p_avg
  f3 = (rho_avg * e_avg + p_avg) * v1_avg

  return SVector(f1, f2, f3)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)

Entropy conserving two-point flux by
- Chandrashekar (2013)
  Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
  for Compressible Euler and Navier-Stokes Equations
  [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)
  beta_ll = 0.5 * rho_ll / p_ll
  beta_rr = 0.5 * rho_rr / p_rr
  specific_kin_ll = 0.5 * (v1_ll^2)
  specific_kin_rr = 0.5 * (v1_rr^2)

  # Compute the necessary mean values
  rho_avg = 0.5 * (rho_ll + rho_rr)
  rho_mean  = ln_mean(rho_ll, rho_rr)
  beta_mean = ln_mean(beta_ll, beta_rr)
  beta_avg = 0.5 * (beta_ll + beta_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  p_mean = 0.5 * rho_avg / beta_avg
  velocity_square_avg = specific_kin_ll + specific_kin_rr

  # Calculate fluxes
  # Ignore orientation since it is always "1" in 1D
  f1 = rho_mean * v1_avg
  f2 = f1 * v1_avg + p_mean
  f3 = f1 * 0.5*(1/(equations.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg

  return SVector(f1, f2, f3)
end


"""
    flux_ranocha(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)

Entropy conserving and kinetic energy preserving two-point flux by
- Hendrik Ranocha (2018)
  Generalised Summation-by-Parts Operators and Entropy Stability of Numerical Methods
  for Hyperbolic Balance Laws
  [PhD thesis, TU Braunschweig](https://cuvillier.de/en/shop/publications/7743)
See also
- Hendrik Ranocha (2020)
  Entropy Conserving and Kinetic Energy Preserving Numerical Methods for
  the Euler Equations Using Summation-by-Parts Operators
  [Proceedings of ICOSAHOM 2018](https://doi.org/10.1007/978-3-030-39647-3_42)
"""
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  # Unpack left and right state
  rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

  # Compute the necessary mean values
  rho_mean = ln_mean(rho_ll, rho_rr)
  # Algebraically equivalent to `inv_ln_mean(rho_ll / p_ll, rho_rr / p_rr)`
  # in exact arithmetic since
  #     log((ϱₗ/pₗ) / (ϱᵣ/pᵣ)) / (ϱₗ/pₗ - ϱᵣ/pᵣ)
  #   = pₗ pᵣ log((ϱₗ pᵣ) / (ϱᵣ pₗ)) / (ϱₗ pᵣ - ϱᵣ pₗ)
  inv_rho_p_mean = p_ll * p_rr * inv_ln_mean(rho_ll * p_rr, rho_rr * p_ll)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  p_avg  = 0.5 * (p_ll + p_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr)

  # Calculate fluxes
  # Ignore orientation since it is always "1" in 1D
  f1 = rho_mean * v1_avg
  f2 = f1 * v1_avg + p_avg
  f3 = f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)

  return SVector(f1, f2, f3)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  # Calculate primitive variables and speed of sound
  v1_ll = rho_v1_ll / rho_ll
  v_mag_ll = abs(v1_ll)
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v_mag_ll^2)
  c_ll = sqrt(equations.gamma * p_ll / rho_ll)
  v1_rr = rho_v1_rr / rho_rr
  v_mag_rr = abs(v1_rr)
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v_mag_rr^2)
  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

  λ_max = max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  rho_ll, v1_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, p_rr = cons2prim(u_rr, equations)

  λ_min = v1_ll - sqrt(equations.gamma * p_ll / rho_ll)
  λ_max = v1_rr + sqrt(equations.gamma * p_rr / rho_rr)

  return λ_min, λ_max
end


"""
    flux_hllc(u_ll, u_rr, orientation, equations::CompressibleEulerEquations1D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations1D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  e_ll  = rho_e_ll / rho_ll
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * v1_ll^2)
  c_ll = sqrt(equations.gamma*p_ll/rho_ll)

  v1_rr = rho_v1_rr / rho_rr
  e_rr  = rho_e_rr / rho_rr
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * v1_rr^2 )
  c_rr = sqrt(equations.gamma*p_rr/rho_rr)

  # Obtain left and right fluxes
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # Compute Roe averages
  sqrt_rho_ll = sqrt(rho_ll)
  sqrt_rho_rr = sqrt(rho_rr)
  sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
  vel_L = v1_ll
  vel_R = v1_rr
  vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
  ekin_roe = 0.5 * vel_roe^2
  H_ll = (rho_e_ll + p_ll) / rho_ll
  H_rr = (rho_e_rr + p_rr) / rho_rr
  H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) / sum_sqrt_rho
  c_roe = sqrt((equations.gamma - 1) * (H_roe - ekin_roe))

  Ssl = min(vel_L - c_ll, vel_roe - c_roe)
  Ssr = max(vel_R + c_rr, vel_roe + c_roe)
  sMu_L = Ssl - vel_L
  sMu_R = Ssr - vel_R
  if Ssl >= 0.0
    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
  elseif Ssr <= 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
  else
    SStar = (p_rr - p_ll + rho_ll*vel_L*sMu_L - rho_rr*vel_R*sMu_R) / (rho_ll*sMu_L - rho_rr*sMu_R)
    if Ssl <= 0.0 <= SStar
      densStar = rho_ll*sMu_L / (Ssl-SStar)
      enerStar = e_ll + (SStar - vel_L) * (SStar + p_ll / (rho_ll * sMu_L))
      UStar1 = densStar
      UStar2 = densStar*SStar
      UStar3 = densStar*enerStar

      f1 = f_ll[1]+Ssl*(UStar1 - rho_ll)
      f2 = f_ll[2]+Ssl*(UStar2 - rho_v1_ll)
      f3 = f_ll[3]+Ssl*(UStar3 - rho_e_ll)
    else
      densStar = rho_rr*sMu_R / (Ssr-SStar)
      enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
      UStar1 = densStar
      UStar2 = densStar*SStar
      UStar3 = densStar*enerStar

      #end
      f1 = f_rr[1]+Ssr*(UStar1 - rho_rr)
      f2 = f_rr[2]+Ssr*(UStar2 - rho_v1_rr)
      f3 = f_rr[3]+Ssr*(UStar3 - rho_e_rr)
    end
  end
  return SVector(f1, f2, f3)
end




@inline function max_abs_speeds(u, equations::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u
  v1 = rho_v1 / rho
  p = (equations.gamma - 1) * (rho_e - 1/2 * rho * v1^2)
  c = sqrt(equations.gamma * p / rho)

  return (abs(v1) + c,)
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u

  v1 = rho_v1 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho_v1 * v1)

  return SVector(rho, v1, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerEquations1D)
  rho, rho_v1, rho_e = u

  v1 = rho_v1 / rho
  v_square = v1^2
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s = log(p) - equations.gamma*log(rho)
  rho_p = rho / p

  w1 = (equations.gamma - s) * equations.inv_gamma_minus_one - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = -rho_p

  return SVector(w1, w2, w3)
end

@inline function entropy2cons(w, equations::CompressibleEulerEquations1D)
  # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
  # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
  @unpack gamma = equations

  # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
  # instead of `-rho * s / (gamma - 1)`
  V1, V2, V5 = w .* (gamma - 1)

  # specific entropy, eq. (53)
  s = gamma - V1 + 0.5 * (V2^2) / V5

  # eq. (52)
  energy_internal = ((gamma - 1) / (-V5)^gamma)^(equations.inv_gamma_minus_one) * exp(-s * equations.inv_gamma_minus_one)

  # eq. (51)
  rho    = -V5 * energy_internal
  rho_v1 = V2 * energy_internal
  rho_e  = (1 - 0.5 * (V2^2) / V5) * energy_internal
  return SVector(rho, rho_v1, rho_e)
end


# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerEquations1D)
  rho, v1, p = prim
  rho_v1 = rho * v1
  rho_e  = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1 * v1)
  return SVector(rho, rho_v1, rho_e)
end


@inline function density(u, equations::CompressibleEulerEquations1D)
 rho = u[1]
 return rho
end

@inline function pressure(u, equations::CompressibleEulerEquations1D)
 rho, rho_v1, rho_e = u
 p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2) / rho)
 return p
end


@inline function density_pressure(u, equations::CompressibleEulerEquations1D)
 rho, rho_v1, rho_e = u
 rho_times_p = (equations.gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2))
 return rho_times_p
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::CompressibleEulerEquations1D)
  # Pressure
  p = (equations.gamma - 1) * (cons[3] - 1/2 * (cons[2]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::CompressibleEulerEquations1D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) * cons[1] * equations.inv_gamma_minus_one

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleEulerEquations1D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquations1D) = cons[3]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(cons, equations::CompressibleEulerEquations1D)
  return 0.5 * (cons[2]^2)/cons[1]
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::CompressibleEulerEquations1D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


end # @muladd
