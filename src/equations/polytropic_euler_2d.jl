# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    PolytropicEulerEquations2D(gamma, kappa)

The compressible Euler equations
```math
\partial t
\begin{pmatrix}
\rho \\ \rho v_1 \\ \rho v_2
\end{pmatrix}
+
\partial x
\begin{pmatrix}
 \rho v_1 \\ \rho v_1^2 + \kappa\rho^\gamma \\ \rho v_1 v_2
\end{pmatrix}
+
\partial y
\begin{pmatrix}
\rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + \kappa\rho^\gamma
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0
\end{pmatrix}
```
for an ideal gas with ratio of specific heats `gamma`
in two space dimensions.
Here, ``\rho`` is the density and ``v_1`` and`v_2` the velocities and
```math
p = \kappa\rho^\gamma
```
the pressure, which we replaced using this relation.

"""
struct PolytropicEulerEquations2D{RealT<:Real, RealT<:Real} <: AbstractPolytropicEulerEquations{2, 3}
  gamma::RealT               # ratio of specific heats
  kappa::RealT               # fluid scaling factor
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

  function PolytropicEulerEquations2D(gamma, kappa)
    new{typeof(gamma), typeof(kappa)}(gamma, kappa)
  end
end


varnames(::typeof(cons2cons), ::PolytropicEulerEquations2D) = ("rho", "rho_v1", "rho_v2")
varnames(::typeof(cons2prim), ::PolytropicEulerEquations2D) = ("rho", "v1", "v2")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::PolytropicEulerEquations2D)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2

  return SVector(rho, rho_v1, rho_v2)
end


"""
    initial_condition_convergence_test(x, t, equations::PolytropicEulerEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::PolytropicEulerEquations2D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  rho = ini
  rho_v1 = ini
  rho_v2 = ini

  return SVector(rho, rho_v1, rho_v2)
end


"""
    initial_condition_density_wave(x, t, equations::PolytropicEulerEquations2D)

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
function initial_condition_density_wave(x, t, equations::PolytropicEulerEquations2D)
  v1 = 0.1
  v2 = 0.2
  rho = 1 + 0.98 * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 20
  return SVector(rho, rho_v1, rho_v2)
end


"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::PolytropicEulerEquations2D)

Determine the boundary numerical surface flux for a slip wall condition.
Imposes a zero normal velocity at the wall.
Density is taken from the internal solution state and pressure is computed as an
exact solution of a 1D Riemann problem. Further details about this boundary state
are available in the paper:
- J. J. W. van der Vegt and H. van der Ven (2002)
  Slip flow boundary conditions in discontinuous Galerkin discretizations of
  the Euler equations of gas dynamics
  [PDF](https://reports.nlr.nl/bitstream/handle/10921/692/TP-2002-300.pdf?sequence=1)

Details about the 1D pressure Riemann solution can be found in Section 6.3.3 of the book
- Eleuterio F. Toro (2009)
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

Should be used together with [`UnstructuredMesh2D`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::PolytropicEulerEquations2D)

  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_

  # rotate the internal solution state
  u_local = rotate_to_x(u_inner, normal, equations)
  p_local = equations.kappa*rho^equations.gamma

  # compute the primitive variables
  rho_local, v_normal, v_tangent = cons2prim(u_local, equations)

  # Get the solution of the pressure Riemann problem
  # See Section 6.3.3 of
  # Eleuterio F. Toro (2009)
  # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Pratical Introduction
  # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
  if v_normal <= 0.0
    sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
    p_star = p_local * (1.0 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2.0 * equations.gamma * equations.inv_gamma_minus_one)
  else # v_normal > 0.0
    A = 2.0 / ((equations.gamma + 1) * rho_local)
    B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
    p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4.0 * A * (p_local + B)))
  end

  # For the slip wall we directly set the flux as the normal velocity is zero
  return SVector(zero(eltype(u_inner)),
                 p_star * normal[1],
                 p_star * normal[2]) * norm_
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::PolytropicEulerEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::PolytropicEulerEquations2D)
  # get the appropriate normal vector from the orientation
  if orientation == 1
    normal = SVector(1, 0)
  else # orientation == 2
    normal = SVector(0, 1)
  end

  # compute and return the flux using `boundary_condition_slip_wall` routine above
  return boundary_condition_slip_wall(u_inner, normal, x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                 surface_flux_function, equations::PolytropicEulerEquations2D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::PolytropicEulerEquations2D)
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


# Calculate 1D flux for a single point
@inline function flux(u, orientation::Integer, equations::PolytropicEulerEquations2D)
  rho, rho_v1, rho_v2 = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = equations.kappa*rho^equations.gamma
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
  else
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
  end
  return SVector(f1, f2, f3)
end

# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::PolytropicEulerEquations2D)
  rho_e = last(u)
  rho, v1, v2 = cons2prim(u, equations)
  p = equations.kappa*rho^equations.gamma

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  rho_v_normal = rho * v_normal
  f1 = rho_v_normal
  f2 = rho_v_normal * v1 + p * normal_direction[1]
  f3 = rho_v_normal * v2 + p * normal_direction[2]
  return SVector(f1, f2, f3)
end


"""
    flux_hllc(u_ll, u_rr, orientation, equations::PolytropicEulerEquations2D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation::Integer, equations::PolytropicEulerEquations2D)
    # Calculate primitive variables and speed of sound
    rho_ll, rho_v1_ll, rho_v2_ll = u_ll
    rho_rr, rho_v1_rr, rho_v2_rr = u_rr
  
    v1_ll = rho_v1_ll / rho_ll
    v2_ll = rho_v2_ll / rho_ll
    p_ll = equations.kappa * rho_ll^equations.gamma
    e_ll  = p_ll * rho_ll / (equations.gamma - 1)
    rho_e_ll = rho_ll * e_ll
    c_ll = sqrt(equations.gamma*p_ll/rho_ll)
  
    v1_rr = rho_v1_rr / rho_rr
    v2_rr = rho_v2_rr / rho_rr
    p_rr = equations.kappa * rho_rr^equations.gamma
    e_rr  = p_rr * rho_rr / (equations.gamma - 1)
    rho_e_rr = rho_rr * e_rr
    c_rr = sqrt(equations.gamma*p_rr/rho_rr)
  
    # Obtain left and right fluxes
    f_ll = flux(u_ll, orientation, equations)
    f_rr = flux(u_rr, orientation, equations)
  
    # Compute Roe averages
    sqrt_rho_ll = sqrt(rho_ll)
    sqrt_rho_rr = sqrt(rho_rr)
    sum_sqrt_rho = sqrt_rho_ll + sqrt_rho_rr
    if orientation == 1 # x-direction
      vel_L = v1_ll
      vel_R = v1_rr
      ekin_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr)^2
    elseif orientation == 2 # y-direction
      vel_L = v2_ll
      vel_R = v2_rr
      ekin_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr)^2
    end
    vel_roe = (sqrt_rho_ll * vel_L + sqrt_rho_rr * vel_R) / sum_sqrt_rho
    ekin_roe = 0.5 * (vel_roe^2 + ekin_roe / sum_sqrt_rho^2)
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
        UStar4 = densStar*enerStar
        if orientation == 1 # x-direction
          UStar2 = densStar*SStar
          UStar3 = densStar*v2_ll
        elseif orientation == 2 # y-direction
          UStar2 = densStar*v1_ll
          UStar3 = densStar*SStar
        end
        f1 = f_ll[1]+Ssl*(UStar1 - rho_ll)
        f2 = f_ll[2]+Ssl*(UStar2 - rho_v1_ll)
        f3 = f_ll[3]+Ssl*(UStar3 - rho_v2_ll)
      else
        densStar = rho_rr*sMu_R / (Ssr-SStar)
        enerStar = e_rr + (SStar - vel_R) * (SStar + p_rr / (rho_rr * sMu_R))
        UStar1 = densStar
        UStar4 = densStar*enerStar
        if orientation == 1 # x-direction
          UStar2 = densStar*SStar
          UStar3 = densStar*v2_rr
        elseif orientation == 2 # y-direction
          UStar2 = densStar*v1_rr
          UStar3 = densStar*SStar
        end
        f1 = f_rr[1]+Ssr*(UStar1 - rho_rr)
        f2 = f_rr[2]+Ssr*(UStar2 - rho_v1_rr)
        f3 = f_rr[3]+Ssr*(UStar3 - rho_v2_rr)
      end
    end
    return SVector(f1, f2, f3)
  end



"""
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                 equations::PolytropicEulerEquations2D)

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
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::PolytropicEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
  p_ll = equations.kappa*rho_ll^equations.gamma
  p_rr = equations.kappa*rho_rr^equations.gamma

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
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
  end

  return SVector(f1, f2, f3)
end


@inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector, equations::PolytropicEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
  p_ll = equations.kappa*rho_ll^equations.gamma
  p_rr = equations.kappa*rho_rr^equations.gamma
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

  # Calculate fluxes depending on normal_direction
  f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]

  return SVector(f1, f2, f3)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::PolytropicEulerEquations2D)
  rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
  p_ll = equations.kappa*rho_ll^equations.gamma
  p_rr = equations.kappa*rho_rr^equations.gamma

  # Get the velocity value in the appropriate direction
  if orientation == 1
    v_ll = v1_ll
    v_rr = v1_rr
  else # orientation == 2
    v_ll = v2_ll
    v_rr = v2_rr
  end
  # Calculate sound speeds
  c_ll = sqrt(equations.gamma * p_ll / rho_ll)
  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

  λ_max = max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr)
end


@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::PolytropicEulerEquations2D)
  rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
  p_ll = equations.kappa*rho_ll^equations.gamma
  p_rr = equations.kappa*rho_rr^equations.gamma

  # Calculate normal velocities and sound speed
  # left
  v_ll = (  v1_ll * normal_direction[1]
          + v2_ll * normal_direction[2] )
  c_ll = sqrt(equations.gamma * p_ll / rho_ll)
  # right
  v_rr = (  v1_rr * normal_direction[1]
          + v2_rr * normal_direction[2] )
  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

  return max(abs(v_ll), abs(v_rr)) + max(c_ll, c_rr) * norm(normal_direction)
end


# Calculate minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(u_ll, u_rr, orientation::Integer,
                                     equations::PolytropicEulerEquations2D)
  rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
  p_ll = equations.kappa*rho_ll^equations.gamma
  p_rr = equations.kappa*rho_rr^equations.gamma

  if orientation == 1 # x-direction
    λ_min = v1_ll - sqrt(equations.gamma * p_ll / rho_ll)
    λ_max = v1_rr + sqrt(equations.gamma * p_rr / rho_rr)
  else # y-direction
    λ_min = v2_ll - sqrt(equations.gamma * p_ll / rho_ll)
    λ_max = v2_rr + sqrt(equations.gamma * p_rr / rho_rr)
  end

  return λ_min, λ_max
end

@inline function min_max_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                     equations::PolytropicEulerEquations2D)
  rho_ll, v1_ll, v2_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr = cons2prim(u_rr, equations)
  p_ll = equations.kappa*rho_ll^equations.gamma
  p_rr = equations.kappa*rho_rr^equations.gamma

  v_normal_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_normal_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  norm_ = norm(normal_direction)
  # The v_normals are already scaled by the norm
  λ_min = v_normal_ll - sqrt(equations.gamma * p_ll / rho_ll) * norm_
  λ_max = v_normal_rr + sqrt(equations.gamma * p_rr / rho_rr) * norm_

  return λ_min, λ_max
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this rotation of the state vector
@inline function rotate_to_x(u, normal_vector, equations::PolytropicEulerEquations2D)
  # cos and sin of the angle between the x-axis and the normalized normal_vector are
  # the normalized vector's x and y coordinates respectively (see unit circle).
  c = normal_vector[1]
  s = normal_vector[2]

  # Apply the 2D rotation matrix with normal and tangent directions of the form
  # [ 1    0    0   
  #   0   n_1  t_1  
  #   0   n_2  t_2 ]
  # where t_1 = -n_2 and t_2 = n_1

  return SVector(u[1],
                 c * u[2] + s * u[3],
                 -s * u[2] + c * u[3])
end


# Called inside `FluxRotated` in `numerical_fluxes.jl` so the direction
# has been normalized prior to this back-rotation of the state vector
@inline function rotate_from_x(u, normal_vector, equations::PolytropicEulerEquations2D)
  # cos and sin of the angle between the x-axis and the normalized normal_vector are
  # the normalized vector's x and y coordinates respectively (see unit circle).
  c = normal_vector[1]
  s = normal_vector[2]

  # Apply the 2D back-rotation matrix with normal and tangent directions of the form
  # [ 1    0    0   
  #   0   n_1  t_1  
  #   0   n_2  t_2 ]
  # where t_1 = -n_2 and t_2 = n_1

  return SVector(u[1],
                 c * u[2] - s * u[3],
                 s * u[2] + c * u[3])
end


@inline function max_abs_speeds(u, equations::PolytropicEulerEquations2D)
  rho, v1, v2 = cons2prim(u, equations)
  c = sqrt(equations.gamma * equations.kappa*rho^(equations.gamma-1))

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::PolytropicEulerEquations2D)
  rho, rho_v1, rho_v2 = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho

  return SVector(rho, v1, v2)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::PolytropicEulerEquations2D)
  rho, rho_v1, rho_v2 = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  p = equations.kappa * rho^equations.gamma
  s = rho/2*v_square + rho*equations.kappa*rho^(equations.gamma-1)/(equations.gamma-1)
  rho_p = rho / p

  w1 = (equations.gamma - s) * (equations.gamma - 1) - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2

  return SVector(w1, w2, w3)
end

# TODO: Do we need this? (SC)
# @inline function entropy2cons(w, equations::PolytropicEulerEquations2D)
#   # See Hughes, Franca, Mallet (1986) A new finite element formulation for CFD
#   # [DOI: 10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)
#   @unpack gamma, kappa = equations

#   # convert to entropy `-rho * s` used by Hughes, France, Mallet (1986)
#   # instead of `-rho * s / (gamma - 1)`
#   V1, V2, V3, V5 = w .* (gamma-1)

#   # s = specific entropy, eq. (53)
#   s = gamma - V1 + (V2^2 + V3^2)/(2*V5)

#   # eq. (52)
#   rho_iota = ((gamma-1) / (-V5)^gamma)^(equations.inv_gamma_minus_one)*exp(-s * equations.inv_gamma_minus_one)

#   # eq. (51)
#   rho      = -rho_iota * V5
#   rho_v1   =  rho_iota * V2
#   rho_v2   =  rho_iota * V3
#   rho_e    =  rho_iota * (1-(V2^2 + V3^2)/(2*V5))
#   return SVector(rho, rho_v1, rho_v2, rho_e)
# end




# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::PolytropicEulerEquations2D)
  rho, v1, v2 = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  return SVector(rho, rho_v1, rho_v2)
end


@inline function density(u, equations::PolytropicEulerEquations2D)
 rho = u[1]
 return rho
end


@inline function pressure(u, equations::PolytropicEulerEquations2D)
 rho, rho_v1, rho_v2 = u
 p = equations.kappa*rho^equations.gamma
 return p
end


@inline function density_pressure(u, equations::PolytropicEulerEquations2D)
 rho, rho_v1, rho_v2 = u
 rho_times_p = equations.kappa*rho^(equations.gamma+1)
 return rho_times_p
end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::PolytropicEulerEquations2D)
  # Pressure
  p = equations.kappa*cons[1]^equations.gamma

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::PolytropicEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) * cons[1] * (equations.gamma - 1)

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::PolytropicEulerEquations2D) = entropy_math(cons, equations)

end # @muladd
