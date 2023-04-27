# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    CompressibleEulerEquations2D(gamma)

The compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho v_1 \\ \rho v_2 \\ \rho e
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
 \rho v_1 \\ \rho v_1^2 + p \\ \rho v_1 v_2 \\ (\rho e +p) v_1
\end{pmatrix}
+
\frac{\partial}{\partial y}
\begin{pmatrix}
\rho v_2 \\ \rho v_1 v_2 \\ \rho v_2^2 + p \\ (\rho e +p) v_2
\end{pmatrix}
=
\begin{pmatrix}
0 \\ 0 \\ 0 \\ 0
\end{pmatrix}
```
for an ideal gas with ratio of specific heats `gamma`
in two space dimensions.
Here, ``\rho`` is the density, ``v_1``,`v_2` the velocities, ``e`` the specific total energy **rather than** specific internal energy, and
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
```
the pressure.
"""
struct CompressibleEulerEquations2D{RealT<:Real} <: AbstractCompressibleEulerEquations{2, 4}
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

  function CompressibleEulerEquations2D(gamma)
    γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))
    new{typeof(γ)}(γ, inv_gamma_minus_one)
  end
end


varnames(::typeof(cons2cons), ::CompressibleEulerEquations2D) = ("rho", "rho_v1", "rho_v2", "rho_e")
varnames(::typeof(cons2prim), ::CompressibleEulerEquations2D) = ("rho", "v1", "v2", "p")


# Set initial conditions at physical location `x` for time `t`
"""
    initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)

A constant initial condition to test free-stream preservation.
"""
function initial_condition_constant(x, t, equations::CompressibleEulerEquations2D)
  rho = 1.0
  rho_v1 = 0.1
  rho_v2 = -0.2
  rho_e = 10.0
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)

A smooth initial condition used for convergence tests in combination with
[`source_terms_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
function initial_condition_convergence_test(x, t, equations::CompressibleEulerEquations2D)
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  ini = c + A * sin(ω * (x[1] + x[2] - t))

  rho = ini
  rho_v1 = ini
  rho_v2 = ini
  rho_e = ini^2

  return SVector(rho, rho_v1, rho_v2, rho_e)
end

"""
    source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations2D)

Source terms used for convergence tests in combination with
[`initial_condition_convergence_test`](@ref)
(and [`BoundaryConditionDirichlet(initial_condition_convergence_test)`](@ref) in non-periodic domains).
"""
@inline function source_terms_convergence_test(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition`
  c = 2
  A = 0.1
  L = 2
  f = 1/L
  ω = 2 * pi * f
  γ = equations.gamma

  x1, x2 = x
  si, co = sincos(ω * (x1 + x2 - t))
  rho = c + A * si
  rho_x = ω * A * co
  # Note that d/dt rho = -d/dx rho = -d/dy rho.

  tmp = (2 * rho - 1) * (γ - 1)

  du1 = rho_x
  du2 = rho_x * (1 + tmp)
  du3 = du2
  du4 = 2 * rho_x * (rho + tmp)

  return SVector(du1, du2, du3, du4)
end


"""
    initial_condition_density_wave(x, t, equations::CompressibleEulerEquations2D)

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
function initial_condition_density_wave(x, t, equations::CompressibleEulerEquations2D)
  v1 = 0.1
  v2 = 0.2
  rho = 1 + 0.98 * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  p = 20
  rho_e = p / (equations.gamma - 1) + 1/2 * rho * (v1^2 + v2^2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


"""
    initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A weak blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_weak_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # From Hennemann & Gassner JCP paper 2020 (Sec. 6.3)
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p   = r > 0.5 ? 1.0 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equations)
end


"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::CompressibleEulerEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`source_terms_eoc_test_coupled_euler_gravity`](@ref)
or [`source_terms_eoc_test_euler`](@ref).
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::CompressibleEulerEquations2D)
  # OBS! this assumes that γ = 2 other manufactured source terms are incorrect
  if equations.gamma != 2.0
    error("adiabatic constant must be 2 for the coupling convergence test")
  end
  c = 2.0
  A = 0.1
  ini = c + A * sin(pi * (x[1] + x[2] - t))
  G = 1.0 # gravitational constant

  rho = ini
  v1 = 1.0
  v2 = 1.0
  p = ini^2 * G / pi # * 2 / ndims, but ndims==2 here

  return prim2cons(SVector(rho, v1, v2, p), equations)
end

"""
    source_terms_eoc_test_coupled_euler_gravity(u, x, t, equations::CompressibleEulerEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`initial_condition_eoc_test_coupled_euler_gravity`](@ref).
"""
@inline function source_terms_eoc_test_coupled_euler_gravity(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0 # gravitational constant, must match coupling solver
  C_grav = -2 * G / pi # 2 == 4 / ndims

  x1, x2 = x
  si, co = sincos(pi * (x1 + x2 - t))
  rhox = A * pi * co
  rho  = c + A *  si

  du1 = rhox
  du2 = rhox
  du3 = rhox
  du4 = (1.0 - C_grav*rho)*rhox

  return SVector(du1, du2, du3, du4)
end

"""
    source_terms_eoc_test_euler(u, x, t, equations::CompressibleEulerEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`initial_condition_eoc_test_coupled_euler_gravity`](@ref).
"""
@inline function source_terms_eoc_test_euler(u, x, t, equations::CompressibleEulerEquations2D)
  # Same settings as in `initial_condition_eoc_test_coupled_euler_gravity`
  c = 2.0
  A = 0.1
  G = 1.0
  C_grav = -2 * G / pi # 2 == 4 / ndims

  x1, x2 = x
  si, co = sincos(pi * (x1 + x2 - t))
  rhox = A * pi * co
  rho  = c + A *  si

  du1 = rhox
  du2 = rhox * (1 -     C_grav * rho)
  du3 = rhox * (1 -     C_grav * rho)
  du4 = rhox * (1 - 3 * C_grav * rho)

  return SVector(du1, du2, du3, du4)
end


"""
    boundary_condition_slip_wall(u_inner, normal_direction, x, t, surface_flux_function,
                                 equations::CompressibleEulerEquations2D)

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
  Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  3rd edition
  [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)

Should be used together with [`UnstructuredMesh2D`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquations2D)

  norm_ = norm(normal_direction)
  # Normalize the vector without using `normalize` since we need to multiply by the `norm_` later
  normal = normal_direction / norm_

  # rotate the internal solution state
  u_local = rotate_to_x(u_inner, normal, equations)

  # compute the primitive variables
  rho_local, v_normal, v_tangent, p_local = cons2prim(u_local, equations)

  # Get the solution of the pressure Riemann problem
  # See Section 6.3.3 of
  # Eleuterio F. Toro (2009)
  # Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction
  # [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761)
  if v_normal <= 0.0
    sound_speed = sqrt(equations.gamma * p_local / rho_local) # local sound speed
    p_star = p_local * (1 + 0.5 * (equations.gamma - 1) * v_normal / sound_speed)^(2 * equations.gamma * equations.inv_gamma_minus_one)
  else # v_normal > 0.0
    A = 2 / ((equations.gamma + 1) * rho_local)
    B = p_local * (equations.gamma - 1) / (equations.gamma + 1)
    p_star = p_local + 0.5 * v_normal / A * (v_normal + sqrt(v_normal^2 + 4 * A * (p_local + B)))
  end

  # For the slip wall we directly set the flux as the normal velocity is zero
  return SVector(zero(eltype(u_inner)),
                 p_star * normal[1],
                 p_star * normal[2],
                 zero(eltype(u_inner))) * norm_
end

"""
    boundary_condition_slip_wall(u_inner, orientation, direction, x, t,
                                 surface_flux_function, equations::CompressibleEulerEquations2D)

Should be used together with [`TreeMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, orientation,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquations2D)
  # get the appropriate normal vector from the orientation
  if orientation == 1
    normal_direction = SVector(1, 0)
  else # orientation == 2
    normal_direction = SVector(0, 1)
  end

  # compute and return the flux using `boundary_condition_slip_wall` routine above
  return boundary_condition_slip_wall(u_inner, normal_direction, direction,
                                      x, t, surface_flux_function, equations)
end

"""
    boundary_condition_slip_wall(u_inner, normal_direction, direction, x, t,
                                 surface_flux_function, equations::CompressibleEulerEquations2D)

Should be used together with [`StructuredMesh`](@ref).
"""
@inline function boundary_condition_slip_wall(u_inner, normal_direction::AbstractVector,
                                              direction, x, t,
                                              surface_flux_function,
                                              equations::CompressibleEulerEquations2D)
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


# Calculate 2D flux for a single point
@inline function flux(u, orientation::Integer, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  if orientation == 1
    f1 = rho_v1
    f2 = rho_v1 * v1 + p
    f3 = rho_v1 * v2
    f4 = (rho_e + p) * v1
  else
    f1 = rho_v2
    f2 = rho_v2 * v1
    f3 = rho_v2 * v2 + p
    f4 = (rho_e + p) * v2
  end
  return SVector(f1, f2, f3, f4)
end

# Calculate 2D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::CompressibleEulerEquations2D)
  rho_e = last(u)
  rho, v1, v2, p = cons2prim(u, equations)

  v_normal = v1 * normal_direction[1] + v2 * normal_direction[2]
  rho_v_normal = rho * v_normal
  f1 = rho_v_normal
  f2 = rho_v_normal * v1 + p * normal_direction[1]
  f3 = rho_v_normal * v2 + p * normal_direction[2]
  f4 = (rho_e + p) * v_normal
  return SVector(f1, f2, f3, f4)
end


"""
    flux_shima_etal(u_ll, u_rr, orientation_or_normal_direction,
                    equations::CompressibleEulerEquations2D)

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
@inline function flux_shima_etal(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  kin_avg = 1/2 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    pv1_avg = 1/2 * (p_ll*v1_rr + p_rr*v1_ll)
    f1 = rho_avg * v1_avg
    f2 = f1 * v1_avg + p_avg
    f3 = f1 * v2_avg
    f4 = p_avg*v1_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv1_avg
  else
    pv2_avg = 1/2 * (p_ll*v2_rr + p_rr*v2_ll)
    f1 = rho_avg * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = p_avg*v2_avg * equations.inv_gamma_minus_one + f1 * kin_avg + pv2_avg
  end

  return SVector(f1, f2, f3, f4)
end

@inline function flux_shima_etal(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
  v_dot_n_ll = v1_ll * normal_direction[1] + v2_ll * normal_direction[2]
  v_dot_n_rr = v1_rr * normal_direction[1] + v2_rr * normal_direction[2]

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  v_dot_n_avg = 1/2 * (v_dot_n_ll + v_dot_n_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on normal_direction
  f1 = rho_avg * v_dot_n_avg
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]
  f4 = ( f1 * velocity_square_avg + p_avg * v_dot_n_avg * equations.inv_gamma_minus_one
        + 0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll) )

  return SVector(f1, f2, f3, f4)
end


"""
    flux_kennedy_gruber(u_ll, u_rr, orientation_or_normal_direction,
                        equations::CompressibleEulerEquations2D)

Kinetic energy preserving two-point flux by
- Kennedy and Gruber (2008)
  Reduced aliasing formulations of the convective terms within the
  Navier-Stokes equations for a compressible fluid
  [DOI: 10.1016/j.jcp.2007.09.020](https://doi.org/10.1016/j.jcp.2007.09.020)
"""
@inline function flux_kennedy_gruber(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_e_ll = last(u_ll)
  rho_e_rr = last(u_rr)
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 1/2 * (rho_ll + rho_rr)
  v1_avg  = 1/2 * ( v1_ll +  v1_rr)
  v2_avg  = 1/2 * ( v2_ll +  v2_rr)
  p_avg   = 1/2 * (  p_ll +   p_rr)
  e_avg   = 1/2 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_avg * v1_avg
    f2 = rho_avg * v1_avg * v1_avg + p_avg
    f3 = rho_avg * v1_avg * v2_avg
    f4 = (rho_avg * e_avg + p_avg) * v1_avg
  else
    f1 = rho_avg * v2_avg
    f2 = rho_avg * v2_avg * v1_avg
    f3 = rho_avg * v2_avg * v2_avg + p_avg
    f4 = (rho_avg * e_avg + p_avg) * v2_avg
  end

  return SVector(f1, f2, f3, f4)
end

@inline function flux_kennedy_gruber(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_e_ll = last(u_ll)
  rho_e_rr = last(u_rr)
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

  # Average each factor of products in flux
  rho_avg = 0.5 * (rho_ll + rho_rr)
  v1_avg  = 0.5 * (v1_ll + v1_rr)
  v2_avg  = 0.5 * (v2_ll + v2_rr)
  v_dot_n_avg = v1_avg * normal_direction[1] + v2_avg * normal_direction[2]
  p_avg = 0.5 * (p_ll + p_rr)
  e_avg = 0.5 * (rho_e_ll / rho_ll + rho_e_rr / rho_rr)

  # Calculate fluxes depending on normal_direction
  f1 = rho_avg * v_dot_n_avg
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]
  f4 = f1 * e_avg + p_avg * v_dot_n_avg

  return SVector(f1, f2, f3, f4)
end


"""
    flux_chandrashekar(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Entropy conserving two-point flux by
- Chandrashekar (2013)
  Kinetic Energy Preserving and Entropy Stable Finite Volume Schemes
  for Compressible Euler and Navier-Stokes Equations
  [DOI: 10.4208/cicp.170712.010313a](https://doi.org/10.4208/cicp.170712.010313a)
"""
@inline function flux_chandrashekar(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
  beta_ll = 0.5 * rho_ll / p_ll
  beta_rr = 0.5 * rho_rr / p_rr
  specific_kin_ll = 0.5 * (v1_ll^2 + v2_ll^2)
  specific_kin_rr = 0.5 * (v1_rr^2 + v2_rr^2)

  # Compute the necessary mean values
  rho_avg = 0.5 * (rho_ll + rho_rr)
  rho_mean  = ln_mean(rho_ll, rho_rr)
  beta_mean = ln_mean(beta_ll, beta_rr)
  beta_avg = 0.5 * (beta_ll + beta_rr)
  v1_avg = 0.5 * (v1_ll + v1_rr)
  v2_avg = 0.5 * (v2_ll + v2_rr)
  p_mean = 0.5 * rho_avg / beta_avg
  velocity_square_avg = specific_kin_ll + specific_kin_rr

  # Calculate fluxes depending on orientation
  if orientation == 1
    f1 = rho_mean * v1_avg
    f2 = f1 * v1_avg + p_mean
    f3 = f1 * v2_avg
    f4 = f1 * 0.5*(1/(equations.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg + f3*v2_avg
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_mean
    f4 = f1 * 0.5*(1/(equations.gamma-1)/beta_mean - velocity_square_avg)+f2*v1_avg + f3*v2_avg
  end

  return SVector(f1, f2, f3, f4)
end


"""
    flux_ranocha(u_ll, u_rr, orientation_or_normal_direction,
                 equations::CompressibleEulerEquations2D)

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
@inline function flux_ranocha(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

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
    f4 = f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one ) + 0.5 * (p_ll*v1_rr + p_rr*v1_ll)
  else
    f1 = rho_mean * v2_avg
    f2 = f1 * v1_avg
    f3 = f1 * v2_avg + p_avg
    f4 = f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one ) + 0.5 * (p_ll*v2_rr + p_rr*v2_ll)
  end

  return SVector(f1, f2, f3, f4)
end

@inline function flux_ranocha(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleEulerEquations2D)
  # Unpack left and right state
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)
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
  velocity_square_avg = 0.5 * (v1_ll*v1_rr + v2_ll*v2_rr)

  # Calculate fluxes depending on normal_direction
  f1 = rho_mean * 0.5 * (v_dot_n_ll + v_dot_n_rr)
  f2 = f1 * v1_avg + p_avg * normal_direction[1]
  f3 = f1 * v2_avg + p_avg * normal_direction[2]
  f4 = ( f1 * ( velocity_square_avg + inv_rho_p_mean * equations.inv_gamma_minus_one )
        + 0.5 * (p_ll * v_dot_n_rr + p_rr * v_dot_n_ll) )

  return SVector(f1, f2, f3, f4)
end


"""
    splitting_steger_warming(u, orientation::Integer,
                             equations::CompressibleEulerEquations2D)
    splitting_steger_warming(u, which::Union{Val{:minus}, Val{:plus}}
                             orientation::Integer,
                             equations::CompressibleEulerEquations2D)

Splitting of the compressible Euler flux of Steger and Warming.

Returns a tuple of the fluxes "minus" (associated with waves going into the
negative axis direction) and "plus" (associated with waves going into the
positive axis direction). If only one of the fluxes is required, use the
function signature with argument `which` set to `Val{:minus}()` or `Val{:plus}`.

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.

## References

- Joseph L. Steger and R. F. Warming (1979)
  Flux Vector Splitting of the Inviscid Gasdynamic Equations
  With Application to Finite Difference Methods
  [NASA Technical Memorandum](https://ntrs.nasa.gov/api/citations/19790020779/downloads/19790020779.pdf)
"""
@inline function splitting_steger_warming(u, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  fm = splitting_steger_warming(u, Val{:minus}(), orientation, equations)
  fp = splitting_steger_warming(u, Val{:plus}(),  orientation, equations)
  return fm, fp
end

@inline function splitting_steger_warming(u, ::Val{:plus}, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  a = sqrt(equations.gamma * p / rho)

  if orientation == 1
    lambda1 = v1
    lambda2 = v1 + a
    lambda3 = v1 - a

    lambda1_p = positive_part(lambda1) # Same as (lambda_i + abs(lambda_i)) / 2, but faster :)
    lambda2_p = positive_part(lambda2)
    lambda3_p = positive_part(lambda3)

    alpha_p = 2 * (equations.gamma - 1) * lambda1_p + lambda2_p + lambda3_p

    rho_2gamma = 0.5 * rho / equations.gamma
    f1p = rho_2gamma * alpha_p
    f2p = rho_2gamma * (alpha_p * v1 + a * (lambda2_p - lambda3_p))
    f3p = rho_2gamma * alpha_p * v2
    f4p = rho_2gamma * (alpha_p * 0.5 * (v1^2 + v2^2) + a * v1 * (lambda2_p - lambda3_p)
                        + a^2 * (lambda2_p + lambda3_p) * equations.inv_gamma_minus_one)
  else # orientation == 2
    lambda1 = v2
    lambda2 = v2 + a
    lambda3 = v2 - a

    lambda1_p = positive_part(lambda1) # Same as (lambda_i + abs(lambda_i)) / 2, but faster :)
    lambda2_p = positive_part(lambda2)
    lambda3_p = positive_part(lambda3)

    alpha_p = 2 * (equations.gamma - 1) * lambda1_p + lambda2_p + lambda3_p

    rho_2gamma = 0.5 * rho / equations.gamma
    f1p = rho_2gamma * alpha_p
    f2p = rho_2gamma * alpha_p * v1
    f3p = rho_2gamma * (alpha_p * v2 + a * (lambda2_p - lambda3_p))
    f4p = rho_2gamma * (alpha_p * 0.5 * (v1^2 + v2^2) + a * v2 * (lambda2_p - lambda3_p)
                        + a^2 * (lambda2_p + lambda3_p) * equations.inv_gamma_minus_one)
  end
  return SVector(f1p, f2p, f3p, f4p)
end

@inline function splitting_steger_warming(u, ::Val{:minus}, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))
  a = sqrt(equations.gamma * p / rho)

  if orientation == 1
    lambda1 = v1
    lambda2 = v1 + a
    lambda3 = v1 - a

    lambda1_m = negative_part(lambda1) # Same as (lambda_i - abs(lambda_i)) / 2, but faster :)
    lambda2_m = negative_part(lambda2)
    lambda3_m = negative_part(lambda3)

    alpha_m = 2 * (equations.gamma - 1) * lambda1_m + lambda2_m + lambda3_m

    rho_2gamma = 0.5 * rho / equations.gamma
    f1m = rho_2gamma * alpha_m
    f2m = rho_2gamma * (alpha_m * v1 + a * (lambda2_m - lambda3_m))
    f3m = rho_2gamma * alpha_m * v2
    f4m = rho_2gamma * (alpha_m * 0.5 * (v1^2 + v2^2) + a * v1 * (lambda2_m - lambda3_m)
                        + a^2 * (lambda2_m + lambda3_m) * equations.inv_gamma_minus_one)
  else # orientation == 2
    lambda1 = v2
    lambda2 = v2 + a
    lambda3 = v2 - a

    lambda1_m = negative_part(lambda1) # Same as (lambda_i - abs(lambda_i)) / 2, but faster :)
    lambda2_m = negative_part(lambda2)
    lambda3_m = negative_part(lambda3)

    alpha_m = 2 * (equations.gamma - 1) * lambda1_m + lambda2_m + lambda3_m

    rho_2gamma = 0.5 * rho / equations.gamma
    f1m = rho_2gamma * alpha_m
    f2m = rho_2gamma * alpha_m * v1
    f3m = rho_2gamma * (alpha_m * v2 + a * (lambda2_m-lambda3_m))
    f4m = rho_2gamma * (alpha_m * 0.5 * (v1^2 + v2^2) + a * v2 * (lambda2_m - lambda3_m)
                        + a^2 * (lambda2_m + lambda3_m) * equations.inv_gamma_minus_one)
  end
  return SVector(f1m, f2m, f3m, f4m)
end


"""
    splitting_vanleer_haenel(u, orientation::Integer,
                             equations::CompressibleEulerEquations2D)
    splitting_vanleer_haenel(u, which::Union{Val{:minus}, Val{:plus}}
                             orientation::Integer,
                             equations::CompressibleEulerEquations2D)

Splitting of the compressible Euler flux from van Leer. This splitting further
contains a reformulation due to Hänel et al. where the energy flux uses the
enthalpy. The pressure splitting is independent from the splitting of the
convective terms. As such there are many pressure splittings suggested across
the literature. We implement the 'p4' variant suggested by Liou and Steffen as
it proved the most robust in practice.

Returns a tuple of the fluxes "minus" (associated with waves going into the
negative axis direction) and "plus" (associated with waves going into the
positive axis direction). If only one of the fluxes is required, use the
function signature with argument `which` set to `Val{:minus}()` or `Val{:plus}`.

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.

## References

- Bram van Leer (1982)
  Flux-Vector Splitting for the Euler Equation
  [DOI: 10.1007/978-3-642-60543-7_5](https://doi.org/10.1007/978-3-642-60543-7_5)
- D. Hänel, R. Schwane and G. Seider (1987)
  On the accuracy of upwind schemes for the solution of the Navier-Stokes equations
  [DOI: 10.2514/6.1987-1105](https://doi.org/10.2514/6.1987-1105)
- Meng-Sing Liou and Chris J. Steffen, Jr. (1991)
  High-Order Polynomial Expansions (HOPE) for Flux-Vector Splitting
  [NASA Technical Memorandum](https://ntrs.nasa.gov/citations/19910016425)
"""
@inline function splitting_vanleer_haenel(u, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  fm = splitting_vanleer_haenel(u, Val{:minus}(), orientation, equations)
  fp = splitting_vanleer_haenel(u, Val{:plus}(),  orientation, equations)
  return fm, fp
end

@inline function splitting_vanleer_haenel(u, ::Val{:plus}, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  a = sqrt(equations.gamma * p / rho)
  H = (rho_e + p) / rho

  if orientation == 1
    M = v1 / a
    p_plus = 0.5 * (1 + equations.gamma * M) * p

    f1p = 0.25 * rho * a * (M + 1)^2
    f2p = f1p * v1 + p_plus
    f3p = f1p * v2
    f4p = f1p * H
  else # orientation == 2
    M = v2 / a
    p_plus = 0.5 * (1 + equations.gamma * M) * p

    f1p = 0.25 * rho * a * (M + 1)^2
    f2p = f1p * v1
    f3p = f1p * v2 + p_plus
    f4p = f1p * H
  end
  return SVector(f1p, f2p, f3p, f4p)
end

@inline function splitting_vanleer_haenel(u, ::Val{:minus}, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  a = sqrt(equations.gamma * p / rho)
  H = (rho_e + p) / rho

  if orientation == 1
    M = v1 / a
    p_minus = 0.5 * (1 - equations.gamma * M) * p

    f1m= -0.25 * rho * a * (M - 1)^2
    f2m = f1m * v1 + p_minus
    f3m = f1m * v2
    f4m = f1m * H
  else # orientation == 2
    M = v2 / a
    p_minus = 0.5 * (1 - equations.gamma * M) * p

    f1m= -0.25 * rho * a * (M - 1)^2
    f2m = f1m * v1
    f3m = f1m * v2 + p_minus
    f4m = f1m * H
  end
  return SVector(f1m, f2m, f3m, f4m)
end


"""
    splitting_lax_friedrichs(u, orientation::Integer,
                             equations::CompressibleEulerEquations2D)
    splitting_lax_friedrichs(u, which::Union{Val{:minus}, Val{:plus}}
                             orientation::Integer,
                             equations::CompressibleEulerEquations2D)

Naive local Lax-Friedrichs style flux splitting of the form `f⁺ = 0.5 (f + λ u)`
and `f⁻ = 0.5 (f - λ u)` similar to a flux splitting one would apply, e.g.,
to Burgers' equation.

Returns a tuple of the fluxes "minus" (associated with waves going into the
negative axis direction) and "plus" (associated with waves going into the
positive axis direction). If only one of the fluxes is required, use the
function signature with argument `which` set to `Val{:minus}()` or `Val{:plus}`.

!!! warning "Experimental implementation (upwind SBP)"
    This is an experimental feature and may change in future releases.
"""
@inline function splitting_lax_friedrichs(u, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  fm = splitting_lax_friedrichs(u, Val{:minus}(), orientation, equations)
  fp = splitting_lax_friedrichs(u, Val{:plus}(),  orientation, equations)
  return fm, fp
end

@inline function splitting_lax_friedrichs(u, ::Val{:plus}, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  a = sqrt(equations.gamma * p / rho)
  H = (rho_e + p) / rho
  lambda = 0.5 * (sqrt(v1^2 + v2^2) + a)

  if orientation == 1
    #lambda = 0.5 * (abs(v1) + a)
    f1p = 0.5 * rho * v1 + lambda * u[1]
    f2p = 0.5 * rho * v1 * v1 + 0.5 * p + lambda * u[2]
    f3p = 0.5 * rho * v1 * v2 + lambda * u[3]
    f4p = 0.5 * rho * v1 * H + lambda * u[4]
  else # orientation == 2
    #lambda = 0.5 * (abs(v2) + a)
    f1p = 0.5 * rho * v2 + lambda * u[1]
    f2p = 0.5 * rho * v2 * v1 + lambda * u[2]
    f3p = 0.5 * rho * v2 * v2 + 0.5 * p + lambda * u[3]
    f4p = 0.5 * rho * v2 * H + lambda * u[4]
  end
  return SVector(f1p, f2p, f3p, f4p)
end

@inline function splitting_lax_friedrichs(u, ::Val{:minus}, orientation::Integer,
                                          equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  a = sqrt(equations.gamma * p / rho)
  H = (rho_e + p) / rho
  lambda = 0.5 * (sqrt(v1^2 + v2^2) + a)

  if orientation == 1
    #lambda = 0.5 * (abs(v1) + a)
    f1m = 0.5 * rho * v1 - lambda * u[1]
    f2m = 0.5 * rho * v1 * v1 + 0.5 * p - lambda * u[2]
    f3m = 0.5 * rho * v1 * v2 - lambda * u[3]
    f4m = 0.5 * rho * v1 * H - lambda * u[4]
  else # orientation == 2
    #lambda = 0.5 * (abs(v2) + a)
    f1m = 0.5 * rho * v2 - lambda * u[1]
    f2m = 0.5 * rho * v2 * v1 - lambda * u[2]
    f3m = 0.5 * rho * v2 * v2 + 0.5 * p - lambda * u[3]
    f4m = 0.5 * rho * v2 * H - lambda * u[4]
  end
  return SVector(f1m, f2m, f3m, f4m)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation as the
# maximum velocity magnitude plus the maximum speed of sound
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

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


@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::CompressibleEulerEquations2D)
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

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
                                     equations::CompressibleEulerEquations2D)
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

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
                                     equations::CompressibleEulerEquations2D)
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

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
@inline function rotate_to_x(u, normal_vector, equations::CompressibleEulerEquations2D)
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
@inline function rotate_from_x(u, normal_vector, equations::CompressibleEulerEquations2D)
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


"""
    flux_hllc(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Computes the HLLC flux (HLL with Contact) for compressible Euler equations developed by E.F. Toro
[Lecture slides](http://www.prague-sum.com/download/2012/Toro_2-HLLC-RiemannSolver.pdf)
Signal speeds: [DOI: 10.1137/S1064827593260140](https://doi.org/10.1137/S1064827593260140)
"""
function flux_hllc(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Calculate primitive variables and speed of sound
  rho_ll, rho_v1_ll, rho_v2_ll, rho_e_ll = u_ll
  rho_rr, rho_v1_rr, rho_v2_rr, rho_e_rr = u_rr

  v1_ll = rho_v1_ll / rho_ll
  v2_ll = rho_v2_ll / rho_ll
  e_ll  = rho_e_ll / rho_ll
  p_ll = (equations.gamma - 1) * (rho_e_ll - 1/2 * rho_ll * (v1_ll^2 + v2_ll^2))
  c_ll = sqrt(equations.gamma*p_ll/rho_ll)

  v1_rr = rho_v1_rr / rho_rr
  v2_rr = rho_v2_rr / rho_rr
  e_rr  = rho_e_rr / rho_rr
  p_rr = (equations.gamma - 1) * (rho_e_rr - 1/2 * rho_rr * (v1_rr^2 + v2_rr^2))
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
    f4 = f_ll[4]
  elseif Ssr <= 0.0
    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
    f4 = f_rr[4]
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
      f4 = f_ll[4]+Ssl*(UStar4 - rho_e_ll)
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
      f4 = f_rr[4]+Ssr*(UStar4 - rho_e_rr)
    end
  end
  return SVector(f1, f2, f3, f4)
end


"""
    flux_hlle(u_ll, u_rr, orientation, equations::CompressibleEulerEquations2D)

Computes the HLLE (Harten-Lax-van Leer-Einfeldt) flux for the compressible Euler equations.
Special estimates of the signal velocites and linearization of the Riemann problem developed
by Einfeldt to ensure that the internal energy and density remain positive during the computation
of the numerical flux.

- Bernd Einfeldt (1988)
  On Godunov-type methods for gas dynamics.
  [DOI: 10.1137/0725021](https://doi.org/10.1137/0725021)
- Bernd Einfeldt, Claus-Dieter Munz, Philip L. Roe and Björn Sjögreen (1991)
  On Godunov-type methods near low densities.
  [DOI: 10.1016/0021-9991(91)90211-3](https://doi.org/10.1016/0021-9991(91)90211-3)
"""
function flux_hlle(u_ll, u_rr, orientation::Integer, equations::CompressibleEulerEquations2D)
  # Calculate primitive variables, enthalpy and speed of sound
  rho_ll, v1_ll, v2_ll, p_ll = cons2prim(u_ll, equations)
  rho_rr, v1_rr, v2_rr, p_rr = cons2prim(u_rr, equations)

  # `u_ll[4]` is total energy `rho_e_ll` on the left
  H_ll = (u_ll[4] + p_ll) / rho_ll
  c_ll = sqrt(equations.gamma * p_ll / rho_ll)

  # `u_rr[4]` is total energy `rho_e_rr` on the right
  H_rr = (u_rr[4] + p_rr) / rho_rr
  c_rr = sqrt(equations.gamma * p_rr / rho_rr)

  # Compute Roe averages
  sqrt_rho_ll = sqrt(rho_ll)
  sqrt_rho_rr = sqrt(rho_rr)
  inv_sum_sqrt_rho = inv(sqrt_rho_ll + sqrt_rho_rr)

  v1_roe = (sqrt_rho_ll * v1_ll + sqrt_rho_rr * v1_rr) * inv_sum_sqrt_rho
  v2_roe = (sqrt_rho_ll * v2_ll + sqrt_rho_rr * v2_rr) * inv_sum_sqrt_rho
  v_roe_mag = v1_roe^2 + v2_roe^2

  H_roe = (sqrt_rho_ll * H_ll + sqrt_rho_rr * H_rr) * inv_sum_sqrt_rho
  c_roe = sqrt((equations.gamma - 1) * (H_roe - 0.5 * v_roe_mag))

  # Compute convenience constant for positivity preservation, see
  # https://doi.org/10.1016/0021-9991(91)90211-3
  beta = sqrt(0.5 * (equations.gamma - 1) / equations.gamma)

  # Estimate the edges of the Riemann fan (with positivity conservation)
  if orientation == 1 # x-direction
    SsL = min(v1_roe - c_roe, v1_ll - beta * c_ll, zero(v1_roe))
    SsR = max(v1_roe + c_roe, v1_rr + beta * c_rr, zero(v1_roe))
  elseif orientation == 2 # y-direction
    SsL = min(v2_roe - c_roe, v2_ll - beta * c_ll, zero(v2_roe))
    SsR = max(v2_roe + c_roe, v2_rr + beta * c_rr, zero(v2_roe))
  end

  if SsL >= 0.0 && SsR > 0.0
    # Positive supersonic speed
    f_ll = flux(u_ll, orientation, equations)

    f1 = f_ll[1]
    f2 = f_ll[2]
    f3 = f_ll[3]
    f4 = f_ll[4]
  elseif SsR <= 0.0 && SsL < 0.0
    # Negative supersonic speed
    f_rr = flux(u_rr, orientation, equations)

    f1 = f_rr[1]
    f2 = f_rr[2]
    f3 = f_rr[3]
    f4 = f_rr[4]
  else
    # Subsonic case
    # Compute left and right fluxes
    f_ll = flux(u_ll, orientation, equations)
    f_rr = flux(u_rr, orientation, equations)

    f1 = (SsR * f_ll[1] - SsL * f_rr[1] + SsL * SsR * (u_rr[1] - u_ll[1])) / (SsR - SsL)
    f2 = (SsR * f_ll[2] - SsL * f_rr[2] + SsL * SsR * (u_rr[2] - u_ll[2])) / (SsR - SsL)
    f3 = (SsR * f_ll[3] - SsL * f_rr[3] + SsL * SsR * (u_rr[3] - u_ll[3])) / (SsR - SsL)
    f4 = (SsR * f_ll[4] - SsL * f_rr[4] + SsL * SsR * (u_rr[4] - u_ll[4])) / (SsR - SsL)
  end

  return SVector(f1, f2, f3, f4)
end


@inline function max_abs_speeds(u, equations::CompressibleEulerEquations2D)
  rho, v1, v2, p = cons2prim(u, equations)
  c = sqrt(equations.gamma * p / rho)

  return abs(v1) + c, abs(v2) + c
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1 * v1 + rho_v2 * v2))

  return SVector(rho, v1, v2, p)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square)
  s = log(p) - equations.gamma*log(rho)
  rho_p = rho / p

  w1 = (equations.gamma - s) * equations.inv_gamma_minus_one - 0.5 * rho_p * v_square
  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = -rho_p

  return SVector(w1, w2, w3, w4)
end

@inline function entropy2cons(w, equations::CompressibleEulerEquations2D)
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
  rho_e    =  rho_iota * (1-(V2^2 + V3^2)/(2*V5))
  return SVector(rho, rho_v1, rho_v2, rho_e)
end




# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::CompressibleEulerEquations2D)
  rho, v1, v2, p = prim
  rho_v1 = rho * v1
  rho_v2 = rho * v2
  rho_e  = p * equations.inv_gamma_minus_one + 0.5 * (rho_v1 * v1 + rho_v2 * v2)
  return SVector(rho, rho_v1, rho_v2, rho_e)
end


@inline function density(u, equations::CompressibleEulerEquations2D)
 rho = u[1]
 return rho
end


@inline function pressure(u, equations::CompressibleEulerEquations2D)
 rho, rho_v1, rho_v2, rho_e = u
 p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
 return p
end


@inline function density_pressure(u, equations::CompressibleEulerEquations2D)
 rho, rho_v1, rho_v2, rho_e = u
 rho_times_p = (equations.gamma - 1) * (rho * rho_e - 0.5 * (rho_v1^2 + rho_v2^2))
 return rho_times_p
end


# Calculates the entropy flux in direction "orientation" and the entropy variables for a state cons
# NOTE: This method seems to work currently (b82534e) but is never used anywhere. Thus it is
# commented here until someone uses it or writes a test for it.
# @inline function cons2entropyvars_and_flux(gamma::Float64, cons, orientation::Int)
#   entropy = MVector{4, Float64}(undef)
#   v = (cons[2] / cons[1] , cons[3] / cons[1])
#   v_square= v[1]*v[1]+v[2]*v[2]
#   p = (gamma - 1) * (cons[4] - 1/2 * (cons[2] * v[1] + cons[3] * v[2]))
#   rho_p = cons[1] / p
#   # thermodynamic entropy
#   s = log(p) - gamma*log(cons[1])
#   # mathematical entropy
#   S = - s*cons[1]/(gamma-1)
#   # entropy variables
#   entropy[1] = (gamma - s)/(gamma-1) - 0.5*rho_p*v_square
#   entropy[2] = rho_p*v[1]
#   entropy[3] = rho_p*v[2]
#   entropy[4] = -rho_p
#   # entropy flux
#   entropy_flux = S*v[orientation]
#   return entropy, entropy_flux
# end


# Calculate thermodynamic entropy for a conservative state `cons`
@inline function entropy_thermodynamic(cons, equations::CompressibleEulerEquations2D)
  # Pressure
  p = (equations.gamma - 1) * (cons[4] - 1/2 * (cons[2]^2 + cons[3]^2) / cons[1])

  # Thermodynamic entropy
  s = log(p) - equations.gamma*log(cons[1])

  return s
end


# Calculate mathematical entropy for a conservative state `cons`
@inline function entropy_math(cons, equations::CompressibleEulerEquations2D)
  # Mathematical entropy
  S = -entropy_thermodynamic(cons, equations) * cons[1] * equations.inv_gamma_minus_one

  return S
end


# Default entropy is the mathematical entropy
@inline entropy(cons, equations::CompressibleEulerEquations2D) = entropy_math(cons, equations)


# Calculate total energy for a conservative state `cons`
@inline energy_total(cons, ::CompressibleEulerEquations2D) = cons[4]


# Calculate kinetic energy for a conservative state `cons`
@inline function energy_kinetic(u, equations::CompressibleEulerEquations2D)
  rho, rho_v1, rho_v2, rho_e = u
  return (rho_v1^2 + rho_v2^2) / (2 * rho)
end


# Calculate internal energy for a conservative state `cons`
@inline function energy_internal(cons, equations::CompressibleEulerEquations2D)
  return energy_total(cons, equations) - energy_kinetic(cons, equations)
end


end # @muladd
