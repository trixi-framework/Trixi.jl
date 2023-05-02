# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    HyperbolicDiffusionEquations3D

The linear hyperbolic diffusion equations in three space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at http://www.cfdbooks.com/ and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations3D{RealT<:Real} <: AbstractHyperbolicDiffusionEquations{3, 4}
  Lr::RealT     # reference length scale
  inv_Tr::RealT # inverse of the reference time scale
  nu::RealT     # diffusion constant
end

function HyperbolicDiffusionEquations3D(; nu=1.0, Lr=inv(2pi))
  Tr = Lr^2 / nu
  HyperbolicDiffusionEquations3D(promote(Lr, inv(Tr), nu)...)
end


varnames(::typeof(cons2cons), ::HyperbolicDiffusionEquations3D) = ("phi", "q1", "q2", "q3")
varnames(::typeof(cons2prim), ::HyperbolicDiffusionEquations3D) = ("phi", "q1", "q2", "q3")
default_analysis_errors(::HyperbolicDiffusionEquations3D)     = (:l2_error, :linf_error, :residual)

"""
    residual_steady_state(du, ::AbstractHyperbolicDiffusionEquations)

Used to determine the termination criterion of a [`SteadyStateCallback`](@ref).
For hyperbolic diffusion, this checks convergence of the potential ``\\phi``.
"""
@inline function residual_steady_state(du, ::HyperbolicDiffusionEquations3D)
  abs(du[1])
end


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
    q3  = 1.0
  else
    phi =  2.0 *      cos(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3]) + 2.0 # ϕ
    q1  = -2.0 * pi * sin(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_x
    q2  =  4.0 * pi * cos(pi * x[1]) * cos(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_y
    q3  =  4.0 * pi * cos(pi * x[1]) * sin(2.0 * pi * x[2]) * cos(2.0 * pi * x[3])   # ϕ_z
  end
  return SVector(phi, q1, q2, q3)
end

@inline function source_terms_poisson_nonperiodic(u, x, t, equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = 2 cos(πx)sin(2πy)sin(2πz) + 2 and f = 18 π^2 cos(πx)sin(2πy)sin(2πz)
  @unpack inv_Tr = equations

  x1, x2, x3 = x
  du1 = 18 * pi^2 * cospi(x1) * sinpi(2 * x2) * sinpi(2 * x3)
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]
  du4 = -inv_Tr * u[4]

  return SVector(du1, du2, du3, du4)
end

function boundary_condition_poisson_nonperiodic(u_inner, orientation, direction, x, t,
                                                 surface_flux_function,
                                                 equations::HyperbolicDiffusionEquations3D)
  # elliptic equation: -νΔϕ = f
  phi =  2.0 *      cos(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3]) + 2.0 # ϕ
  q1  = -2.0 * pi * sin(pi * x[1]) * sin(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_x
  q2  =  4.0 * pi * cos(pi * x[1]) * cos(2.0 * pi * x[2]) * sin(2.0 * pi * x[3])   # ϕ_y
  q3  =  4.0 * pi * cos(pi * x[1]) * sin(2.0 * pi * x[2]) * cos(2.0 * pi * x[3])   # ϕ_z
  u_boundary = SVector(phi, q1, q2, q3)

  # Calculate boundary flux
  if iseven(direction) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


"""
    source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations3D)

Source term that only includes the forcing from the hyperbolic diffusion system.
"""
@inline function source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations3D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  @unpack inv_Tr = equations

  du1 = zero(u[1])
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]
  du4 = -inv_Tr * u[4]

  return SVector(du1, du2, du3, du4)
end


"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations3D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`source_terms_harmonic`](@ref).
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations3D)

  # Determine phi_x, phi_y
  G = 1.0 # gravitational constant
  C_grav = -4 * G / (3 * pi) # "3" is the number of spatial dimensions  # 2D: -2.0*G/pi
  A = 0.1 # perturbation coefficient must match Euler setup
  rho1 = A * sin(pi * (x[1] + x[2] + x[3] - t))
  # initialize with ansatz of gravity potential
  phi = C_grav * rho1
  q1  = C_grav * A * pi * cos(pi*(x[1] + x[2] + x[3] - t)) # = gravity acceleration in x-direction
  q2  = q1                                                 # = gravity acceleration in y-direction
  q3  = q1                                                 # = gravity acceleration in z-direction

  return SVector(phi, q1, q2, q3)
end



# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicDiffusionEquations3D)
  phi, q1, q2, q3 = u

  if orientation == 1
    f1 = -equations.nu*q1
    f2 = -phi * equations.inv_Tr
    f3 = zero(phi)
    f4 = zero(phi)
  elseif orientation == 2
    f1 = -equations.nu*q2
    f2 = zero(phi)
    f3 = -phi * equations.inv_Tr
    f4 = zero(phi)
  else
    f1 = -equations.nu*q3
    f2 = zero(phi)
    f3 = zero(phi)
    f4 = -phi * equations.inv_Tr
  end

  return SVector(f1, f2, f3, f4)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicDiffusionEquations3D)
  λ_max = sqrt(equations.nu * equations.inv_Tr)
end


@inline function flux_godunov(u_ll, u_rr, orientation::Integer, equations::HyperbolicDiffusionEquations3D)
  # Obtain left and right fluxes
  phi_ll, q1_ll, q2_ll, q3_ll = u_ll
  phi_rr, q1_rr, q2_rr, q3_rr = u_rr
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equations.nu * equations.inv_Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (q1_rr - q1_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
    f4 = 1/2 * (f_ll[4] + f_rr[4])
  elseif orientation == 2 # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q2_rr - q2_ll)
    f4 = 1/2 * (f_ll[4] + f_rr[4])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3])
    f4 = 1/2 * (f_ll[4] + f_rr[4]) - 1/2 * λ_max * (q3_rr - q3_ll)
  end

  return SVector(f1, f2, f3, f4)
end



@inline have_constant_speed(::HyperbolicDiffusionEquations3D) = True()

@inline function max_abs_speeds(eq::HyperbolicDiffusionEquations3D)
  λ = sqrt(eq.nu * eq.inv_Tr)
  return λ, λ, λ
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicDiffusionEquations3D) = u


# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicDiffusionEquations3D)
  phi, q1, q2, q3 = u
  w1 = phi
  w2 = equations.Lr^2 * q1
  w3 = equations.Lr^2 * q2
  w4 = equations.Lr^2 * q3

  return SVector(w1, w2, w3, w4)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicDiffusionEquations3D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicDiffusionEquations3D)
  # energy function as found in equation (2.5.12) in the book "I Do Like CFD, Vol. 1"
  phi, q1, q2, q3 = u
  return 0.5 * (phi^2 + equations.Lr^2 * (q1^2 + q2^2 + q3^2))
end


end # @muladd
