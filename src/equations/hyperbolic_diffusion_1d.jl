# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    HyperbolicDiffusionEquations1D

The linear hyperbolic diffusion equations in one space dimension.
A description of this system can be found in Sec. 2.5 of the book
- Masatsuka (2013)
  I Do Like CFD, Too: Vol 1.
  Freely available at [http://www.cfdbooks.com/](http://www.cfdbooks.com/)
Further analysis can be found in the paper
- Nishikawa (2007)
  A first-order system approach for diffusion equation. I: Second-order residual-distribution
  schemes
  [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations1D{RealT<:Real} <: AbstractHyperbolicDiffusionEquations{1, 2}
  Lr::RealT     # reference length scale
  inv_Tr::RealT # inverse of the reference time scale
  nu::RealT     # diffusion constant
end

function HyperbolicDiffusionEquations1D(; nu=1.0, Lr=inv(2pi))
  Tr = Lr^2 / nu
  HyperbolicDiffusionEquations1D(promote(Lr, inv(Tr), nu)...)
end


varnames(::typeof(cons2cons), ::HyperbolicDiffusionEquations1D) = ("phi", "q1")
varnames(::typeof(cons2prim), ::HyperbolicDiffusionEquations1D) = ("phi", "q1")
default_analysis_errors(::HyperbolicDiffusionEquations1D) = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicDiffusionEquations1D)
  abs(du[1])
end

"""
    initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations1D)

A non-priodic smooth initial condition. Can be used for convergence tests in combination with
[`source_terms_poisson_nonperiodic`](@ref) and [`boundary_condition_poisson_nonperiodic`](@ref).
!!! note
    The solution is periodic but the initial guess is not.
"""
function initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations1D)
  # elliptic equation: -νΔϕ = f
  # Taken from Section 6.1 of Nishikawa https://doi.org/10.1016/j.jcp.2007.07.029
  if t == 0.0
    # initial "guess" of the solution and its derivative
    phi = x[1]^2 - x[1]
    q1  = 2*x[1] - 1
  else
    phi = sinpi(x[1])      # ϕ
    q1  = pi * cospi(x[1]) # ϕ_x
  end
  return SVector(phi, q1)
end

"""
    source_terms_poisson_nonperiodic(u, x, t,
                                     equations::HyperbolicDiffusionEquations1D)

Source terms that include the forcing function `f(x)` and right hand side for the hyperbolic
diffusion system that is used with [`initial_condition_poisson_nonperiodic`](@ref) and
[`boundary_condition_poisson_nonperiodic`](@ref).
"""
@inline function source_terms_poisson_nonperiodic(u, x, t,
                                                  equations::HyperbolicDiffusionEquations1D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: ϕ = sin(πx) and f = π^2sin(πx)
  @unpack inv_Tr = equations

  dphi = pi^2 * sinpi(x[1])
  dq1  = -inv_Tr * u[2]

  return SVector(dphi, dq1)
end

"""
    boundary_condition_poisson_nonperiodic(u_inner, orientation, direction, x, t,
                                           surface_flux_function,
                                           equations::HyperbolicDiffusionEquations1D)

Boundary conditions used for convergence tests in combination with
[`initial_condition_poisson_nonperiodic`](@ref) and [`source_terms_poisson_nonperiodic`](@ref).
"""
function boundary_condition_poisson_nonperiodic(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::HyperbolicDiffusionEquations1D)
  # elliptic equation: -νΔϕ = f
  phi = sinpi(x[1])      # ϕ
  q1  = pi * cospi(x[1]) # ϕ_x
  u_boundary = SVector(phi, q1)

  # Calculate boundary flux
  if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


"""
    source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations1D)

Source term that only includes the forcing from the hyperbolic diffusion system.
"""
@inline function source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations1D)
  # harmonic solution of the form ϕ = A + B * x, so f = 0
  @unpack inv_Tr = equations

  dq1 = -inv_Tr * u[2]

  return SVector(zero(dq1), dq1)
end


"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations1D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`source_terms_harmonic`](@ref).
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations1D)

  # Determine phi_x
  G = 1.0           # gravitational constant
  C = -4.0 * G / pi # -4 * G / ndims * pi
  A = 0.1           # perturbation coefficient must match Euler setup
  rho1 = A * sinpi(x[1] - t)
  # initialize with ansatz of gravity potential
  phi = C * rho1
  q1  = C * A * pi * cospi(x[1] - t) # = gravity acceleration in x-direction

  return SVector(phi, q1)
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicDiffusionEquations1D)
  phi, q1 = u
  @unpack inv_Tr = equations

  # Ignore orientation since it is always "1" in 1D
  f1 = -equations.nu * q1
  f2 = -phi * inv_Tr

  return SVector(f1, f2)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicDiffusionEquations1D)
  λ_max = sqrt(equations.nu * equations.inv_Tr)
end


@inline have_constant_speed(::HyperbolicDiffusionEquations1D) = True()

@inline function max_abs_speeds(eq::HyperbolicDiffusionEquations1D)
  return sqrt(eq.nu * eq.inv_Tr)
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicDiffusionEquations1D) = u

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicDiffusionEquations1D)
  phi, q1 = u

  w1 = phi
  w2 = equations.Lr^2 * q1

  return SVector(w1, w2)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicDiffusionEquations1D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicDiffusionEquations1D)
  # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
  phi, q1 = u
  return 0.5 * (phi^2 + equations.Lr^2 * q1^2)
end


end # @muladd
