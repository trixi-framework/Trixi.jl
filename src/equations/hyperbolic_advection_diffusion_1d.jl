# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    HyperbolicDiffusionEquations1D

The linear hyperbolic advection diffusion equations in one space dimension.
A description of this system can be found in the book
- Masatsuka (2013)
  I Do Like CFD, Too: Vol 1.
  Freely available at [http://www.cfdbooks.com/](http://www.cfdbooks.com/)
Further analysis can be found in the paper
- Nishikawa (2014)
  First, second, and third order finite-volume schemes for advection–diffusion
  Journal of Computational Physics (P. 287–309)
"""
struct HyperbolicAdvectionDiffusionEquations1D{RealT<:Real} <: AbstractHyperbolicAdvectionDiffusionEquations{1, 2}
  advectionvelocity::SVector{1, RealT}
  Lr::RealT     # reference length scale
  inv_Tr::RealT # inverse of the reference time scale
  nu::RealT     # diffusion constant
end

function HyperbolicAdvectionDiffusionEquations1D(a::Real; nu=1.0, Lr=inv(2pi))
  # Tr = Lr^2 / nu
  Tr = Lr / (abs(a) + nu / Lr)
  HyperbolicAdvectionDiffusionEquations1D(SVector(a), promote(Lr, inv(Tr), nu)...)
end


varnames(::typeof(cons2cons), ::HyperbolicAdvectionDiffusionEquations1D) = ("scalar", "q1")
varnames(::typeof(cons2prim), ::HyperbolicAdvectionDiffusionEquations1D) = ("scalar", "q1")
default_analysis_errors(::HyperbolicAdvectionDiffusionEquations1D) = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicAdvectionDiffusionEquations1D)
  abs(du[1])
end

"""
    Example "idlCFD"
    This example is taken from the free book "I Do Like CFD".
"""
function initial_condition_idlCFD_nonperiodic(x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  @unpack advectionvelocity, nu = equations

  Re = advectionvelocity[1] / nu
  v = (1-exp(x[1]*Re)) / (1-exp(Re))
  q1 = (-Re*exp(x[1]*Re)) / (1-exp(Re))
  return SVector(v, q1)
end

"""
    Example "mytest"
    As a first test function a quadratic polynomial is implemented. The
    boundary conditions are nonperiodic. This equation should be solved exactly
    by using polynomials of second degree.
"""
function initial_condition_mytest_nonperiodic(x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  c = 1.0
  v = c * x[1] + 0.5 * c * x[1]^2 - c * t * x[1] + 0.5 * c * t^2
  q1 = c + c*x[1] - c*t
  return SVector(v, q1)
end

"""
    Expample "myexp"
    The boundary conditions are nonperiodic. This example is solved by an
    exponential function.
"""
function initial_condition_myexp_nonperiodic(x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  @unpack advectionvelocity, nu = equations
  c = 2.0
  v = exp(c*x[1] + (nu*c^2-advectionvelocity[1]*c)*t)
  q1 = c * v
  return SVector(v, q1)
end

@inline function source_terms_harmonic(u, x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  # harmonic solution of the form ϕ = A + B * x, so f = 0
  @unpack inv_Tr = equations

  dq1 = -inv_Tr * u[2]

  return SVector(zero(dq1), dq1)
end

"""
    Example "sin"
    This example is taken from the paper "First-Order Hyperbolic System Method
    for Time-Dependent Advection Diffusion Problems" published by Allireza
    Mazaheri in 2014. Periodic boundary conditions are used.
"""

function initial_condition_sin_periodic(x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  @unpack advectionvelocity, nu = equations
  kappa = 2.0*pi
  v = exp(-kappa^2*nu*t)*sin(kappa*(x[1]-advectionvelocity[1]*t))
  q1 = kappa*exp(-kappa^2*nu*t)*cos(kappa*(x[1]-advectionvelocity[1]*t))
  return SVector(v, q1)
end

function initial_condition_sin_nonperiodic(x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  @unpack advectionvelocity, nu = equations
  if t == 0.0
    # initial "guess" of the solution and its derivative
    v = x[1]^2 - x[1]
    q1  = 2*x[1] - 1
  else
    v = sinpi(2*x[1])
    q1 = 2*pi*cospi(2*x[1])
  end
  return SVector(v, q1)
end

@inline function source_terms_sin_nonperiodic(u, x, t, equations::HyperbolicAdvectionDiffusionEquations1D)
  # harmonic solution of the form ϕ = A + B * x, so f = 0
  @unpack advectionvelocity, nu, inv_Tr = equations

  du1 = 2*advectionvelocity[1]*pi*cospi(2*x[1])+4*nu*pi^2*sinpi(2*x[1])
  du2 = -inv_Tr * u[2]

  return SVector(du1, du2)
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicAdvectionDiffusionEquations1D)
  v, q1 = u
  @unpack inv_Tr, advectionvelocity = equations

  # Ignore orientation since it is always "1" in 1D
  f1 = advectionvelocity[orientation] * v - equations.nu * q1
  f2 = -v * inv_Tr

  return SVector(f1, f2)
end


@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicAdvectionDiffusionEquations1D)
  λ_max = abs(equations.advectionvelocity[orientation]) + sqrt(equations.nu * equations.inv_Tr)
end


@inline have_constant_speed(::HyperbolicAdvectionDiffusionEquations1D) = Val(true)

@inline function max_abs_speeds(eq::HyperbolicAdvectionDiffusionEquations1D)
  return abs(eq.advectionvelocity[1]) + sqrt(eq.nu * eq.inv_Tr)
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicAdvectionDiffusionEquations1D) = u

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicAdvectionDiffusionEquations1D)
  v, q1 = u

  w1 = v
  w2 = equations.Lr^2 * q1

  return SVector(w1, w2)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicAdvectionDiffusionEquations1D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicAdvectionDiffusionEquations1D)
  # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
  v, q1 = u
  return 0.5 * (v^2 + equations.Lr^2 * q1^2)
end


end # @muladd
