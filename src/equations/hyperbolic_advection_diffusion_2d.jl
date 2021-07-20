# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


@doc raw"""
    HyperbolicAdvectionDiffusionEquations

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
struct HyperbolicAdvectionDiffusionEquations2D{RealT<:Real} <: AbstractHyperbolicAdvectionDiffusionEquations{2, 3}
  advectionvelocity::SVector{2, RealT}
  Lr::RealT     # reference length scale
  inv_Tr::RealT # inverse of the reference time scale
  nu::RealT     # diffusion constant
end

function HyperbolicAdvectionDiffusionEquations2D(a::NTuple{2,<:Real}; nu=1.0, Lr=inv(2pi))
  Tr = Lr^2 / nu
  HyperbolicAdvectionDiffusionEquations2D(SVector(a), promote(Lr, inv(Tr), nu)...)
end

function HyperbolicAdvectionDiffusionEquations2D(a1::Real, a2::Real; nu=1.0, Lr=inv(2pi))
  Tr = Lr^2 / nu
  HyperbolicAdvectionDiffusionEquations2D(SVector(a1, a2), promote(Lr, inv(Tr), nu)...)
end


varnames(::typeof(cons2cons), ::HyperbolicAdvectionDiffusionEquations2D) = ("phi", "q1", "q2")
varnames(::typeof(cons2prim), ::HyperbolicAdvectionDiffusionEquations2D) = ("phi", "q1", "q2")
default_analysis_errors(::HyperbolicAdvectionDiffusionEquations2D)     = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicAdvectionDiffusionEquations2D)
  abs(du[1])
end


@inline function source_terms_harmonic(u, x, t, equations::HyperbolicAdvectionDiffusionEquations2D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  @unpack inv_Tr = equations
  phi, q1, q2 = u

  du2 = -inv_Tr * q1
  du3 = -inv_Tr * q2

  return SVector(0, du2, du3)
end


"""
    Expample "exp_nonperiodic"
    The boundary conditions are nonperiodic. This example is solved by an
    exponential function.
"""

@inline function initial_condition_exp_nonperiodic(x, t, equations::HyperbolicAdvectionDiffusionEquations2D)
  @unpack advectionvelocity, nu = equations
  c = advectionvelocity[1] / nu
  d = advectionvelocity[2] / nu
  v = exp(c*x[1] + d*x[2])
  q1 = c*v
  q2 = d*v
  return SVector(v, q1, q2)
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicAdvectionDiffusionEquations2D)
  v, q1, q2 = u
  @unpack inv_Tr, advectionvelocity = equations

  if orientation == 1
    f1 = -equations.nu*q1
    f2 = -v * inv_Tr
    f3 = zero(v)
  else
    f1 = -equations.nu*q2
    f2 = zero(v)
    f3 = -v * inv_Tr
  end
  f1 += advectionvelocity[orientation] * v

  return SVector(f1, f2, f3)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicAdvectionDiffusionEquations2D)
  λ_max = abs(equations.advectionvelocity[orientation]) + sqrt(equations.nu * equations.inv_Tr)
end


@inline function flux_godunov(u_ll, u_rr, orientation::Integer, equations::HyperbolicAdvectionDiffusionEquations2D)
  # Obtain left and right fluxes
  v_ll, p_ll, q_ll = u_ll
  v_rr, p_rr, q_rr = u_rr
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = abs(equations.advectionvelocity[orientation]) + sqrt(equations.nu * equations.inv_Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (v_rr - v_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (p_rr - p_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q_rr - q_ll)
  end

  return SVector(f1, f2, f3)
end



@inline have_constant_speed(::HyperbolicAdvectionDiffusionEquations2D) = Val(true)

@inline function max_abs_speeds(eq::HyperbolicAdvectionDiffusionEquations2D)
  λ = abs.(eq.advectionvelocity) .+ sqrt(eq.nu * eq.inv_Tr)
  return λ
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicAdvectionDiffusionEquations2D) = u

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicAdvectionDiffusionEquations2D)
  v, q1, q2 = u
  w1 = v
  w2 = equations.Lr^2 * q1
  w3 = equations.Lr^2 * q2

  return SVector(w1, w2, w3)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicAdvectionDiffusionEquations2D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicAdvectionDiffusionEquations2D)
  # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
  v, q1, q2 = u
  return 0.5 * (v^2 + equations.Lr^2 * (q1^2 + q2^2))
end


end # @muladd
