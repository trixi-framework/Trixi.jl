
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


get_name(::HyperbolicDiffusionEquations1D) = "HyperbolicDiffusionEquations1D"
varnames_cons(::HyperbolicDiffusionEquations1D) = @SVector ["phi", "q1"]
varnames_prim(::HyperbolicDiffusionEquations1D) = @SVector ["phi", "q1"]
default_analysis_errors(::HyperbolicDiffusionEquations1D) = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicDiffusionEquations1D)
  abs(du[1])
end

"""
    initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations1D)

A non-priodic smooth initial condition. Can be used for convergence tests in combination with
[`source_terms_poisson_nonperiodic`](@ref) and [`boundary_condition_poisson_nonperiodic`](@ref).
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
  return @SVector [phi, q1]
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

  du1 = pi^2 * sinpi(x[1])
  du2 = -inv_Tr * u[2]

  return SVector(du1, du2)
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
  u_boundary = @SVector [phi, q1]

  # Calculate boundary flux
  if direction == 2 # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


# function initial_condition_harmonic_nonperiodic(x, t, equations::HyperbolicDiffusionEquations1D)
#   # elliptic equation: -νΔϕ = f
#   if t == 0.0
#     phi = 1.0
#     q1  = 1.0
#     q2  = 1.0
#   else
#     C   = 1.0/sinh(pi)
#     phi = C*(sinh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*sin(pi*x[1]))
#     q1  = C*pi*(cosh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*cos(pi*x[1]))
#     q2  = C*pi*(sinh(pi*x[1])*cos(pi*x[2]) + cosh(pi*x[2])*sin(pi*x[1]))
#   end
#   return @SVector [phi, q1, q2]
# end
#
# @inline function source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations1D)
#   # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
#   @unpack inv_Tr = equations
#   phi, q1, q2 = u
#
#   du2 = -inv_Tr * q1
#   du3 = -inv_Tr * q2
#
#   return SVector(0, du2, du3)
# end
#
# function boundary_condition_harmonic_nonperiodic(u_inner, orientation, direction, x, t,
#                                                   surface_flux_function,
#                                                   equations::HyperbolicDiffusionEquations1D)
#   # elliptic equation: -νΔϕ = f
#   C   = 1.0/sinh(pi)
#   phi = C*(sinh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*sin(pi*x[1]))
#   q1  = C*pi*(cosh(pi*x[1])*sin(pi*x[2]) + sinh(pi*x[2])*cos(pi*x[1]))
#   q2  = C*pi*(sinh(pi*x[1])*cos(pi*x[2]) + cosh(pi*x[2])*sin(pi*x[1]))
#   u_boundary = @SVector [phi, q1, q2]
#
#   # Calculate boundary flux
#   if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
#     flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
#   else # u_boundary is "left" of boundary, u_inner is "right" of boundary
#     flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
#   end
#
#   return flux
# end


# Calculate 1D flux in for a single point
@inline function calcflux(u, orientation, equations::HyperbolicDiffusionEquations1D)
  phi, q1 = u
  @unpack inv_Tr = equations

  # Ignore orientation since it is always "1" in 1D
  f1 = -equations.nu * q1
  f2 = -phi * inv_Tr

  return SVector(f1, f2)
end


@inline function flux_lax_friedrichs(u_ll, u_rr, orientation, equations::HyperbolicDiffusionEquations1D)
  # Obtain left and right fluxes
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  λ_max = sqrt(equations.nu * equations.inv_Tr)

  return 0.5 * (f_ll + f_rr - λ_max * (u_rr - u_ll))
end


@inline function flux_upwind(u_ll, u_rr, orientation, equations::HyperbolicDiffusionEquations1D)
  # Obtain left and right fluxes
  phi_ll, q1_ll = u_ll
  phi_rr, q1_rr = u_rr
  f_ll = calcflux(u_ll, orientation, equations)
  f_rr = calcflux(u_rr, orientation, equations)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  # Ignore orientation since it is always "1" in 1D
  λ_max = sqrt(equations.nu * equations.inv_Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (q1_rr - q1_ll)

  return SVector(f1, f2, f3)
end



@inline have_constant_speed(::HyperbolicDiffusionEquations1D) = Val(true)

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
