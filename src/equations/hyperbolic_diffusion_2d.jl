
@doc raw"""
    HyperbolicDiffusionEquations2D

The linear hyperbolic diffusion equations in two space dimensions.
A description of this system can be found in Sec. 2.5 of the book "I Do Like CFD, Too: Vol 1".
The book is freely available at http://www.cfdbooks.com/ and further analysis can be found in
the paper by Nishikawa [DOI: 10.1016/j.jcp.2007.07.029](https://doi.org/10.1016/j.jcp.2007.07.029)
"""
struct HyperbolicDiffusionEquations2D{RealT<:Real} <: AbstractHyperbolicDiffusionEquations{2, 3}
  Lr::RealT     # reference length scale
  inv_Tr::RealT # inverse of the reference time scale
  nu::RealT     # diffusion constant
end

function HyperbolicDiffusionEquations2D(; nu=1.0, Lr=inv(2pi))
  Tr = Lr^2 / nu
  HyperbolicDiffusionEquations2D(promote(Lr, inv(Tr), nu)...)
end


varnames(::typeof(cons2cons), ::HyperbolicDiffusionEquations2D) = ("phi", "q1", "q2")
varnames(::typeof(cons2prim), ::HyperbolicDiffusionEquations2D) = ("phi", "q1", "q2")
default_analysis_errors(::HyperbolicDiffusionEquations2D)     = (:l2_error, :linf_error, :residual)

@inline function residual_steady_state(du, ::HyperbolicDiffusionEquations2D)
  abs(du[1])
end


# Set initial conditions at physical location `x` for pseudo-time `t`
function initial_condition_poisson_periodic(x, t, equations::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  # depending on initial constant state, c, for phi this converges to the solution ϕ + c
  if iszero(t)
    phi = 0.0
    q1  = 0.0
    q2  = 0.0
  else
    phi = sin(2.0*pi*x[1])*sin(2.0*pi*x[2])
    q1  = 2*pi*cos(2.0*pi*x[1])*sin(2.0*pi*x[2])
    q2  = 2*pi*sin(2.0*pi*x[1])*cos(2.0*pi*x[2])
  end
  return SVector(phi, q1, q2)
end

@inline function source_terms_poisson_periodic(u, x, t, equations::HyperbolicDiffusionEquations2D)
  # elliptic equation: -νΔϕ = f
  # analytical solution: phi = sin(2πx)*sin(2πy) and f = -8νπ^2 sin(2πx)*sin(2πy)
  @unpack inv_Tr = equations
  C = -8 * equations.nu * pi^2

  x1, x2 = x
  tmp1 = sinpi(2 * x1)
  tmp2 = sinpi(2 * x2)
  du1 = -C*tmp1*tmp2
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]

  return SVector(du1, du2, du3)
end


@inline function initial_condition_poisson_nonperiodic(x, t, equations::HyperbolicDiffusionEquations2D)
  # elliptic equation: -ν Δϕ = f in Ω, u = g on ∂Ω
  if iszero(t)
    T = eltype(x)
    phi = one(T)
    q1  = one(T)
    q2  = one(T)
  else
    sinpi_x1,  cospi_x1  = sincos(pi*x[1])
    sinpi_2x2, cospi_2x2 = sincos(pi*2*x[2])
    phi =  2 *      cospi_x1 * sinpi_2x2 + 2 # ϕ
    q1  = -2 * pi * sinpi_x1 * sinpi_2x2     # ϕ_x
    q2  =  4 * pi * cospi_x1 * cospi_2x2     # ϕ_y
  end
  return SVector(phi, q1, q2)
end

@inline function source_terms_poisson_nonperiodic(u, x, t, equations::HyperbolicDiffusionEquations2D)
  # elliptic equation: -ν Δϕ = f in Ω, u = g on ∂Ω
  # analytical solution: ϕ = 2cos(πx)sin(2πy) + 2 and f = 10π^2cos(πx)sin(2πy)
  @unpack inv_Tr = equations

  x1, x2 = x
  du1 = 10 * pi^2 * cospi(x1) * sinpi(2 * x2)
  du2 = -inv_Tr * u[2]
  du3 = -inv_Tr * u[3]

  return SVector(du1, du2, du3)
end

@inline function boundary_condition_poisson_nonperiodic(u_inner, orientation, direction, x, t,
                                                        surface_flux_function,
                                                        equations::HyperbolicDiffusionEquations2D)
  # elliptic equation: -ν Δϕ = f in Ω, u = g on ∂Ω
  u_boundary = initial_condition_poisson_nonperiodic(x, one(t), equations)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


@inline function initial_condition_harmonic_nonperiodic(x, t, equations::HyperbolicDiffusionEquations2D)
  # elliptic equation: -ν Δϕ = 0 in Ω, u = g on ∂Ω
  if t == 0.0
    phi = 1.0
    q1  = 1.0
    q2  = 1.0
  else
    C   = inv(sinh(pi))
    sinpi_x1, cospi_x1 = sincos(pi*x[1])
    sinpi_x2, cospi_x2 = sincos(pi*x[2])
    sinh_pix1 = sinh(pi*x[1])
    cosh_pix1 = cosh(pi*x[1])
    sinh_pix2 = sinh(pi*x[2])
    cosh_pix2 = cosh(pi*x[2])
    phi = C *      (sinh_pix1 * sinpi_x2 + sinh_pix2 * sinpi_x1)
    q1  = C * pi * (cosh_pix1 * sinpi_x2 + sinh_pix2 * cospi_x1)
    q2  = C * pi * (sinh_pix1 * cospi_x2 + cosh_pix2 * sinpi_x1)
  end
  return SVector(phi, q1, q2)
end

@inline function source_terms_harmonic(u, x, t, equations::HyperbolicDiffusionEquations2D)
  # harmonic solution ϕ = (sinh(πx)sin(πy) + sinh(πy)sin(πx))/sinh(π), so f = 0
  @unpack inv_Tr = equations
  phi, q1, q2 = u

  du2 = -inv_Tr * q1
  du3 = -inv_Tr * q2

  return SVector(0, du2, du3)
end


"""
    initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations2D)

Setup used for convergence tests of the Euler equations with self-gravity used in
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
in combination with [`source_terms_harmonic`](@ref).
"""
function initial_condition_eoc_test_coupled_euler_gravity(x, t, equations::HyperbolicDiffusionEquations2D)

  # Determine phi_x, phi_y
  G = 1.0 # gravitational constant
  C = -2.0*G/pi
  A = 0.1 # perturbation coefficient must match Euler setup
  rho1 = A * sin(pi * (x[1] + x[2] - t))
  # intialize with ansatz of gravity potential
  phi = C * rho1
  q1  = C * A * pi * cos(pi*(x[1] + x[2] - t)) # = gravity acceleration in x-direction
  q2  = q1                                     # = gravity acceleration in y-direction

  return SVector(phi, q1, q2)
end


"""
    initial_condition_sedov_self_gravity(x, t, equations::HyperbolicDiffusionEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
Should be used together with [`boundary_condition_sedov_self_gravity`](@ref).
"""
function initial_condition_sedov_self_gravity(x, t, equations::HyperbolicDiffusionEquations2D)
  # for now just use constant initial condition for sedov blast wave (can likely be improved)
  phi = 0.0
  q1  = 0.0
  q2  = 0.0
  return SVector(phi, q1, q2)
end

"""
    boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                          surface_flux_function,
                                          equations::HyperbolicDiffusionEquations2D)

Adaptation of the Sedov blast wave with self-gravity taken from
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics
  [arXiv: 2008.10593](https://arxiv.org/abs/2008.10593)
based on
- http://flash.uchicago.edu/site/flashcode/user_support/flash4_ug_4p62/node184.html#SECTION010114000000000000000
Should be used together with [`initial_condition_sedov_self_gravity`](@ref).
"""
function boundary_condition_sedov_self_gravity(u_inner, orientation, direction, x, t,
                                                surface_flux_function,
                                                equations::HyperbolicDiffusionEquations2D)
  u_boundary = initial_condition_sedov_self_gravity(x, t, equations)

  # Calculate boundary flux
  if direction in (2, 4) # u_inner is "left" of boundary, u_boundary is "right" of boundary
    flux = surface_flux_function(u_inner, u_boundary, orientation, equations)
  else # u_boundary is "left" of boundary, u_inner is "right" of boundary
    flux = surface_flux_function(u_boundary, u_inner, orientation, equations)
  end

  return flux
end


# Calculate 1D flux in for a single point
@inline function flux(u, orientation::Integer, equations::HyperbolicDiffusionEquations2D)
  phi, q1, q2 = u
  @unpack inv_Tr = equations

  if orientation == 1
    f1 = -equations.nu*q1
    f2 = -phi * inv_Tr
    f3 = zero(phi)
  else
    f1 = -equations.nu*q2
    f2 = zero(phi)
    f3 = -phi * inv_Tr
  end

  return SVector(f1, f2, f3)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, orientation::Integer, equations::HyperbolicDiffusionEquations2D)
  λ_max = sqrt(equations.nu * equations.inv_Tr)
end


# Calculate 1D flux for a single point in the normal direction
# Note, this directional vector is not normalized
@inline function flux(u, normal_direction::AbstractVector, equations::HyperbolicDiffusionEquations2D)
  phi, q1, q2 = u
  @unpack inv_Tr = equations

  f1 = -equations.nu * (normal_direction[1] * q1 + normal_direction[2] * q2)
  f2 = -phi * inv_Tr * normal_direction[1]
  f3 = -phi * inv_Tr * normal_direction[2]

  return SVector(f1, f2, f3)
end


# Calculate maximum wave speed for local Lax-Friedrichs-type dissipation
@inline function max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector, equations::HyperbolicDiffusionEquations2D)
  λ_max = sqrt(equations.nu * equations.inv_Tr) * norm(normal_direction)
end


# TODO: Could add a `rotate_to_x` and `rotate_from_x` in order to use this numerical surface flux
#       in the FluxRotated functionality
@inline function flux_godunov(u_ll, u_rr, orientation::Integer, equations::HyperbolicDiffusionEquations2D)
  # Obtain left and right fluxes
  phi_ll, p_ll, q_ll = u_ll
  phi_rr, p_rr, q_rr = u_rr
  f_ll = flux(u_ll, orientation, equations)
  f_rr = flux(u_rr, orientation, equations)

  # this is an optimized version of the application of the upwind dissipation matrix:
  #   dissipation = 0.5*R_n*|Λ|*inv(R_n)[[u]]
  λ_max = sqrt(equations.nu * equations.inv_Tr)
  f1 = 1/2 * (f_ll[1] + f_rr[1]) - 1/2 * λ_max * (phi_rr - phi_ll)
  if orientation == 1 # x-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2]) - 1/2 * λ_max * (p_rr - p_ll)
    f3 = 1/2 * (f_ll[3] + f_rr[3])
  else # y-direction
    f2 = 1/2 * (f_ll[2] + f_rr[2])
    f3 = 1/2 * (f_ll[3] + f_rr[3]) - 1/2 * λ_max * (q_rr - q_ll)
  end

  return SVector(f1, f2, f3)
end



@inline have_constant_speed(::HyperbolicDiffusionEquations2D) = Val(true)

@inline function max_abs_speeds(eq::HyperbolicDiffusionEquations2D)
  λ = sqrt(eq.nu * eq.inv_Tr)
  return λ, λ
end


# Convert conservative variables to primitive
@inline cons2prim(u, equations::HyperbolicDiffusionEquations2D) = u

# Convert conservative variables to entropy found in I Do Like CFD, Too, Vol. 1
@inline function cons2entropy(u, equations::HyperbolicDiffusionEquations2D)
  phi, q1, q2 = u
  w1 = phi
  w2 = equations.Lr^2 * q1
  w3 = equations.Lr^2 * q2

  return SVector(w1, w2, w3)
end


# Calculate entropy for a conservative state `u` (here: same as total energy)
@inline entropy(u, equations::HyperbolicDiffusionEquations2D) = energy_total(u, equations)


# Calculate total energy for a conservative state `u`
@inline function energy_total(u, equations::HyperbolicDiffusionEquations2D)
  # energy function as found in equations (2.5.12) in the book "I Do Like CFD, Vol. 1"
  phi, q1, q2 = u
  return 0.5 * (phi^2 + equations.Lr^2 * (q1^2 + q2^2))
end
