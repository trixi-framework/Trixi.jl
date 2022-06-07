@doc raw"""
    CompressibleNaiverStokes2D(gamma,
                               Re,
                               Pr,
                               Ma_inf,
                               kappa,
                               equations)

`CompressibleNaiverStokes2D` represents the diffusion (i.e. parabolic) terms applied
to mass, momenta, and total energy together with the advective from
the `CompressibleEulerEquations2D`.

gamma: adiabatic constant,
Re: Reynolds number,
Pr: Prandtl number,
Ma_inf: free-stream Mach number
kappa: thermal diffusivity for Fick's law

For the particular scaling the vicosity is set internally to be μ = 1/Re.
Further, the nondimensionalization takes the density-temperature-sound speed as
the principle quantities such that
rho_inf = 1.0
T_ref = 1.0
c_inf = 1.0
p_inf = 1.0 / gamma
u_inf = Ma_inf
R = 1.0 / gamma

Other normalization strategies exist, see the reference below for details.
- Marc Montagnac (2013)
  Variable Normalization (nondimensionalization and scaling) for Navier-Stokes
  equations: a practical guide
  [CERFACS Technical report](https://www.cerfacs.fr/~montagna/TR-CFD-13-77.pdf)
The scaling used herein is Section 4.5 of the reference.

In two spatial dimensions we require gradients for three quantities, e.g.,
primitive quantities
  grad(u), grad(v), and grad(T)
or the entropy variables
  grad(w_2), grad(w_3), grad(w_4)
where
  w_2 = rho v_1 / p, w_3 = rho v_2 / p, w_4 = -rho / p
"""
# TODO:
# 1) For now I save gamma and inv(gamma-1) again, but we could potentially reuse them from
#    the Euler equations
# 2) Add more here and probably some equations
struct CompressibleNaiverStokes2D{RealT<:Real, E} <: AbstractCompressibleNavierStokesEquations{2, 3}
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
  Re::RealT                  # Reynolds number
  Pr::RealT                  # Prandtl number
  Ma_inf::RealT              # free-stream Mach number
  kappa::RealT               # thermal diffusivity for Fick's law

  p_inf::RealT               # free-stream pressure
  u_inf::RealT               # free-stream velocity
  R::RealT                   # gas constant (depends on nondimensional scaling!)

  equations::E               # CompressibleEulerEquations2D
end

function CompressibleNaiverStokes2D(gamma, Reynolds, Prandtl, Mach_frestream, kappa, equations)
  γ, inv_gamma_minus_one = promote(gamma, inv(gamma - 1))

  # From the nondimensionalization discussed above set the remaining free-stream
  # quantities
  p_inf = 1.0 / γ
  u_inf = Mach_frestream
  R     = 1.0 / γ
  CompressibleNaiverStokes2D{typeof(γ),typeof(equations)}(γ, inv_gamma_minus_one,
                                                          Reynolds, Prandtl, Mach_freestream,
                                                          kappa, p_inf, u_inf, R,
                                                          equations)
end

# I was not sure what to do here to allow flexibility of selecting primitive or entropy
# grandient variables
varnames(variable_mapping, equations_parabolic::CompressibleNaiverStokes2D) =
  varnames(variable_mapping, equations_parabolic.equations)


# no orientation specified since the flux is vector-valued
# This form of the difussive Navier-Stokes fluxes is taken from Section 2
# of the paper by Svärd, Carpenter and Nordström
# "A stable high-order finite difference scheme for the compressible Navier–Stokes
#  equations, far-field boundary conditions"
# Note, could be generalized to use Sutherland's law to get the molecular and thermal
# diffusivity
function flux(u, grad_u, equations::CompressibleNaiverStokes2D)
  # Here grad_u is assumed to contain the gradients of the primitive variables (v1,v2,T)
  # either computed directly or reverse engineered from the gradient of the entropy vairables
  # by way of the `convert_gradient_variables` function
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  # I was not sure what shape this array has or or if it was a tuple
  # or how to properly "unpack" it. So I just guessed...
  dv1dx, dv1dy, dv2dx, dv2dy, dTdx, dTdy = grad_u

  # Components of viscous stress tensor

  # (4/3*(v1)_x - 2/3*(v2)_y)
  tau_xx = ( 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy )
  # ((v1)_y + (v2)_x)
  tau_xy = ( dv1dy + dv2dx )
  tau_yx = tau_xy
  # (4/3*(v2)_y - 2/3*(v1)_x)
  tau_yy = ( 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx )

  # Fick's law q = -kappa*grad(T); constant is kappa*gamma/(Pr*(gamma-1))
  # Important note! Due to nondimensional scaling R = 1/gamma, so the
  # temperature T in the gradient computation already contains a factor of gamma
  q1 = ( equations.kappa * equations.inv_gamma_minus_one * dTdx ) / equations.Pr
  q2 = ( equations.kappa * equations.inv_gamma_minus_one * dTdy ) / equations.Pr

  # viscous flux components in the x-direction
  f1 = zero(rho)
  f2 = tau_xx / equations.Re
  f3 = tau_xy / equations.Re
  f4 = ( v1 * tau_xx + v2 * tau_xy + q1) / equations.Re

  # viscous flux components in y-direction
  g1 = zero(rho)
  g2 = tau_yx / equations.Re
  g3 = tau_yy / equations.Re
  g4 = ( v1 * tau_yx + v2 * tau_yy + q2) / equations.Re

  # TODO: I was not sure how to return this properly. Right now it is a vector of vectors
  return SVector( SVector(f1, f2, f3, f4) , SVector(g1, g2, g3, g4) )
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleNaiverStokes2D)
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  return SVector(v1, v2, T)
end


# Convert conservative variables to entropy
@inline function cons2entropy(u, equations::CompressibleNaiverStokes2D)
  rho, rho_v1, rho_v2, rho_e = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v_square = v1^2 + v2^2
  p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square)

  rho_p = rho / p

  w2 = rho_p * v1
  w3 = rho_p * v2
  w4 = -rho_p

  return SVector(w2, w3, w4)
end


@inline function convert_gradient_variables(u, grad_entropy_vars, equations::CompressibleNaiverStokes2D)
# Takes the solution values `u` and gradient of the variables (w_2, w_3, w_4) and
# reverse engineers the gradients to be terms of the primitive vairables (u, v, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  return SVector(equations.R * T * (grad_entropy_vars(1) + v1 * grad_entropy_vars(3)), # grad(u) = R*T*(grad(w_2)+v1*grad(w_4))
                 equations.R * T * (grad_entropy_vars(2) + v2 * grad_entropy_vars(3)), # grad(v) = R*T*(grad(w_3)+v2*grad(w_4))
                 equations.R * T * T * grad_entropy_vars(3)                            # grad(T) = R*T^2*grad(w_4))
                )
end


@inline function temperature(u, equations::CompressibleNaiverStokes2D)
  rho, rho_v1, rho_v2, rho_e = u

  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
  T = p / (equations.R * rho)
  return T
end


#
#  All this boundary conditions stuff I did not touch
#

# TODO: should this remain in the equations file, be moved to solvers, or live in the elixir?
# The penalization depends on the solver, but also depends explicitly on physical parameters,
# and would probably need to be specialized for every different equation.
# function penalty(u_outer, u_inner, inv_h, equations::LaplaceDiffusion2D, dg::ViscousFormulationLocalDG)
#   return dg.penalty_parameter * (u_outer - u_inner) * equations.diffusivity * inv_h
# end

# # Dirichlet-type boundary condition for use with a parabolic solver in weak form
# @inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, normal::AbstractVector,
#                                                                   x, t, operator_type::Gradient,
#                                                                   equations::LaplaceDiffusion2D)
#   return boundary_condition.boundary_value_function(x, t, equations)
# end

# @inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, normal::AbstractVector,
#                                                                   x, t, operator_type::Divergence,
#                                                                   equations::LaplaceDiffusion2D)
#   return u_inner
# end

# @inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, normal::AbstractVector,
#                                                                 x, t, operator_type::Divergence,
#                                                                 equations::LaplaceDiffusion2D)
#   return boundary_condition.boundary_normal_flux_function(x, t, equations)
# end

# @inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, normal::AbstractVector,
#                                                                 x, t, operator_type::Gradient,
#                                                                 equations::LaplaceDiffusion2D)
#   return flux_inner
# end
