@doc raw"""
CompressibleMhdDiffusion3D(gamma, inv_gamma_minus_one,
μ, Pr, eta, kappa,
equations, gradient_variables)

These equations contain the viscous Navier-Stokes equtions coupled to
the magnetic field together with the magnetic diffusion applied
to mass, momenta, magnetic field and total energy together with the advective terms from
the [`IdealGlmMhdEquations3D`](@ref).
Together they constitute the compressible, viscous and resistive MHD equations with energy.

- `gamma`: adiabatic constant,
- `mu`: dynamic viscosity,
- `Pr`: Prandtl number,
- `eta`: magnetic diffusion (resistivity)
- `equations`: instance of the [`IdealGlmMhdEquations3D`](@ref)
- `gradient_variables`: which variables the gradients are taken with respect to.
Defaults to `GradientVariablesPrimitive()`.

Fluid properties such as the dynamic viscosity $\mu$ and magnetic diffusion $\eta$
can be provided in any consistent unit system, e.g.,
[$\mu$] = kg m⁻¹ s⁻¹.

#!!! warning "Experimental code"
#    This code is experimental and may be changed or removed in any future release.
"""


struct CompressibleMhdDiffusion3D{GradientVariables, RealT <: Real, E <: AbstractIdealGlmMhdEquations{3}} <: AbstractCompressibleMhdDiffusion{3, 9}
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
  mu::RealT                  # viscosity
  Pr::RealT                  # Prandtl number
  eta::RealT                 # magnetic diffusion
  kappa::RealT               # thermal diffusivity for Fick's law
  equations_hyperbolic::E    # IdealGlmMhdEquations3D
  gradient_variables::GradientVariables # GradientVariablesPrimitive or GradientVariablesEntropy
end

# default to primitive gradient variables
function CompressibleMhdDiffusion3D(equations::IdealGlmMhdEquations3D;
                                    mu, Prandtl, eta, gradient_variables = GradientVariablesPrimitive())
  gamma = equations.gamma
  inv_gamma_minus_one = equations.inv_gamma_minus_one
  μ, Pr, eta = promote(mu, Prandtl, eta)

  # Under the assumption of constant Prandtl number the thermal conductivity
  # constant is kappa = gamma μ / ((gamma-1) Pr).
  # Important note! Factor of μ is accounted for later in `flux`.
  kappa = gamma * inv_gamma_minus_one / Pr

  CompressibleMhdDiffusion3D{typeof(gradient_variables), typeof(gamma), typeof(equations)}(gamma, inv_gamma_minus_one,
                                                                                           μ, Pr, eta, kappa,
                                                                                           equations, gradient_variables)
end

have_nonconservative_terms(::CompressibleMhdDiffusion3D) = Val(true)
varnames(::typeof(cons2cons), ::CompressibleMhdDiffusion3D) = ("rho", "rho_v1", "rho_v2", "rho_v3", "rho_e", "B1", "B2", "B3", "psi")
varnames(::typeof(cons2prim), ::CompressibleMhdDiffusion3D) = ("rho", "v1", "v2", "v3", "p", "B1", "B2", "B3", "psi")
default_analysis_integrals(::CompressibleMhdDiffusion3D)  = (entropy_timederivative, Val(:l2_divb), Val(:linf_divb))


# TODO: parabolic
# This is the flexibility a user should have to select the different gradient variable types
# varnames(::typeof(cons2prim)   , ::CompressibleMhdDiffusion3D) = ("v1", "v2", "v3", "T")
# varnames(::typeof(cons2entropy), ::CompressibleMhdDiffusion3D) = ("w2", "w3", "w4", "w5")

varnames(variable_mapping, equations_parabolic::CompressibleMhdDiffusion3D) =
  varnames(variable_mapping, equations_parabolic.equations_hyperbolic)

# # we specialize this function to compute gradients of primitive variables instead of
# # conservative variables.
# gradient_variable_transformation(::CompressibleMhdDiffusion3D{GradientVariablesPrimitive}) = cons2prim
# gradient_variable_transformation(::CompressibleMhdDiffusion3D{GradientVariablesEntropy}) = cons2entropy


# Explicit formulas for the diffusive MHD fluxes are available, e.g., in Section 2
# of the paper by Rueda-Ramírez, Hennemann, Hindenlang, Winters, and Gassner
# "An Entropy Stable Nodal Discontinuous Galerkin Method for the resistive
#  MHD Equations. Part II: Subcell Finite Volume Shock Capturing"
function flux(u, gradients, orientation::Integer, equations::CompressibleMhdDiffusion3D)
  # Here, `u` is assumed to be the "transformed" variables specified by `gradient_variable_transformation`.
  rho, v1, v2, v3, E, B1, B2, B3, psi = convert_transformed_to_primitive(u, equations)
  # Here `gradients` is assumed to contain the gradients of the primitive variables (rho, v1, v2, v3, T)
  # either computed directly or reverse engineered from the gradient of the entropy vairables
  # by way of the `convert_gradient_variables` function.
  
  @unpack eta = equations

  # TODO: use primitive gradients
  _, dv1dx, dv2dx, dv3dx, dTdx, dB1dx, dB2dx, dB3dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
  _, dv1dy, dv2dy, dv3dy, dTdy, dB1dy, dB2dy, dB3dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)
  _, dv1dz, dv2dz, dv3dz, dTdz, dB1dz, dB2dz, dB3dz, _ = convert_derivative_to_primitive(u, gradients[3], equations)

  # Components of viscous stress tensor

  # Diagonal parts
  # (4/3 * (v1)_x - 2/3 * ((v2)_y + (v3)_z)
  tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * (dv2dy + dv3dz)
  # (4/3 * (v2)_y - 2/3 * ((v1)_x + (v3)_z)
  tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * (dv1dx + dv3dz)
  # (4/3 * (v3)_z - 2/3 * ((v1)_x + (v2)_y)
  tau_33 = 4.0 / 3.0 * dv3dz - 2.0 / 3.0 * (dv1dx + dv2dy)

  # Off diagonal parts, exploit that stress tensor is symmetric
  # ((v1)_y + (v2)_x)
  tau_12 = dv1dy + dv2dx # = tau_21
  # ((v1)_z + (v3)_x)
  tau_13 = dv1dz + dv3dx # = tau_31
  # ((v2)_z + (v3)_y)
  tau_23 = dv2dz + dv3dy # = tau_32

  # Fick's law q = -kappa * grad(T) = -kappa * grad(p / (R rho))
  # with thermal diffusivity constant kappa = gamma μ R / ((gamma-1) Pr)
  # Note, the gas constant cancels under this formulation, so it is not present
  # in the implementation
  q1 = equations.kappa * dTdx
  q2 = equations.kappa * dTdy
  q3 = equations.kappa * dTdz

  # Constant dynamic viscosity is copied to a variable for readibility.
  # Offers flexibility for dynamic viscosity via Sutherland's law where it depends
  # on temperature and reference values, Ts and Tref such that mu(T)
  mu = equations.mu

  if orientation == 1
    # viscous flux components in the x-direction
    f1 = zero(rho)
    f2 = tau_11 * mu
    f3 = tau_12 * mu
    f4 = tau_13 * mu
    f5 = ( v1 * tau_11 + v2 * tau_12 + v3 * tau_13 + q1 ) * mu + (- B3*dB1dz + B3*dB3dx + B2*dB2dx - B2*dB1dx) * eta
    f6 = eta * dB1dx
    f7 = eta * dB2dx
    f8 = eta * dB3dx
    f9 = zero(rho)

    return SVector(f1, f2, f3, f4, f5, f6, f7, f8, f9)
  elseif orientation == 2
    # viscous flux components in the y-direction
    # Note, symmetry is exploited for tau_12 = tau_21
    g1 = zero(rho)
    g2 = tau_12 * mu # tau_21 * mu
    g3 = tau_22 * mu
    g4 = tau_23 * mu
    g5 = ( v1 * tau_12 + v2 * tau_22 + v3 * tau_23 + q2 ) * mu + (- B1*dB2dx + B1*dB1dy + B3*dB3dy - B3*dB2dy) * eta
    g6 = eta * dB1dy
    g7 = eta * dB2dy
    g8 = eta * dB3dy
    g9 = zero(rho)

    return SVector(g1, g2, g3, g4, g5, g6, g7, g8, g9)
  else # if orientation == 3
    # viscous flux components in the z-direction
    # Note, symmetry is exploited for tau_13 = tau_31, tau_23 = tau_32
    h1 = zero(rho)
    h2 = tau_13 * mu # tau_31 * mu
    h3 = tau_23 * mu # tau_32 * mu
    h4 = tau_33 * mu
    h5 = ( v1 * tau_13 + v2 * tau_23 + v3 * tau_33 + q3 ) * mu + (- B2*dB3dy + B2*dB2dz + B1*dB1dz - B1*dB3dz) * eta
    h6 = eta * dB1dz
    h7 = eta * dB2dz
    h8 = eta * dB3dz
    h9 = zero(rho)

    return SVector(h1, h2, h3, h4, h5, h6, h7, h8, h9)
  end
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleMhdDiffusion3D)
  rho, rho_v1, rho_v2, rho_v3, T, B1, B2, B3, psi = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  v3 = rho_v3 / rho
  T  = temperature(u, equations)

  return SVector(rho, v1, v2, v3, T, B1, B2, B3, psi)
end


# the `flux` function takes in transformed variables `u` which depend on the type of the gradient variables.
# For CNS, it is simplest to formulate the viscous terms in primitive variables, so we transform the transformed
# variables into primitive variables.
@inline function convert_transformed_to_primitive(u_transformed, equations::CompressibleMhdDiffusion3D{GradientVariablesPrimitive})
  return u_transformed
end


# Takes the solution values `u` and gradient of the entropy variables (w_2, w_3, w_4, w_5) and
# reverse engineers the gradients to be terms of the primitive variables (v1, v2, v3, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
# Note, the first component of `gradient_entropy_vars` contains gradient(rho) which is unused.
# TODO: parabolic; entropy stable viscous terms
@inline function convert_derivative_to_primitive(u, gradient, ::CompressibleMhdDiffusion3D{GradientVariablesPrimitive})
  return gradient
end


@inline function temperature(u, equations::CompressibleMhdDiffusion3D)
  rho, rho_v1, rho_v2, rho_v3, rho_e, _ = u

  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2 + rho_v3^2) / rho)
  T = p / rho
  return T
end


@inline function enstrophy(u, gradients, equations::CompressibleMhdDiffusion3D)
  # Enstrophy is 0.5 rho ω⋅ω where ω = ∇ × v

  omega = vorticity(u, gradients, equations)
  return 0.5 * u[1] * (omega[1]^2 + omega[2]^2 + omega[3]^2)
end


@inline function vorticity(u, gradients, equations::CompressibleMhdDiffusion3D)
  # Ensure that we have velocity `gradients` by way of the `convert_gradient_variables` function.
  _, dv1dx, dv2dx, dv3dx, _ = convert_derivative_to_primitive(u, gradients[1], equations)
  _, dv1dy, dv2dy, dv3dy, _ = convert_derivative_to_primitive(u, gradients[2], equations)
  _, dv1dz, dv2dz, dv3dz, _ = convert_derivative_to_primitive(u, gradients[3], equations)

  return SVector(dv3dy - dv2dz , dv1dz - dv3dx , dv2dx - dv1dy)
end

