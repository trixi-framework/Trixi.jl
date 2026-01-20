# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    ViscoResistiveMhd2D(gamma, inv_gamma_minus_one,
                               μ, Pr, eta, kappa,
                               equations, gradient_variables)

These equations contain the viscous Navier-Stokes equations coupled to
the magnetic field together with the magnetic diffusion applied
to mass, momenta, magnetic field and total energy together with the advective terms from
the [`IdealGlmMhdEquations2D`](@ref).
Together they constitute the compressible, viscous and resistive MHD equations with energy.

- `gamma`: adiabatic constant,
- `mu`: dynamic viscosity,
- `Pr`: Prandtl number,
- `eta`: magnetic diffusion (resistivity)
- `equations`: instance of the [`IdealGlmMhdEquations2D`](@ref)
- `gradient_variables`: which variables the gradients are taken with respect to.
Defaults to `GradientVariablesPrimitive()`.

Fluid properties such as the dynamic viscosity $\mu$ and magnetic diffusion $\eta$
can be provided in any consistent unit system, e.g.,
[$\mu$] = kg m⁻¹ s⁻¹.

The equations given here are the visco-resistive part of the MHD equations
in conservation form:
```math
\overleftrightarrow{f}^\mathrm{\mu\eta} =
\begin{pmatrix}
\overrightarrow{0} \\
\underline{\tau} \\
\underline{\tau}\overrightarrow{v} - \overrightarrow{\nabla}q -
 \eta\left( (\overrightarrow{\nabla}\times\overrightarrow{B})\times\overrightarrow{B} \right) \\
\eta \left( (\overrightarrow{\nabla}\overrightarrow{B})^\mathrm{T} - \overrightarrow{\nabla}\overrightarrow{B} \right) \\
\overrightarrow{0}
\end{pmatrix},
```
where `\tau` is the viscous stress tensor and `q = \kappa \overrightarrow{\nabla} T`.
For the induction term we have the usual Laplace operator on the magnetic field
but we also include terms with `div(B)`.
Divergence cleaning is done using the `\psi` field.

For more details see e.g. arXiv:2012.12040.
"""
struct ViscoResistiveMhd2D{GradientVariables, RealT <: Real,
                           E <: AbstractIdealGlmMhdEquations{2}} <:
       AbstractViscoResistiveMhd{2, 9}
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
    mu::RealT                  # viscosity
    Pr::RealT                  # Prandtl number
    eta::RealT                 # magnetic diffusion
    kappa::RealT               # thermal diffusivity for Fick's law
    equations_hyperbolic::E    # IdealGlmMhdEquations2D
    gradient_variables::GradientVariables # GradientVariablesPrimitive or GradientVariablesEntropy
end

# default to primitive gradient variables
function ViscoResistiveMhd2D(equations::IdealGlmMhdEquations2D;
                             mu, Prandtl, eta,
                             gradient_variables = GradientVariablesPrimitive())
    gamma = equations.gamma
    inv_gamma_minus_one = equations.inv_gamma_minus_one
    μ, Pr, eta = promote(mu, Prandtl, eta)

    # Under the assumption of constant Prandtl number the thermal conductivity
    # constant is kappa = gamma μ / ((gamma-1) Pr).
    # Important note! Factor of μ is accounted for later in `flux`.
    kappa = gamma * inv_gamma_minus_one / Pr

    ViscoResistiveMhd2D{typeof(gradient_variables), typeof(gamma), typeof(equations)
                        }(gamma, inv_gamma_minus_one,
                          μ, Pr, eta, kappa,
                          equations, gradient_variables)
end

# Explicit formulas for the diffusive MHD fluxes are available, e.g., in Section 2
# of the paper by Rueda-Ramírez, Hennemann, Hindenlang, Winters, and Gassner
# "An Entropy Stable Nodal Discontinuous Galerkin Method for the resistive
#  MHD Equations. Part II: Subcell Finite Volume Shock Capturing"
function flux(u, gradients, orientation::Integer, equations::ViscoResistiveMhd2D)
    # Here, `u` is assumed to be the "transformed" variables specified by `gradient_variable_transformation`.
    rho, v1, v2, v3, E, B1, B2, B3, psi = convert_transformed_to_primitive(u, equations)
    # Here `gradients` is assumed to contain the gradients of the primitive variables (rho, v1, v2, v3, T)
    # either computed directly or reverse engineered from the gradient of the entropy variables
    # by way of the `convert_gradient_variables` function.

    @unpack eta = equations

    _, dv1dx, dv2dx, dv3dx, dTdx, dB1dx, dB2dx, dB3dx, _ = convert_derivative_to_primitive(u,
                                                                                           gradients[1],
                                                                                           equations)
    _, dv1dy, dv2dy, dv3dy, dTdy, dB1dy, dB2dy, dB3dy, _ = convert_derivative_to_primitive(u,
                                                                                           gradients[2],
                                                                                           equations)

    # Components of viscous stress tensor

    # Diagonal parts
    tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
    tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

    # Off diagonal parts, exploit that stress tensor is symmetric
    # ((v1)_y + (v2)_x)
    tau_12 = dv1dy + dv2dx # = tau_21
    # ((v1)_z + (v3)_x)
    tau_13 = dv3dx # = tau_31
    # ((v2)_z + (v3)_y)
    tau_23 = dv3dy # = tau_32

    # Fick's law q = -kappa * grad(T) = -kappa * grad(p / (R rho))
    # with thermal diffusivity constant kappa = gamma μ R / ((gamma-1) Pr)
    # Note, the gas constant cancels under this formulation, so it is not present
    # in the implementation
    q1 = equations.kappa * dTdx
    q2 = equations.kappa * dTdy

    # Constant dynamic viscosity is copied to a variable for readability.
    # Offers flexibility for dynamic viscosity via Sutherland's law where it depends
    # on temperature and reference values, Ts and Tref such that mu(T)
    mu = equations.mu

    if orientation == 1
        # viscous flux components in the x-direction
        f1 = zero(rho)
        f2 = tau_11 * mu
        f3 = tau_12 * mu
        f4 = tau_13 * mu
        # TODO: Check sign for magnetic part
        f5 = (v1 * tau_11 + v2 * tau_12 + v3 * tau_13 + q1) * mu +
             (B2 * (dB2dx - dB1dy) + B3 * dB3dx) * eta
        f6 = zero(rho)
        f7 = eta * (dB2dx - dB1dy)
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
        # TODO: Check sign for magnetic part
        g5 = (v1 * tau_12 + v2 * tau_22 + v3 * tau_23 + q2) * mu +
             (B1 * (dB1dy - dB2dx) + B3 * dB3dy) * eta
        g6 = eta * (dB1dy - dB2dx)
        g7 = zero(rho)
        g8 = eta * dB3dy
        g9 = zero(rho)

        return SVector(g1, g2, g3, g4, g5, g6, g7, g8, g9)
    end
end

# the `flux` function takes in transformed variables `u` which depend on the type of the gradient variables.
# For CNS, it is simplest to formulate the viscous terms in primitive variables, so we transform the transformed
# variables into primitive variables.
@inline function convert_transformed_to_primitive(u_transformed,
                                                  equations::ViscoResistiveMhd2D{
                                                                                 GradientVariablesPrimitive
                                                                                 })
    return u_transformed
end

# Takes the solution values `u` and gradient of the entropy variables (w_2, w_3, w_4) and
# reverse engineers the gradients to be terms of the primitive variables (v1, v2, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
# Note, the first component of `gradient_entropy_vars` contains gradient(rho) which is unused.
# TODO: parabolic; entropy stable viscous terms
@inline function convert_derivative_to_primitive(u, gradient,
                                                 ::ViscoResistiveMhd2D{
                                                                       GradientVariablesPrimitive
                                                                       })
    return gradient
end

# Calculate the magnetic energy for a conservative state `cons'.
@inline function energy_magnetic_mhd(cons, ::ViscoResistiveMhd2D)
    return 0.5 * (cons[5]^2 + cons[6]^2)
end
end # @muladd
