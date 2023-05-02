@doc raw"""
    CompressibleNavierStokesDiffusion2D(equations; mu, Pr,
                                        gradient_variables=GradientVariablesPrimitive())

Contains the diffusion (i.e. parabolic) terms applied
to mass, momenta, and total energy together with the advective terms from
the [`CompressibleEulerEquations2D`](@ref).

- `equations`: instance of the [`CompressibleEulerEquations2D`](@ref)
- `mu`: dynamic viscosity,
- `Pr`: Prandtl number,
- `gradient_variables`: which variables the gradients are taken with respect to.
                        Defaults to `GradientVariablesPrimitive()`.

Fluid properties such as the dynamic viscosity ``\mu`` can be provided in any consistent unit system, e.g.,
[``\mu``] = kg m⁻¹ s⁻¹.

The particular form of the compressible Navier-Stokes implemented is
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho \mathbf{v} \\ \rho e
\end{pmatrix}
+
\nabla \cdot
\begin{pmatrix}
 \rho \mathbf{v} \\ \rho \mathbf{v}\mathbf{v}^T + p \underline{I} \\ (\rho e + p) \mathbf{v}
\end{pmatrix}
=
\nabla \cdot
\begin{pmatrix}
0 \\ \underline{\tau} \\ \underline{\tau}\mathbf{v} - \nabla q
\end{pmatrix}
```
where the system is closed with the ideal gas assumption giving
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho (v_1^2+v_2^2) \right)
```
as the pressure. The value of the adiabatic constant `gamma` is taken from the [`CompressibleEulerEquations2D`](@ref).
The terms on the right hand side of the system above
are built from the viscous stress tensor
```math
\underline{\tau} = \mu \left(\nabla\mathbf{v} + \left(\nabla\mathbf{v}\right)^T\right) - \frac{2}{3} \mu \left(\nabla\cdot\mathbf{v}\right)\underline{I}
```
where ``\underline{I}`` is the ``2\times 2`` identity matrix and the heat flux is
```math
\nabla q = -\kappa\nabla\left(T\right),\quad T = \frac{p}{R\rho}
```
where ``T`` is the temperature and ``\kappa`` is the thermal conductivity for Fick's law.
Under the assumption that the gas has a constant Prandtl number,
the thermal conductivity is
```math
\kappa = \frac{\gamma \mu R}{(\gamma - 1)\textrm{Pr}}.
```
From this combination of temperature ``T`` and thermal conductivity ``\kappa`` we see
that the gas constant `R` cancels and the heat flux becomes
```math
\nabla q = -\kappa\nabla\left(T\right) = -\frac{\gamma \mu}{(\gamma - 1)\textrm{Pr}}\nabla\left(\frac{p}{\rho}\right)
```
which is the form implemented below in the [`flux`](@ref) function.

In two spatial dimensions we require gradients for three quantities, e.g.,
primitive quantities
```math
\nabla v_1,\, \nabla v_2,\, \nabla T
```
or the entropy variables
```math
\nabla w_2,\, \nabla w_3,\, \nabla w_4
```
where
```math
w_2 = \frac{\rho v_1}{p},\, w_3 = \frac{\rho v_2}{p},\, w_4 = -\frac{\rho}{p}
```

#!!! warning "Experimental code"
#    This code is experimental and may be changed or removed in any future release.
"""
struct CompressibleNavierStokesDiffusion2D{GradientVariables, RealT <: Real, E <: AbstractCompressibleEulerEquations{2}} <: AbstractCompressibleNavierStokesDiffusion{2, 4}
  # TODO: parabolic
  # 1) For now save gamma and inv(gamma-1) again, but could potentially reuse them from the Euler equations
  # 2) Add NGRADS as a type parameter here and in AbstractEquationsParabolic, add `ngradients(...)` accessor function
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

  mu::RealT                  # viscosity
  Pr::RealT                  # Prandtl number
  kappa::RealT               # thermal diffusivity for Fick's law

  equations_hyperbolic::E    # CompressibleEulerEquations2D
  gradient_variables::GradientVariables # GradientVariablesPrimitive or GradientVariablesEntropy
end

"""
#!!! warning "Experimental code"
#    This code is experimental and may be changed or removed in any future release.

`GradientVariablesPrimitive` and `GradientVariablesEntropy` are gradient variable type parameters
for `CompressibleNavierStokesDiffusion2D`. By default, the gradient variables are set to be
`GradientVariablesPrimitive`. Specifying `GradientVariablesEntropy` instead uses the entropy variable
formulation from
- Hughes, Mallet, Franca (1986)
  A new finite element formulation for computational fluid dynamics: I. Symmetric forms of the
  compressible Euler and Navier-Stokes equations and the second law of thermodynamics.
  [https://doi.org/10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)

Under `GradientVariablesEntropy`, the Navier-Stokes discretization is provably entropy stable.
"""
struct GradientVariablesPrimitive end
struct GradientVariablesEntropy end

# default to primitive gradient variables
function CompressibleNavierStokesDiffusion2D(equations::CompressibleEulerEquations2D;
                                             mu, Prandtl,
                                             gradient_variables = GradientVariablesPrimitive())
  gamma = equations.gamma
  inv_gamma_minus_one = equations.inv_gamma_minus_one
  μ, Pr = promote(mu, Prandtl)

  # Under the assumption of constant Prandtl number the thermal conductivity
  # constant is kappa = gamma μ / ((gamma-1) Pr).
  # Important note! Factor of μ is accounted for later in `flux`.
  kappa = gamma * inv_gamma_minus_one / Pr

  CompressibleNavierStokesDiffusion2D{typeof(gradient_variables), typeof(gamma), typeof(equations)}(gamma, inv_gamma_minus_one,
                                                                                                    μ, Pr, kappa,
                                                                                                    equations, gradient_variables)
end

# TODO: parabolic
# This is the flexibility a user should have to select the different gradient variable types
# varnames(::typeof(cons2prim)   , ::CompressibleNavierStokesDiffusion2D) = ("v1", "v2", "T")
# varnames(::typeof(cons2entropy), ::CompressibleNavierStokesDiffusion2D) = ("w2", "w3", "w4")

varnames(variable_mapping, equations_parabolic::CompressibleNavierStokesDiffusion2D) =
  varnames(variable_mapping, equations_parabolic.equations_hyperbolic)

# we specialize this function to compute gradients of primitive variables instead of
# conservative variables.
gradient_variable_transformation(::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive}) = cons2prim
gradient_variable_transformation(::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy}) = cons2entropy


# Explicit formulas for the diffusive Navier-Stokes fluxes are available, e.g., in Section 2
# of the paper by Rueda-Ramírez, Hennemann, Hindenlang, Winters, and Gassner
# "An Entropy Stable Nodal Discontinuous Galerkin Method for the resistive
#  MHD Equations. Part II: Subcell Finite Volume Shock Capturing"
# where one sets the magnetic field components equal to 0.
function flux(u, gradients, orientation::Integer, equations::CompressibleNavierStokesDiffusion2D)
  # Here, `u` is assumed to be the "transformed" variables specified by `gradient_variable_transformation`.
  rho, v1, v2, _ = convert_transformed_to_primitive(u, equations)
  # Here `gradients` is assumed to contain the gradients of the primitive variables (rho, v1, v2, T)
  # either computed directly or reverse engineered from the gradient of the entropy variables
  # by way of the `convert_gradient_variables` function.
  _, dv1dx, dv2dx, dTdx = convert_derivative_to_primitive(u, gradients[1], equations)
  _, dv1dy, dv2dy, dTdy = convert_derivative_to_primitive(u, gradients[2], equations)

  # Components of viscous stress tensor

  # (4/3 * (v1)_x - 2/3 * (v2)_y)
  tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
  # ((v1)_y + (v2)_x)
  # stress tensor is symmetric
  tau_12 = dv1dy + dv2dx # = tau_21
  # (4/3 * (v2)_y - 2/3 * (v1)_x)
  tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

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
    f4 = ( v1 * tau_11 + v2 * tau_12 + q1 ) * mu

    return SVector(f1, f2, f3, f4)
  else # if orientation == 2
    # viscous flux components in the y-direction
    # Note, symmetry is exploited for tau_12 = tau_21
    g1 = zero(rho)
    g2 = tau_12 * mu # tau_21 * mu
    g3 = tau_22 * mu
    g4 = ( v1 * tau_12 + v2 * tau_22 + q2 ) * mu

    return SVector(g1, g2, g3, g4)
  end
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleNavierStokesDiffusion2D)
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  return SVector(rho, v1, v2, T)
end

# Convert conservative variables to entropy
# TODO: parabolic. We can improve efficiency by not computing w_1, which involves logarithms
# This can be done by specializing `cons2entropy` and `entropy2cons` to `CompressibleNavierStokesDiffusion2D`,
# but this may be confusing to new users.
cons2entropy(u, equations::CompressibleNavierStokesDiffusion2D) = cons2entropy(u, equations.equations_hyperbolic)
entropy2cons(w, equations::CompressibleNavierStokesDiffusion2D) = entropy2cons(w, equations.equations_hyperbolic)

# the `flux` function takes in transformed variables `u` which depend on the type of the gradient variables.
# For CNS, it is simplest to formulate the viscous terms in primitive variables, so we transform the transformed
# variables into primitive variables.
@inline function convert_transformed_to_primitive(u_transformed, equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  return u_transformed
end

# TODO: parabolic. Make this more efficient!
@inline function convert_transformed_to_primitive(u_transformed, equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  # note: this uses CompressibleNavierStokesDiffusion2D versions of cons2prim and entropy2cons
  return cons2prim(entropy2cons(u_transformed, equations), equations)
end


# Takes the solution values `u` and gradient of the entropy variables (w_2, w_3, w_4) and
# reverse engineers the gradients to be terms of the primitive variables (v1, v2, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
# Note, the first component of `gradient_entropy_vars` contains gradient(rho) which is unused.
# TODO: parabolic; entropy stable viscous terms
@inline function convert_derivative_to_primitive(u, gradient, ::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  return gradient
end

# the first argument is always the "transformed" variables.
@inline function convert_derivative_to_primitive(w, gradient_entropy_vars,
                                                 equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})

  # TODO: parabolic. This is inefficient to pass in transformed variables but then transform them back.
  # We can fix this if we directly compute v1, v2, T from the entropy variables
  u = entropy2cons(w, equations) # calls a "modified" entropy2cons defined for CompressibleNavierStokesDiffusion2D
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  return SVector(gradient_entropy_vars[1],
                 T * (gradient_entropy_vars[2] + v1 * gradient_entropy_vars[4]), # grad(u) = T*(grad(w_2)+v1*grad(w_4))
                 T * (gradient_entropy_vars[3] + v2 * gradient_entropy_vars[4]), # grad(v) = T*(grad(w_3)+v2*grad(w_4))
                 T * T * gradient_entropy_vars[4]                                # grad(T) = T^2*grad(w_4))
                )
end


# This routine is required because `prim2cons` is called in `initial_condition`, which
# is called with `equations::CompressibleEulerEquations2D`. This means it is inconsistent
# with `cons2prim(..., ::CompressibleNavierStokesDiffusion2D)` as defined above.
# TODO: parabolic. Is there a way to clean this up?
@inline prim2cons(u, equations::CompressibleNavierStokesDiffusion2D) =
    prim2cons(u, equations.equations_hyperbolic)


@inline function temperature(u, equations::CompressibleNavierStokesDiffusion2D)
  rho, rho_v1, rho_v2, rho_e = u

  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
  T = p / rho
  return T
end

# TODO: can we generalize this to MHD?
"""
    struct BoundaryConditionNavierStokesWall

Creates a wall-type boundary conditions for the compressible Navier-Stokes equations.
The fields `boundary_condition_velocity` and `boundary_condition_heat_flux` are intended
to be boundary condition types such as the `NoSlip` velocity boundary condition and the
`Adiabatic` or `Isothermal` heat boundary condition.

!!! warning "Experimental feature"
    This is an experimental feature and may change in future releases.
"""
struct BoundaryConditionNavierStokesWall{V, H}
  boundary_condition_velocity::V
  boundary_condition_heat_flux::H
end

"""
    struct NoSlip

Use to create a no-slip boundary condition with `BoundaryConditionNavierStokesWall`. The field `boundary_value_function`
should be a function with signature `boundary_value_function(x, t, equations)`
and should return a `SVector{NDIMS}` whose entries are the velocity vector at a
point `x` and time `t`.
"""
struct NoSlip{F}
  boundary_value_function::F # value of the velocity vector on the boundary
end

"""
    struct Isothermal

Used to create a no-slip boundary condition with [`BoundaryConditionNavierStokesWall`](@ref).
The field `boundary_value_function` should be a function with signature
`boundary_value_function(x, t, equations)` and return a scalar value for the
temperature at point `x` and time `t`.
"""
struct Isothermal{F}
  boundary_value_function::F # value of the temperature on the boundary
end

"""
    struct Adiabatic

Used to create a no-slip boundary condition with [`BoundaryConditionNavierStokesWall`](@ref).
The field `boundary_value_normal_flux_function` should be a function with signature
`boundary_value_normal_flux_function(x, t, equations)` and return a scalar value for the
normal heat flux at point `x` and time `t`.
"""
struct Adiabatic{F}
  boundary_value_normal_flux_function::F # scaled heat flux 1/T * kappa * dT/dn
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Adiabatic})(flux_inner, u_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Gradient,
                                                                                           equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  return SVector(u_inner[1], v1, v2, u_inner[4])
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Adiabatic})(flux_inner, u_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Divergence,
                                                                                           equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  # rho, v1, v2, _ = u_inner
  normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x, t, equations)
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  _, tau_1n, tau_2n, _ = flux_inner # extract fluxes for 2nd and 3rd equations
  normal_energy_flux = v1 * tau_1n + v2 * tau_2n + normal_heat_flux
  return SVector(flux_inner[1], flux_inner[2], flux_inner[3], normal_energy_flux)
end


@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Isothermal})(flux_inner, u_inner, normal::AbstractVector,
                                                                                            x, t, operator_type::Gradient,
                                                                                            equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t, equations)
  return SVector(u_inner[1], v1, v2, T)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Isothermal})(flux_inner, u_inner, normal::AbstractVector,
                                                                                            x, t, operator_type::Divergence,
                                                                                            equations::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  return flux_inner
end

# specialized BC impositions for GradientVariablesEntropy.

# This should return a SVector containing the boundary values of entropy variables.
# Here, `w_inner` are the transformed variables (e.g., entropy variables).
#
# Taken from "Entropy stable modal discontinuous Galerkin schemes and wall boundary conditions
#             for the compressible Navier-Stokes equations" by Chan, Lin, Warburton 2022.
# DOI: 10.1016/j.jcp.2021.110723
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Adiabatic})(flux_inner, w_inner, normal::AbstractVector,
                                                                                                x, t, operator_type::Gradient,
                                                                                                equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  negative_rho_inv_p = w_inner[4] # w_4 = -rho / p
  return SVector(w_inner[1], -v1 * negative_rho_inv_p, -v2 * negative_rho_inv_p, negative_rho_inv_p)
end

# this is actually identical to the specialization for GradientVariablesPrimitive, but included for completeness.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Adiabatic})(flux_inner, w_inner, normal::AbstractVector,
                                                                                                x, t, operator_type::Divergence,
                                                                                                equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x, t, equations)
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  _, tau_1n, tau_2n, _ = flux_inner # extract fluxes for 2nd and 3rd equations
  normal_energy_flux = v1 * tau_1n + v2 * tau_2n + normal_heat_flux
  return SVector(flux_inner[1], flux_inner[2], flux_inner[3], normal_energy_flux)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Isothermal})(flux_inner, w_inner, normal::AbstractVector,
                                                                                                 x, t, operator_type::Gradient,
                                                                                                 equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t, equations)

  # the entropy variables w2 = rho * v1 / p = v1 / T = -v1 * w4. Similarly for w3
  w4 = -1 / T
  return SVector(w_inner[1], -v1 * w4, -v2 * w4, w4)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Isothermal})(flux_inner, w_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Divergence,
                                                                                           equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  return SVector(flux_inner[1], flux_inner[2], flux_inner[3], flux_inner[4])
end
