# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    CompressibleNavierStokesDiffusion1D(equations; mu, Pr,
                                        gradient_variables=GradientVariablesPrimitive())

Contains the diffusion (i.e. parabolic) terms applied
to mass, momenta, and total energy together with the advective terms from
the [`CompressibleEulerEquations1D`](@ref).

- `equations`: instance of the [`CompressibleEulerEquations1D`](@ref)
- `mu`: dynamic viscosity,
- `Pr`: Prandtl number,
- `gradient_variables`: which variables the gradients are taken with respect to.
                        Defaults to `GradientVariablesPrimitive()`.

Fluid properties such as the dynamic viscosity ``\mu`` can be provided in any consistent unit system, e.g.,
[``\mu``] = kg m⁻¹ s⁻¹.
The viscosity ``\mu`` may be a constant or a function of the current state, e.g., 
depending on temperature (Sutherland's law): ``\mu = \mu(T)``.
In the latter case, the function `mu` needs to have the signature `mu(u, equations)`.

The particular form of the compressible Navier-Stokes implemented is
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
\rho \\ \rho v \\ \rho e
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
 \rho v \\ \rho v^2 + p \\ (\rho e + p) v
\end{pmatrix}
=
\frac{\partial}{\partial x}
\begin{pmatrix}
0 \\ \tau \\ \tau v - q
\end{pmatrix}
```
where the system is closed with the ideal gas assumption giving
```math
p = (\gamma - 1) \left( \rho e - \frac{1}{2} \rho v^2 \right)
```
as the pressure. The value of the adiabatic constant `gamma` is taken from the [`CompressibleEulerEquations1D`](@ref).
The terms on the right hand side of the system above
are built from the viscous stress
```math
\tau = \mu \frac{\partial}{\partial x} v
```
where the heat flux is
```math
q = -\kappa \frac{\partial}{\partial x} \left(T\right),\quad T = \frac{p}{R\rho}
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
q = -\kappa \frac{\partial}{\partial x} \left(T\right) = -\frac{\gamma \mu}{(\gamma - 1)\textrm{Pr}} \frac{\partial}{\partial x} \left(\frac{p}{\rho}\right)
```
which is the form implemented below in the [`flux`](@ref) function.

In one spatial dimensions we require gradients for two quantities, e.g.,
primitive quantities
```math
\frac{\partial}{\partial x} v,\, \frac{\partial}{\partial x} T
```
or the entropy variables
```math
\frac{\partial}{\partial x} w_2,\, \frac{\partial}{\partial x} w_3
```
where
```math
w_2 = \frac{\rho v1}{p},\, w_3 = -\frac{\rho}{p}
```
"""
struct CompressibleNavierStokesDiffusion1D{GradientVariables, RealT <: Real, Mu,
                                           E <: AbstractCompressibleEulerEquations{1}} <:
       AbstractCompressibleNavierStokesDiffusion{1, 3, GradientVariables}
    # TODO: parabolic
    # 1) For now save gamma and inv(gamma-1) again, but could potentially reuse them from the Euler equations
    # 2) Add NGRADS as a type parameter here and in AbstractEquationsParabolic, add `ngradients(...)` accessor function
    gamma::RealT               # ratio of specific heats
    inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications

    mu::Mu                     # viscosity
    Pr::RealT                  # Prandtl number
    kappa::RealT               # thermal diffusivity for Fick's law

    equations_hyperbolic::E    # CompressibleEulerEquations1D
    gradient_variables::GradientVariables # GradientVariablesPrimitive or GradientVariablesEntropy
end

# default to primitive gradient variables
function CompressibleNavierStokesDiffusion1D(equations::CompressibleEulerEquations1D;
                                             mu, Prandtl,
                                             gradient_variables = GradientVariablesPrimitive())
    gamma = equations.gamma
    inv_gamma_minus_one = equations.inv_gamma_minus_one

    # Under the assumption of constant Prandtl number the thermal conductivity
    # constant is kappa = gamma μ / ((gamma-1) Prandtl).
    # Important note! Factor of μ is accounted for later in `flux`.
    # This avoids recomputation of kappa for non-constant μ.
    kappa = gamma * inv_gamma_minus_one / Prandtl

    CompressibleNavierStokesDiffusion1D{typeof(gradient_variables), typeof(gamma),
                                        typeof(mu),
                                        typeof(equations)}(gamma, inv_gamma_minus_one,
                                                           mu, Prandtl, kappa,
                                                           equations,
                                                           gradient_variables)
end

# TODO: parabolic
# This is the flexibility a user should have to select the different gradient variable types
# varnames(::typeof(cons2prim)   , ::CompressibleNavierStokesDiffusion1D) = ("v1", "v2", "T")
# varnames(::typeof(cons2entropy), ::CompressibleNavierStokesDiffusion1D) = ("w2", "w3", "w4")

function varnames(variable_mapping,
                  equations_parabolic::CompressibleNavierStokesDiffusion1D)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

# we specialize this function to compute gradients of primitive variables instead of
# conservative variables.
function gradient_variable_transformation(::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    cons2prim
end
function gradient_variable_transformation(::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})
    cons2entropy
end

# Explicit formulas for the diffusive Navier-Stokes fluxes are available, e.g., in Section 2
# of the paper by Rueda-Ramírez, Hennemann, Hindenlang, Winters, and Gassner
# "An Entropy Stable Nodal Discontinuous Galerkin Method for the resistive
#  MHD Equations. Part II: Subcell Finite Volume Shock Capturing"
# where one sets the magnetic field components equal to 0.
function flux(u, gradients, orientation::Integer,
              equations::CompressibleNavierStokesDiffusion1D)
    # Here, `u` is assumed to be the "transformed" variables specified by `gradient_variable_transformation`.
    rho, v1, _ = convert_transformed_to_primitive(u, equations)
    # Here `gradients` is assumed to contain the gradients of the primitive variables (rho, v1, v2, T)
    # either computed directly or reverse engineered from the gradient of the entropy variables
    # by way of the `convert_gradient_variables` function.
    _, dv1dx, dTdx = convert_derivative_to_primitive(u, gradients, equations)

    # Viscous stress (tensor)
    tau_11 = dv1dx

    # Fick's law q = -kappa * grad(T) = -kappa * grad(p / (R rho))
    # with thermal diffusivity constant kappa = gamma μ R / ((gamma-1) Pr)
    # Note, the gas constant cancels under this formulation, so it is not present
    # in the implementation
    q1 = equations.kappa * dTdx

    # In the simplest cases, the user passed in `mu` or `mu()` 
    # (which returns just a constant) but
    # more complex functions like Sutherland's law are possible.
    # `dynamic_viscosity` is a helper function that handles both cases
    # by dispatching on the type of `equations.mu`.
    mu = dynamic_viscosity(u, equations)

    # viscous flux components in the x-direction
    f1 = 0
    f2 = tau_11 * mu
    f3 = (v1 * tau_11 + q1) * mu

    return SVector(f1, f2, f3)
end

# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleNavierStokesDiffusion1D)
    rho, rho_v1, _ = u

    v1 = rho_v1 / rho
    T = temperature(u, equations)

    return SVector(rho, v1, T)
end

# Convert conservative variables to entropy
# TODO: parabolic. We can improve efficiency by not computing w_1, which involves logarithms
# This can be done by specializing `cons2entropy` and `entropy2cons` to `CompressibleNavierStokesDiffusion1D`,
# but this may be confusing to new users.
function cons2entropy(u, equations::CompressibleNavierStokesDiffusion1D)
    cons2entropy(u, equations.equations_hyperbolic)
end
function entropy2cons(w, equations::CompressibleNavierStokesDiffusion1D)
    entropy2cons(w, equations.equations_hyperbolic)
end

# the `flux` function takes in transformed variables `u` which depend on the type of the gradient variables.
# For CNS, it is simplest to formulate the viscous terms in primitive variables, so we transform the transformed
# variables into primitive variables.
@inline function convert_transformed_to_primitive(u_transformed,
                                                  equations::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    return u_transformed
end

# TODO: parabolic. Make this more efficient!
@inline function convert_transformed_to_primitive(u_transformed,
                                                  equations::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})
    # note: this uses CompressibleNavierStokesDiffusion1D versions of cons2prim and entropy2cons
    return cons2prim(entropy2cons(u_transformed, equations), equations)
end

# Takes the solution values `u` and gradient of the entropy variables (w_2, w_3, w_4) and
# reverse engineers the gradients to be terms of the primitive variables (v1, v2, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
# Note, the first component of `gradient_entropy_vars` contains gradient(rho) which is unused.
# TODO: parabolic; entropy stable viscous terms
@inline function convert_derivative_to_primitive(u, gradient,
                                                 ::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    return gradient
end

# the first argument is always the "transformed" variables.
@inline function convert_derivative_to_primitive(w, gradient_entropy_vars,
                                                 equations::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})

    # TODO: parabolic. This is inefficient to pass in transformed variables but then transform them back.
    # We can fix this if we directly compute v1, v2, T from the entropy variables
    u = entropy2cons(w, equations) # calls a "modified" entropy2cons defined for CompressibleNavierStokesDiffusion1D
    rho, rho_v1, _ = u

    v1 = rho_v1 / rho
    T = temperature(u, equations)

    return SVector(gradient_entropy_vars[1],
                   T * (gradient_entropy_vars[2] + v1 * gradient_entropy_vars[3]), # grad(u) = T*(grad(w_2)+v1*grad(w_3))
                   T * T * gradient_entropy_vars[3])
end

# This routine is required because `prim2cons` is called in `initial_condition`, which
# is called with `equations::CompressibleEulerEquations1D`. This means it is inconsistent
# with `cons2prim(..., ::CompressibleNavierStokesDiffusion1D)` as defined above.
# TODO: parabolic. Is there a way to clean this up?
@inline function prim2cons(u, equations::CompressibleNavierStokesDiffusion1D)
    prim2cons(u, equations.equations_hyperbolic)
end

@inline function temperature(u, equations::CompressibleNavierStokesDiffusion1D)
    rho, rho_v1, rho_e = u

    p = (equations.gamma - 1) * (rho_e - 0.5f0 * rho_v1^2 / rho)
    T = p / rho
    return T
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      orientation::Integer,
                                                                                      direction,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    v1 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                                equations)
    return SVector(u_inner[1], v1, u_inner[3])
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      u_inner,
                                                                                      orientation::Integer,
                                                                                      direction,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    # rho, v1, v2, _ = u_inner
    normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x,
                                                                                                           t,
                                                                                                           equations)
    v1 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                                equations)
    _, tau_1n, _ = flux_inner # extract fluxes for 2nd equation
    normal_energy_flux = v1 * tau_1n + normal_heat_flux
    return SVector(flux_inner[1], flux_inner[2], normal_energy_flux)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       u_inner,
                                                                                       orientation::Integer,
                                                                                       direction,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Gradient,
                                                                                       equations::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    v1 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                                equations)
    T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t,
                                                                                equations)
    return SVector(u_inner[1], v1, T)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       u_inner,
                                                                                       orientation::Integer,
                                                                                       direction,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Divergence,
                                                                                       equations::CompressibleNavierStokesDiffusion1D{GradientVariablesPrimitive})
    return flux_inner
end

# specialized BC impositions for GradientVariablesEntropy.

# This should return a SVector containing the boundary values of entropy variables.
# Here, `w_inner` are the transformed variables (e.g., entropy variables).
#
# Taken from "Entropy stable modal discontinuous Galerkin schemes and wall boundary conditions
#             for the compressible Navier-Stokes equations" by Chan, Lin, Warburton 2022.
# DOI: 10.1016/j.jcp.2021.110723
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      w_inner,
                                                                                      orientation::Integer,
                                                                                      direction,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Gradient,
                                                                                      equations::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})
    v1 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                                equations)
    negative_rho_inv_p = w_inner[3] # w_3 = -rho / p
    return SVector(w_inner[1], -v1 * negative_rho_inv_p, negative_rho_inv_p)
end

# this is actually identical to the specialization for GradientVariablesPrimitive, but included for completeness.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Adiabatic})(flux_inner,
                                                                                      w_inner,
                                                                                      orientation::Integer,
                                                                                      direction,
                                                                                      x,
                                                                                      t,
                                                                                      operator_type::Divergence,
                                                                                      equations::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})
    normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x,
                                                                                                           t,
                                                                                                           equations)
    v1 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                                equations)
    _, tau_1n, _ = flux_inner # extract fluxes for 2nd equation
    normal_energy_flux = v1 * tau_1n + normal_heat_flux
    return SVector(flux_inner[1], flux_inner[2], normal_energy_flux)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       w_inner,
                                                                                       orientation::Integer,
                                                                                       direction,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Gradient,
                                                                                       equations::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})
    v1 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t,
                                                                                equations)
    T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t,
                                                                                equations)

    # the entropy variables w2 = rho * v1 / p = v1 / T = -v1 * w3.
    w3 = -1 / T
    return SVector(w_inner[1], -v1 * w3, w3)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip,
                                                                        <:Isothermal})(flux_inner,
                                                                                       w_inner,
                                                                                       orientation::Integer,
                                                                                       direction,
                                                                                       x,
                                                                                       t,
                                                                                       operator_type::Divergence,
                                                                                       equations::CompressibleNavierStokesDiffusion1D{GradientVariablesEntropy})
    return SVector(flux_inner[1], flux_inner[2], flux_inner[3])
end
end # @muladd
