@doc raw"""
    CompressibleNavierStokesDiffusion2D(gamma, Re, Pr, Ma_inf, equations)

These equations contain the diffusion (i.e. parabolic) terms applied
to mass, momenta, and total energy together with the advective terms from
the [`CompressibleEulerEquations2D`](@ref).

- `gamma`: adiabatic constant,
- `Re`: Reynolds number,
- `Pr`: Prandtl number,
- `Ma_inf`: free-stream Mach number
- `equations`: instance of the [`CompressibleEulerEquations2D`](@ref)

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
as the pressure. The terms on the right hand side of the system above
are built from the viscous stress tensor
```math
\underline{\tau} = \mu \left(\nabla\mathbf{v} + \left(\nabla\mathbf{v}\right)^T\right) - \frac{2}{3} \mu \left(\nabla\cdot\mathbf{v}\right)\underline{I}
```
where ``\underline{I}`` is the ``2\times 2`` identity matrix and the heat flux is
```math
\nabla q = -\kappa\nabla\left(T\right),\quad T = \frac{p}{R\rho}
```
where ``T`` is the temperature and ``\kappa`` is the thermal conductivity for Fick's law.
Under the assumption that the gas has a constant Prandtl number
the thermal conductivity is
```math
\kappa = \frac{\gamma \mu R}{(\gamma - 1)\textrm{Pr}}
```

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

For this particular scaling the vicosity is set internally to be μ = 1/Re.
Further, the nondimensionalization takes the density-temperature-sound speed as
the principle quantities such that
```
rho_inf = 1.0
T_ref = 1.0
c_inf = 1.0
p_inf = 1.0 / gamma
u_inf = Ma_inf
R = 1.0 / gamma
```

Other normalization strategies exist, see the reference below for details.
- Marc Montagnac (2013)
  Variable Normalization (nondimensionalization and scaling) for Navier-Stokes
  equations: a practical guide
  [CERFACS Technical report](https://www.cerfacs.fr/~montagna/TR-CFD-13-77.pdf)
The scaling used herein is Section 4.5 of the reference.
"""
struct CompressibleNavierStokesDiffusion2D{GradientVariables, RealT <: Real, E <: AbstractCompressibleEulerEquations{2}} <: AbstractCompressibleNavierStokesDiffusion{2, 4}
  # TODO: parabolic
  # 1) For now save gamma and inv(gamma-1) again, but could potentially reuse them from the Euler equations
  # 2) Add NGRADS as a type parameter here and in AbstractEquationsParabolic, add `ngradients(...)` accessor function
  gamma::RealT               # ratio of specific heats
  inv_gamma_minus_one::RealT # = inv(gamma - 1); can be used to write slow divisions as fast multiplications
  Re::RealT                  # Reynolds number
  Pr::RealT                  # Prandtl number
  Ma_inf::RealT              # free-stream Mach number
  kappa::RealT               # thermal diffusivity for Fick's law

  p_inf::RealT               # free-stream pressure
  u_inf::RealT               # free-stream velocity
  R::RealT                   # gas constant (depends on nondimensional scaling!)

  equations_hyperbolic::E    # CompressibleEulerEquations2D
  gradient_variables::GradientVariables # GradientVariablesPrimitive or GradientVariablesEntropy
end

# type parameters for CompressibleNavierStokesDiffusion2D
struct GradientVariablesPrimitive end
struct GradientVariablesEntropy end

# default to primitive gradient variables
function CompressibleNavierStokesDiffusion2D(equations::CompressibleEulerEquations2D;
                                             Reynolds, Prandtl, Mach_freestream,
                                             gradient_variables = GradientVariablesPrimitive())
  gamma = equations.gamma
  inv_gamma_minus_one = equations.inv_gamma_minus_one
  Re, Pr, Ma = promote(Reynolds, Prandtl, Mach_freestream)

  # Under the assumption of constant Prandtl number the thermal conductivity
  # constant is kappa = gamma μ R / ((gamma-1) Pr).
  # Important note! Due to nondimensional scaling R = 1 / gamma, this constant
  # simplifies slightly. Also, the factor of μ is accounted for later.
  kappa = inv_gamma_minus_one / Pr

  # From the nondimensionalization discussed above set the remaining free-stream
  # quantities
  p_inf = 1 / gamma
  u_inf = Mach_freestream
  R     = 1 / gamma
  CompressibleNavierStokesDiffusion2D{typeof(gradient_variables), typeof(gamma), typeof(equations)}(gamma, inv_gamma_minus_one,
                                                                                                    Re, Pr, Ma, kappa,
                                                                                                    p_inf, u_inf, R,
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

# Explicit formulas for the diffussive Navier-Stokes fluxes are available, e.g. in Section 2
# of the paper by Svärd, Carpenter and Nordström
# "A stable high-order finite difference scheme for the compressible Navier–Stokes
#  equations, far-field boundary conditions"
# Although these authors use a different nondimensionalization so some constants are different
# particularly for Fick's law.
#
# Note, could be generalized to use Sutherland's law to get the molecular and thermal
# diffusivity
function flux(u, gradients, orientation::Integer, equations::CompressibleNavierStokesDiffusion2D)
  # Here `gradients` is assumed to contain the gradients of the primitive variables (rho, v1, v2, T)
  # either computed directly or reverse engineered from the gradient of the entropy vairables
  # by way of the `convert_gradient_variables` function
  _, dv1dx, dv2dx, dTdx = convert_derivative_to_primitive(u, gradients[1], equations)
  _, dv1dy, dv2dy, dTdy = convert_derivative_to_primitive(u, gradients[2], equations)

  rho, v1, v2, _ = u

  # Components of viscous stress tensor

  # (4/3 * (v1)_x - 2/3 * (v2)_y)
  tau_11 = 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy
  # ((v1)_y + (v2)_x)
  # stress tensor is symmetric
  tau_12 = dv1dy + dv2dx # = tau_21
  # (4/3 * (v2)_y - 2/3 * (v1)_x)
  tau_22 = 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx

  # Fick's law q = -kappa * grad(T); constant is kappa = gamma μ R / ((gamma-1) Pr)
  # Important note! Due to nondimensional scaling R = 1 / gamma, so the
  # temperature T in the gradient computation already contains a factor of gamma
  q1 = equations.kappa * dTdx
  q2 = equations.kappa * dTdy

  # kinematic viscosity is simply 1/Re for this nondimensionalization
  mu = 1.0 / equations.Re

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
# Note, only w_2, w_3, w_4 are needed for the viscous fluxes so we avoid computing
# w_1 and simply copy over rho.
# TODO: parabolic; entropy stable viscous terms
# JC: is this the same as `cons2entropy` for CompressibleEulerEquations2D?
@inline function cons2entropy(u, equations::CompressibleNavierStokesDiffusion2D)
  rho, rho_v1, rho_v2, rho_e = u

  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)

  return SVector(rho, rho_v1/p, rho_v2/p, -rho/p)
end

# Takes the solution values `u` and gradient of the entropy variables (w_2, w_3, w_4) and
# reverse engineers the gradients to be terms of the primitive variables (v1, v2, T).
# Helpful because then the diffusive fluxes have the same form as on paper.
# Note, the first component of `gradient_entropy_vars` contains gradient(rho) which is unused.
# TODO: parabolic; entropy stable viscous terms
@inline function convert_derivative_to_primitive(u, gradient, ::CompressibleNavierStokesDiffusion2D{GradientVariablesPrimitive})
  return gradient
end

@inline function convert_derivative_to_primitive(u, gradient_entropy_vars,
                                                equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  return SVector(gradient_entropy_vars[1],
                 equations.R * T * (gradient_entropy_vars[2] + v1 * gradient_entropy_vars[4]), # grad(u) = R*T*(grad(w_2)+v1*grad(w_4))
                 equations.R * T * (gradient_entropy_vars[3] + v2 * gradient_entropy_vars[4]), # grad(v) = R*T*(grad(w_3)+v2*grad(w_4))
                 equations.R * T * T * gradient_entropy_vars[4]                                # grad(T) = R*T^2*grad(w_4))
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
  T = p / (equations.R * rho)
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
# Here, `u_inner` are the transformed variables (e.g., entropy variables).
#
# Taken from "Entropy stable modal discontinuous Galerkin schemes and wall boundary conditions
#             for the compressible Navier-Stokes equations" by Chan, Lin, Warburton 2022.
# DOI: 10.1016/j.jcp.2021.110723
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Adiabatic})(flux_inner, w_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Gradient,
                                                                                           equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  return SVector(w_inner[1], -v1 * w_inner[4], -v2 * w_inner[4], w_inner[4])
end

# this is actually identical to the specialization for GradientVariablesPrimitive, but included for completeness.
@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Adiabatic})(flux_inner, u_inner, normal::AbstractVector,
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

  # the entropy variables w2 = rho * v1 / p = v1 / (equations.R * T) = -v1 * w4. Similarly for w3
  w4 = -1 / (equations.R * T)
  return SVector(w_inner[1], -v1 * w4, -v2 * w4, w4)
end

@inline function (boundary_condition::BoundaryConditionNavierStokesWall{<:NoSlip, <:Isothermal})(flux_inner, u_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Divergence,
                                                                                           equations::CompressibleNavierStokesDiffusion2D{GradientVariablesEntropy})
  return SVector(flux_inner[1], flux_inner[2], flux_inner[3], flux_inner[4])
end
