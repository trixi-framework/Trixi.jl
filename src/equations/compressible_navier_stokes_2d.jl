@doc raw"""
    CompressibleNavierStokesEquations2D(gamma, Re, Pr, Ma_inf, equations)

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
struct CompressibleNavierStokesEquations2D{RealT <: Real, E <: AbstractCompressibleEulerEquations{2}} <: AbstractCompressibleNavierStokesEquations{2, 4}
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
end

function CompressibleNavierStokesEquations2D(equations::CompressibleEulerEquations2D; Reynolds, Prandtl, Mach_freestream)
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
  CompressibleNavierStokesEquations2D{typeof(gamma), typeof(equations)}(gamma, inv_gamma_minus_one,
                                                                        Re, Pr, Ma, kappa,
                                                                        p_inf, u_inf, R,
                                                                        equations)
end


# TODO: parabolic
# This is the flexibility a user should have to select the different gradient variable types
# varnames(::typeof(cons2prim)   , ::CompressibleNavierStokesEquations2D) = ("v1", "v2", "T")
# varnames(::typeof(cons2entropy), ::CompressibleNavierStokesEquations2D) = ("w2", "w3", "w4")

varnames(variable_mapping, equations_parabolic::CompressibleNavierStokesEquations2D) =
  varnames(variable_mapping, equations_parabolic.equations_hyperbolic)

# we specialize this function to compute gradients of primitive variables instead of
# conservative variables.
gradient_variable_transformation(::CompressibleNavierStokesEquations2D, dg_parabolic) = cons2prim

# no orientation specified since the flux is vector-valued
# Explicit formulas for the diffussive Navier-Stokes fluxes are available, e.g. in Section 2
# of the paper by Svärd, Carpenter and Nordström
# "A stable high-order finite difference scheme for the compressible Navier–Stokes
#  equations, far-field boundary conditions"
# Although these authors use a different nondimensionalization so some constants are different
# particularly for Fick's law.
#
# Note, could be generalized to use Sutherland's law to get the molecular and thermal
# diffusivity
function flux(u, grad_u, equations::CompressibleNavierStokesEquations2D)
  # Here grad_u is assumed to contain the gradients of the primitive variables (v1, v2, T)
  # either computed directly or reverse engineered from the gradient of the entropy vairables
  # by way of the `convert_gradient_variables` function
  rho, v1, v2, _ = u

  # grad_u contains derivatives of each hyperbolic variable
  _, dv1dx, dv2dx, dTdx = grad_u[1]
  _, dv1dy, dv2dy, dTdy = grad_u[2]

  # Components of viscous stress tensor

  # (4/3 * (v1)_x - 2/3 * (v2)_y)
  tau_11 = ( 4.0 / 3.0 * dv1dx - 2.0 / 3.0 * dv2dy )
  # ((v1)_y + (v2)_x)
  # stress tensor is symmetric
  tau_12 = ( dv1dy + dv2dx ) # = tau_21
  # (4/3 * (v2)_y - 2/3 * (v1)_x)
  tau_22 = ( 4.0 / 3.0 * dv2dy - 2.0 / 3.0 * dv1dx )

  # Fick's law q = -kappa * grad(T); constant is kappa = gamma μ R / ((gamma-1) Pr)
  # Important note! Due to nondimensional scaling R = 1 / gamma, so the
  # temperature T in the gradient computation already contains a factor of gamma
  q1 = equations.kappa * dTdx
  q2 = equations.kappa * dTdy

  # kinematic viscosity is simply 1/Re for this nondimensionalization
  mu = 1.0 / equations.Re

  # viscous flux components in the x-direction
  f1 = zero(rho)
  f2 = tau_11 * mu
  f3 = tau_12 * mu
  f4 = ( v1 * tau_11 + v2 * tau_12 + q1 ) * mu

  # viscous flux components in the y-direction
  # Note, symmetry is exploited for tau_12 = tau_21
  g1 = zero(rho)
  g2 = f3 # tau_21 * mu
  g3 = tau_22 * mu
  g4 = ( v1 * tau_12 + v2 * tau_22 + q2 ) * mu

  return (SVector(f1, f2, f3, f4) , SVector(g1, g2, g3, g4))
end


# Convert conservative variables to primitive
@inline function cons2prim(u, equations::CompressibleNavierStokesEquations2D)
  rho, rho_v1, rho_v2, _ = u

  v1 = rho_v1 / rho
  v2 = rho_v2 / rho
  T  = temperature(u, equations)

  return SVector(rho, v1, v2, T)
end

# TODO: make this consistent with cons2prim above and cons2prim for Euler!
@inline prim2cons(u, equations::CompressibleNavierStokesEquations2D) =
    prim2cons(u, equations.equations_hyperbolic)


# # Convert conservative variables to entropy
# @inline function cons2entropy(u, equations::CompressibleNavierStokesEquations2D)
#   rho, rho_v1, rho_v2, rho_e = u

#   v1 = rho_v1 / rho
#   v2 = rho_v2 / rho
#   v_square = v1^2 + v2^2
#   p = (equations.gamma - 1) * (rho_e - 0.5 * rho * v_square)

#   rho_p = rho / p

#   w2 = rho_p * v1
#   w3 = rho_p * v2
#   w4 = -rho_p

#   return SVector(w2, w3, w4)
# end


# @inline function convert_gradient_variables(u, grad_entropy_vars, equations::CompressibleNavierStokesEquations2D)
# # Takes the solution values `u` and gradient of the variables (w_2, w_3, w_4) and
# # reverse engineers the gradients to be terms of the primitive variables (v1, v2, T).
# # Helpful because then the diffusive fluxes have the same form as on paper.
#   rho, rho_v1, rho_v2, _ = u

#   v1 = rho_v1 / rho
#   v2 = rho_v2 / rho
#   T  = temperature(u, equations)

#   return SVector(equations.R * T * (grad_entropy_vars[1] + v1 * grad_entropy_vars[3]), # grad(u) = R*T*(grad(w_2)+v1*grad(w_4))
#                  equations.R * T * (grad_entropy_vars[2] + v2 * grad_entropy_vars[3]), # grad(v) = R*T*(grad(w_3)+v2*grad(w_4))
#                  equations.R * T * T * grad_entropy_vars[3]                            # grad(T) = R*T^2*grad(w_4))
#                 )
# end


@inline function temperature(u, equations::CompressibleNavierStokesEquations2D)
  rho, rho_v1, rho_v2, rho_e = u

  p = (equations.gamma - 1) * (rho_e - 0.5 * (rho_v1^2 + rho_v2^2) / rho)
  T = p / (equations.R * rho)
  return T
end

"""
    struct BoundaryConditionViscousWall

Creates a wall-type boundary conditions for the compressible Navier-Stokes equations.
The fields `boundary_condition_velocity` and `boundary_condition_heat_flux` are intended
to be boundary condition types such as the `NoSlip` velocity boundary condition and the
`Adiabatic` or `Isothermal` heat boundary condition.
"""
struct BoundaryConditionViscousWall{V, H}
  boundary_condition_velocity::V
  boundary_condition_heat_flux::H
end

"""
    struct NoSlip

Use to create a no-slip boundary condition with `BoundaryConditionViscousWall`. The field `boundary_value_function`
should be a function with signature `boundary_value_function(x, t, equations)`
and should return a `SVector{NDIMS}` whose entries are the velocity vector at a
point `x` and time `t`.
"""
struct NoSlip{F}
  boundary_value_function::F # value of the velocity vector on the boundary
end

"""
    struct Isothermal

Creates an isothermal temperature boundary condition with field `boundary_value_function`,
which should be a function with signature `boundary_value_function(x, t, equations)` and
return a scalar value for the temperature at point `x` and time `t`.
"""
struct Isothermal{F}
  boundary_value_function::F # value of the temperature on the boundary
end

"""
    struct Adiabatic

Creates an adiabatic temperature boundary condition with field `boundary_value_function`,
which should be a function with signature `boundary_value_function(x, t, equations)` and
return a scalar value for the normal heat flux at point `x` and time `t`.
"""
struct Adiabatic{F}
  boundary_value_normal_flux_function::F # scaled heat flux 1/T * kappa * dT/dn
end

@inline function (boundary_condition::BoundaryConditionViscousWall{<:NoSlip, <:Adiabatic})(flux_inner, u_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Gradient,
                                                                                           equations::CompressibleNavierStokesEquations2D)
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  return SVector(u_inner[1], v1, v2, u_inner[4])
end

@inline function (boundary_condition::BoundaryConditionViscousWall{<:NoSlip, <:Adiabatic})(flux_inner, u_inner, normal::AbstractVector,
                                                                                           x, t, operator_type::Divergence,
                                                                                           equations::CompressibleNavierStokesEquations2D)
  # rho, v1, v2, _ = u_inner
  normal_heat_flux = boundary_condition.boundary_condition_heat_flux.boundary_value_normal_flux_function(x, t, equations)
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  tau_1n, tau_2n = flux_inner[2:3] # extract fluxes for 2nd and 3rd equations
  normal_energy_flux = v1 * tau_1n + v2 * tau_2n + normal_heat_flux
  return SVector(flux_inner[1:3]..., normal_energy_flux)
end

@inline function (boundary_condition::BoundaryConditionViscousWall{<:NoSlip, <:Isothermal})(flux_inner, u_inner, normal::AbstractVector,
                                                                                            x, t, operator_type::Gradient,
                                                                                            equations::CompressibleNavierStokesEquations2D)
  v1, v2 = boundary_condition.boundary_condition_velocity.boundary_value_function(x, t, equations)
  T = boundary_condition.boundary_condition_heat_flux.boundary_value_function(x, t, equations)
  return SVector(u_inner[1], v1, v2, T)
end

@inline function (boundary_condition::BoundaryConditionViscousWall{<:NoSlip, <:Isothermal})(flux_inner, u_inner, normal::AbstractVector,
                                                                                            x, t, operator_type::Divergence,
                                                                                            equations::CompressibleNavierStokesEquations2D)
  return flux_inner
end

