# TODO: can we generalize this to MHD?
"""
    struct BoundaryConditionNavierStokesWall

Creates a wall-type boundary conditions for the compressible Navier-Stokes equations.
The fields `boundary_condition_velocity` and `boundary_condition_heat_flux` are intended
to be boundary condition types such as the `NoSlip` velocity boundary condition and the
`Adiabatic` or `Isothermal` heat boundary condition.
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

"""
`GradientVariablesPrimitive` and `GradientVariablesEntropy` are gradient variable type parameters
for `CompressibleNavierStokesDiffusion1D`. By default, the gradient variables are set to be
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

"""
    dynamic_viscosity(u, equations)

Wrapper for the dynamic viscosity that calls
`dynamic_viscosity(u, equations.mu, equations)`, which dispatches on the type of 
`equations.mu`. 
For constant `equations.mu`, i.e., `equations.mu` is of `Real`-type it is returned directly.
In all other cases, `equations.mu` is assumed to be a function with arguments
`u` and `equations` and is called with these arguments.
"""
dynamic_viscosity(u, equations) = dynamic_viscosity(u, equations.mu, equations)
dynamic_viscosity(u, mu::Real, equations) = mu
dynamic_viscosity(u, mu::T, equations) where {T} = mu(u, equations)
