# TODO: can we generalize this to MHD?
"""
    struct BoundaryConditionNavierStokesWall

Creates a wall-type boundary conditions for the compressible Navier-Stokes equations, see
[`CompressibleNavierStokesDiffusion1D`](@ref), [`CompressibleNavierStokesDiffusion2D`](@ref), and
[`CompressibleNavierStokesDiffusion3D`](@ref).
The fields `boundary_condition_velocity` and `boundary_condition_heat_flux` are intended
to be boundary condition types such as the [`NoSlip`](@ref) velocity boundary condition and the
[`Adiabatic`](@ref) or [`Isothermal`](@ref) heat boundary condition.
"""
struct BoundaryConditionNavierStokesWall{V, H}
    boundary_condition_velocity::V
    boundary_condition_heat_flux::H
end

"""
    struct NoSlip

Use to create a no-slip boundary condition with [`BoundaryConditionNavierStokesWall`](@ref). 
The field `boundary_value_function` should be a function with signature 
`boundary_value_function(x, t, equations)` and return a `SVector{NDIMS}` 
whose entries are the velocity vector at a point `x` and time `t`.
"""
struct NoSlip{F}
    boundary_value_function::F # value of the velocity vector on the boundary
end

"""
    struct Slip

Creates a symmetric velocity boundary condition which eliminates any normal velocity gradients across the boundary, i.e., 
allows only the tangential velocity gradients to be non-zero.
When combined with the heat boundary condition [`Adiabatic`](@ref), this creates a truly symmetric boundary condition.
Any boundary on which this combined boundary condition is applied thus acts as a symmetry plane for the flow.
In contrast to the [`NoSlip`](@ref) boundary condition, `Slip` does not require a function to be supplied.

The (purely) hyperbolic equivalent boundary condition is [`boundary_condition_slip_wall`](@ref) which 
permits only tangential velocities.

This boundary condition can also be employed as a reflective wall.

Note that in 1D this degenerates to the [`NoSlip`](@ref) boundary condition which must be used instead.

!!! note
    Currently this (velocity) boundary condition is only implemented for 
    [`P4estMesh`](@ref) and [`GradientVariablesPrimitive`](@ref).
"""
struct Slip end

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
`GradientVariablesPrimitive` is a gradient variable type parameter for the [`CompressibleNavierStokesDiffusion1D`](@ref), 
[`CompressibleNavierStokesDiffusion2D`](@ref), and [`CompressibleNavierStokesDiffusion3D`](@ref).
The other available gradient variable type parameter is [`GradientVariablesEntropy`](@ref).
By default, the gradient variables are set to be `GradientVariablesPrimitive`.
"""
struct GradientVariablesPrimitive end

"""
`GradientVariablesEntropy` is a gradient variable type parameter for the [`CompressibleNavierStokesDiffusion1D`](@ref), 
[`CompressibleNavierStokesDiffusion2D`](@ref), and [`CompressibleNavierStokesDiffusion3D`](@ref).
The other available gradient variable type parameter is [`GradientVariablesPrimitive`](@ref).

Specifying `GradientVariablesEntropy` uses the entropy variable formulation from
- Hughes, Mallet, Franca (1986)
  A new finite element formulation for computational fluid dynamics: I. Symmetric forms of the
  compressible Euler and Navier-Stokes equations and the second law of thermodynamics.
  [https://doi.org/10.1016/0045-7825(86)90127-1](https://doi.org/10.1016/0045-7825(86)90127-1)

Under `GradientVariablesEntropy`, the Navier-Stokes discretization is provably entropy stable.
"""
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

"""
    have_constant_diffusivity(::AbstractCompressibleNavierStokesDiffusion)

# Returns
- `False()`

Used in diffusive CFL condition computation (see [`StepsizeCallback`](@ref)) to indicate that the
diffusivity is not constant in space and that [`max_diffusivity`](@ref) needs to be computed
at every node in every element.

Also employed in [`linear_structure`](@ref) and [`linear_structure_parabolic`](@ref) to check
if the diffusion term is linear in the variables/constant.
"""
@inline have_constant_diffusivity(::AbstractCompressibleNavierStokesDiffusion) = False()
