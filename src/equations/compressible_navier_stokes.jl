
# Compressible Navier-Stokes equations
abstract type AbstractCompressibleNavierStokesDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end

# This enables "forwarded" accesses to e.g.`equations_parabolic.gamma` of the "underlying" `equations_hyperbolic`
# while keeping direct access to parabolic-specific fields like `mu` or `kappa`.
@inline function Base.getproperty(equations_parabolic::AbstractCompressibleNavierStokesDiffusion,
                                  field::Symbol)
    if field === :gamma || field === :inv_gamma_minus_one
        return getproperty(getfield(equations_parabolic, :equations_hyperbolic), field)
    else
        return getfield(equations_parabolic, field)
    end
end

# Provide property names for e.g. tab-completion by combining
# the names from the underlying hyperbolic equations with the fields of this parabolic part.
@inline function Base.propertynames(equations_parabolic::AbstractCompressibleNavierStokesDiffusion,
                                    private::Bool = false)
    names_hyp = (:gamma, :inv_gamma_minus_one)
    names_para = fieldnames(typeof(equations_parabolic))
    names_hyp_para = (names_hyp..., names_para...)

    return names_hyp_para
end

# TODO: can we generalize this to V(R)-MHD?
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

Used in parabolic CFL condition computation (see [`StepsizeCallback`](@ref)) to indicate that the
diffusivity is not constant in space and that [`max_diffusivity`](@ref) needs to be computed
at every node in every element.

Also employed in [`linear_structure`](@ref) and [`linear_structure_parabolic`](@ref) to check
if the diffusion term is linear in the variables/constant.
"""
@inline have_constant_diffusivity(::AbstractCompressibleNavierStokesDiffusion) = False()

@doc raw"""
    RadiativeEquilibriumOneWay

This implements a one-way coupled radiative equilibrium boundary condition for the `AbstractCompressibleNavierStokesDiffusion` equations.
The main use case of this thermal/heat boundary condition is to model radiative cooling,
the main mechanism of heat loss of objects in hypersonic flows, e.g. reentry vehicles.
The wall temperature ``T_w`` is computed from the heat flux at the wall ``q_w`` and the far-field temperature ``T_\infty`` via
```math
T_w = \left(\frac{q_w}{\epsilon \sigma} + T_\infty^4\right)^{1/4}
```
in the above equation, ``\epsilon`` is the emissivity of the gray body wall and ``\sigma`` is the Stefan-Boltzmann constant.
Note that this is only correct for convex surfaces, i.e., surfaces that do not see themselves.
The required treatment involving view factors is not yet implemented.

As a side note: In a fully coupled code, the fluid heat flux would not be taken as input,
but (re)computed from the temperature gradient at the wall, i.e.,
```math
q_w = -\kappa \frac{\partial T}{\partial y}\bigg|_w \approx \kappa \frac{T_\text{f} - T_w}{\Delta y}
```
where ``\kappa`` denotes the thermal conductivity of the fluid and ``T_f`` is the temperature of the fluid at the first interior node,
while ``T_w`` is the temperature of the Gauss-Lobatto node at the wall.
Note that this assumes that the wall has the same temperature as the fluid, i.e.,
temperature jumps as for rarefied gases are not considered.
The radiative heat flux is given by
```math
q_r = \epsilon \sigma (T_w^4 - T_\infty^4)
```

## References
See Chapter 3 and in particular equations (3.12) to (3.14) in

- Hirschel (2015).
  Basics of Aerothermodynamics, 2nd Edition.
  [DOI: 10.1007/978-3-319-14373-6](https://doi.org/10.1007/978-3-319-14373-6)
"""
struct RadiativeEquilibriumOneWay{TempFarfield4 <: Real,
                                  EpsTimesSigma <: Real}
    temp_farfield4::TempFarfield4
    eps_times_sigma::EpsTimesSigma
end

@doc raw"""
    RadiativeEquilibriumOneWay(;
                         emissivity = 1.0,
                         T_far_field = 0.0f0,
                         stefan_boltzmann = 5.670374419f-8)

See [`RadiativeEquilibriumOneWay`](@ref) for details.

`emissivity` is the gray body radiation emissivity ``\epsilon`` of the wall,
`T_far_field` is the far-field temperature ``T_\infty`` of the surrounding fluid, and
`stefan_boltzmann` is the Stefan-Boltzmann constant ``\sigma``.
"""
function RadiativeEquilibriumOneWay(;
                                    emissivity = 1.0f0,
                                    T_far_field = 0.0f0,
                                    stefan_boltzmann = 5.670374419e-8)
    eps_times_sigma = emissivity * stefan_boltzmann
    temp_farfield4 = T_far_field^4

    return RadiativeEquilibriumOneWay{typeof(temp_farfield4),
                                      typeof(eps_times_sigma)}(temp_farfield4,
                                                               eps_times_sigma)
end

@inline function solve_radiative_equilibrium_temperature(normal_heat_flux,
                                                         rad_eq_bc::RadiativeEquilibriumOneWay)
    @unpack eps_times_sigma, temp_farfield4 = rad_eq_bc

    T_wall = (normal_heat_flux / eps_times_sigma + temp_farfield4)^(1 / 4)

    return T_wall
end

include("compressible_navier_stokes_1d.jl")
include("compressible_navier_stokes_2d.jl")
include("compressible_navier_stokes_3d.jl")
