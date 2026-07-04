
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

# Radiative-equilibrium no-slip wall BC for CompressibleNavierStokesDiffusion1D
#
# Physics: convective/conductive heat reaching the wall is balanced by
# grey-body radiative emission to a cold far field (neglecting incoming
# radiation):
#
#     k(T_w) * dT/dn|_w  =  eps * sigma * (T_w^4 - T_far_field^4)
#
# T_w is not prescribed -- it's the root of this nonlinear balance, solved
# locally (per boundary node) via Newton's method, using a one-sided estimate
# of dT/dn purely as the *iteration* surrogate. Trixi's own (more accurate,
# lifted) gradient machinery is what actually gets used in the real flux,
# once we hand back the converged T_w as the boundary value -- exactly the
# same structural slot the built-in `Isothermal` BC fills with a prescribed T.

"""
    RadiativeEquilibrium
"""
struct RadiativeEquilibrium{TempFarfield4 <: Real,
                            EpsTimesSigma <: Real}
    temp_farfield4::TempFarfield4
    eps_times_sigma::EpsTimesSigma
end

"""
    RadiativeEquilibrium(;
                         emissivity = 1.0,
                         T_far_field = 0.0f0,
                         stefan_boltzmann = 5.670374419f-8)
"""
function RadiativeEquilibrium(;
                              emissivity = 1.0f0,
                              T_far_field = 0.0f0,
                              stefan_boltzmann = 5.670374419e-8)
    eps_times_sigma = emissivity * stefan_boltzmann
    temp_farfield4 = T_far_field^4

    return RadiativeEquilibrium{typeof(temp_farfield4),
                                typeof(eps_times_sigma)}(temp_farfield4,
                                                         eps_times_sigma)
end

@inline function solve_radiative_equilibrium_temperature(T_inner, normal_heat_flux,
                                                         rad_eq_bc, equations)
    @unpack eps_times_sigma, temp_farfield4 = rad_eq_bc
    @unpack kappa = equations

    # Initialize wall temperature from `normal_heat_flux`
    T_wall = (normal_heat_flux / eps_times_sigma + temp_farfield4)^(1 / 4)

    #=
    rel_tol = 1e-8 # TODO: Make field of BC
    for _ in 1:max_iter
        T_wall_3 = T_wall^3

        # TODO: Need dx for temperature gradient and conductive heat flux
        q_cond = kappa * (T_inner - T_wall)
        q_rad = eps_times_sigma * (T_wall_3 * T_wall - temp_farfield4)
        q_diff = q_cond - q_rad

        dq_cond_dT = -kappa
        dq_rad_dT = 4 * eps_times_sigma * T_wall_3
        dq_diff_dT = dq_cond_dT - dq_rad_dT

        dT = -q_diff / dq_diff_dT
        rad_eq_bc.temp_wall += dT
        T_wall = max(T_wall, 1)

        if abs(dT) < rel_tol * max(T_wall, 1)
            break
        end
    end
    =#

    return T_wall
end

include("compressible_navier_stokes_1d.jl")
include("compressible_navier_stokes_2d.jl")
include("compressible_navier_stokes_3d.jl")
