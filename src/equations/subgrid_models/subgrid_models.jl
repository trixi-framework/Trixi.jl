# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    AbstractSubgridModel

Abstract supertype of eddy viscosity models for large eddy simulations (LES), which are
passed as the keyword argument `subgrid_model` of parabolic equations such as
[`CompressibleLES3D`](@ref).

A subgrid-scale model `subgrid_model` must have a field `Prandtl_turbulent` and implement
- `calc_subgrid_model_coefficients(u_transformed, gradients, element, subgrid_model, mesh,
  equations, dg, cache)`, which returns the `coefficients` of `element`, e.g., depending
  on the size of the element,
- `eddy_viscosity(u, gradients, coefficients, subgrid_model, equations)`, which returns
  the eddy viscosity `mu_t` for the transformed variables `u` and the `gradients` at a
  node with the `coefficients` of its element.

Optionally, it can implement
- `create_cache_subgrid_model(subgrid_model, mesh, equations, dg, cache, uEltype)`, which
  returns a `NamedTuple` of arrays added to the `cache`, e.g., scratch arrays for
  `calc_subgrid_model_coefficients` (by default, none).

The models access the density, velocity, and velocity gradients only via
`convert_transformed_to_density`, `convert_transformed_to_velocity`, and
`convert_derivative_to_velocity_gradients`, see [`AbstractLESDiffusion`](@ref). Thus,
they can be used with all equations implementing this interface.
"""
abstract type AbstractSubgridModel end

"""
    AbstractLESDiffusion{NDIMS, NVARS, GradientVariables}

Abstract supertype of the parabolic terms of filtered equations for large eddy simulations
(LES) with a subgrid-scale model (see [`AbstractSubgridModel`](@ref)) in the field
`subgrid_model`, e.g., [`CompressibleLES3D`](@ref). The parabolic fluxes are computed with
the eddy viscosity of the subgrid-scale model by a kernel that is the same for all
subgrid-scale models and equations. Concrete equations implement
- `convert_transformed_to_density(u, equations)` and
  `convert_transformed_to_velocity(u, equations)` for the transformed variables `u` (see
  `gradient_variable_transformation`), and
  `convert_derivative_to_velocity_gradients(u, gradients, equations)`, which returns the
  gradients `(dv/dx, dv/dy, dv/dz)` of the velocity `v`,
- `diffusivities(u, mu_t, equations)`, which returns an `SVector` of the diffusivities
  including the eddy viscosity `mu_t`, e.g., the viscosity and the thermal conductivity
  (and the diffusivities of further quantities such as tracers),
- `flux(u, gradients, orientation, diffusivities, equations)`, which returns the parabolic
  flux with these `diffusivities`,
in addition to the functionality of parabolic equations such as the
`gradient_variable_transformation` and the boundary conditions.
"""
abstract type AbstractLESDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end

subgrid_model(equations::AbstractLESDiffusion) = equations.subgrid_model
subgrid_model(equations_parabolic) = nothing

have_constant_diffusivity(::AbstractLESDiffusion) = False()

# Arrays of the subgrid-scale model, which are added to the `cache` of the
# `SemidiscretizationHyperbolicParabolic`, see `AbstractSubgridModel`
create_cache_subgrid_model(::Nothing, mesh, equations_parabolic, dg, cache, uEltype) = (;)

function create_cache_subgrid_model(subgrid_model::AbstractSubgridModel, mesh,
                                    equations_parabolic, dg, cache, uEltype)
    throw(ArgumentError("$(typeof(subgrid_model)) is not implemented for $(typeof(mesh)) and $(typeof(equations_parabolic))"))
end
end # @muladd

include("smagorinsky.jl")
