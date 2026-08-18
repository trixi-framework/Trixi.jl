# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusion{NDIMS, NVARS} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative} end

"""
    have_constant_diffusivity(::AbstractLaplaceDiffusion)

# Returns
- `True()`

Used in parabolic cfl condition computation (see [`StepsizeCallback`](@ref)) to indicate that the
diffusivity is constant in space and that [`max_diffusivity`](@ref) needs **not** to be re-computed
at every node in every element.

Also employed in [`linear_structure`](@ref) and [`linear_structure_parabolic`](@ref) to check
if the diffusion term is linear in the variables/constant.
"""
@inline have_constant_diffusivity(::AbstractLaplaceDiffusion) = True()

"""
    max_diffusivity(equations_parabolic::AbstractLaplaceDiffusion)

# Returns
- `equations_parabolic.diffusivity`

Returns isotropic diffusion coefficient for use in parabolic cfl condition computation,
see [`StepsizeCallback`](@ref).
"""
@inline function max_diffusivity(equations_parabolic::AbstractLaplaceDiffusion)
    return equations_parabolic.diffusivity
end

include("laplace_diffusion_1d.jl")
include("laplace_diffusion_2d.jl")
include("laplace_diffusion_3d.jl")
