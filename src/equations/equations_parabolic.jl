# specify transformation of conservative variables prior to taking gradients.
# specialize this function to compute gradients e.g., of primitive variables instead of conservative
gradient_variable_transformation(::AbstractEquationsParabolic) = cons2cons

# By default, the gradients are taken with respect to the conservative variables.
# this is reflected by the type parameter `GradientVariablesConservative` in the abstract
# type `AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative}`.
struct GradientVariablesConservative end

# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusion{NDIMS, NVARS} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative} end

"""
    max_diffusivity(::AbstractLaplaceDiffusion)

# Returns
- `True()`

Used in diffusive CFL condition computation (see [`StepsizeCallback`](@ref)) to indicate that the
diffusivity is constant in space and that [`max_diffusivity`](@ref) needs **not** to be re-computed
at every node in every element.
"""
@inline have_constant_diffusivity(::AbstractLaplaceDiffusion) = True()

"""
    max_diffusivity(equations_parabolic::AbstractLaplaceDiffusion)

# Returns
- `equations_parabolic.diffusivity`

Returns isotropic diffusion coefficient for use in diffusive CFL condition computation,
see [`StepsizeCallback`](@ref).
"""
@inline function max_diffusivity(equations_parabolic::AbstractLaplaceDiffusion)
    return equations_parabolic.diffusivity
end

include("laplace_diffusion_1d.jl")
include("laplace_diffusion_2d.jl")
include("laplace_diffusion_3d.jl")

include("laplace_diffusion_entropy_variables.jl")
include("laplace_diffusion_entropy_variables_1d.jl")
include("laplace_diffusion_entropy_variables_2d.jl")
include("laplace_diffusion_entropy_variables_3d.jl")

# Compressible Navier-Stokes equations
abstract type AbstractCompressibleNavierStokesDiffusion{NDIMS, NVARS, GradientVariables} <:
              AbstractEquationsParabolic{NDIMS, NVARS, GradientVariables} end
include("compressible_navier_stokes.jl")
include("compressible_navier_stokes_1d.jl")
include("compressible_navier_stokes_2d.jl")
include("compressible_navier_stokes_3d.jl")
