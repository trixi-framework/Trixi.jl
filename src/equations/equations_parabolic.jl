# specify transformation of conservative variables prior to taking gradients.
# specialize this function to compute gradients e.g., of primitive variables instead of conservative
gradient_variable_transformation(::AbstractEquationsParabolic, dg_parabolic) = cons2cons

# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusionEquations{NDIMS, NVARS} <: AbstractEquationsParabolic{NDIMS, NVARS} end
include("laplace_diffusion_2d.jl")

# Compressible Navier-Stokes equations
abstract type AbstractCompressibleNavierStokesEquations{NDIMS, NVARS} <: AbstractEquationsParabolic{NDIMS, NVARS} end
include("compressible_navier_stokes_2d.jl")
