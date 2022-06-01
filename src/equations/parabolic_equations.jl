# Linear scalar diffusion for use in linear scalar advection-diffusion problems
abstract type AbstractLaplaceDiffusionEquations{NDIMS, NVARS} <: AbstractEquationsParabolic{NDIMS, NVARS} end
include("laplace_diffusion_2d.jl")
