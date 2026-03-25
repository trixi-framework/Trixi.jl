# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearDiffusionEquation1D(diffusivity)

The linear diffusion equation (or heat equation) in one space dimension with constant diffusivity `\kappa`:
```math
\partial_t u = \partial_1 \left( \kappa \partial_1 u \right)
```
"""
struct LinearDiffusionEquation1D{RealT <: Real} <: AbstractLaplaceDiffusion{1, 1}
    diffusivity::RealT
end

varnames(::typeof(cons2cons), ::LinearDiffusionEquation1D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearDiffusionEquation1D) = ("scalar",)
varnames(::typeof(cons2entropy), ::LinearDiffusionEquation1D) = ("scalar",)

@inline cons2prim(u, equations::LinearDiffusionEquation1D) = u
@inline cons2entropy(u, equations::LinearDiffusionEquation1D) = u

@inline entropy(u::Real, ::LinearDiffusionEquation1D) = 0.5f0 * u^2
@inline entropy(u, equations::LinearDiffusionEquation1D) = entropy(u[1], equations)

@inline function flux(u, gradients, orientation::Integer,
                      equations::LinearDiffusionEquation1D)
    dudx, = gradients
    # orientation == 1
    return equations.diffusivity * dudx
end
end # @muladd
