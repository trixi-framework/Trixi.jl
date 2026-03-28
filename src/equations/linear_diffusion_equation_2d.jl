# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    LinearDiffusionEquation2D(diffusivity)

The linear diffusion equation (or heat equation) in two space dimensions with constant
diffusivity `\kappa`:
```math
\partial_t u = \partial_1 \left( \kappa \partial_1 u \right)
             + \partial_2 \left( \kappa \partial_2 u \right)
```
"""
struct LinearDiffusionEquation2D{RealT <: Real} <: AbstractLaplaceDiffusion{2, 1}
    diffusivity::RealT
end

varnames(::typeof(cons2cons), ::LinearDiffusionEquation2D) = ("scalar",)
varnames(::typeof(cons2prim), ::LinearDiffusionEquation2D) = ("scalar",)
varnames(::typeof(cons2entropy), ::LinearDiffusionEquation2D) = ("scalar",)

@inline cons2prim(u, equations::LinearDiffusionEquation2D) = u
@inline cons2entropy(u, equations::LinearDiffusionEquation2D) = u

@inline entropy(u::Real, ::LinearDiffusionEquation2D) = 0.5f0 * u^2
@inline entropy(u, equations::LinearDiffusionEquation2D) = entropy(u[1], equations)

@inline function flux(u, gradients, orientation::Integer,
                      equations::LinearDiffusionEquation2D)
    dudx, dudy = gradients
    if orientation == 1
        return SVector(equations.diffusivity * dudx)
    else # if orientation == 2
        return SVector(equations.diffusivity * dudy)
    end
end
end # @muladd
