@doc raw"""
    LaplaceDiffusion1D(diffusivity, equations)

`LaplaceDiffusion1D` represents a scalar diffusion term ``\nabla \cdot (\kappa\nabla u))``
with diffusivity ``\kappa`` applied to each solution component defined by `equations`.
"""
struct LaplaceDiffusion1D{E, N, T} <: AbstractLaplaceDiffusion{1, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusion1D(diffusivity, equations_hyperbolic)
    LaplaceDiffusion1D{typeof(equations_hyperbolic), nvariables(equations_hyperbolic),
                       typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

function varnames(variable_mapping, equations_parabolic::LaplaceDiffusion1D)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer, equations_parabolic::LaplaceDiffusion1D)
    dudx = gradients
    # orientation == 1
    return equations_parabolic.diffusivity * dudx
end

# Dirichlet and Neumann boundary conditions for use with parabolic solvers in weak form.
# Note that these are general, so they apply to LaplaceDiffusion in any spatial dimension.
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations_parabolic::AbstractLaplaceDiffusion)
    return boundary_condition.boundary_value_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations_parabolic::AbstractLaplaceDiffusion)
    return flux_inner
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Divergence,
                                                                equations_parabolic::AbstractLaplaceDiffusion)
    return boundary_condition.boundary_normal_flux_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Gradient,
                                                                equations_parabolic::AbstractLaplaceDiffusion)
    return flux_inner
end
