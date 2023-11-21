@doc raw"""
    LaplaceDiffusionEquations1D(diffusivity, equations)

`LaplaceDiffusionEquations1D` represents a scalar diffusion term ``\nabla \cdot (\kappa\nabla u))``
with diffusivity ``\kappa`` applied to each solution component defined by `equations`.
"""
struct LaplaceDiffusionEquations1D{E, N, T} <: AbstractLaplaceDiffusionEquations{1, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusionEquations1D(diffusivity, equations_hyperbolic)
    LaplaceDiffusionEquations1D{typeof(equations_hyperbolic),
                                nvariables(equations_hyperbolic),
                                typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

function varnames(variable_mapping, equations_parabolic::LaplaceDiffusionEquations1D)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations_parabolic::LaplaceDiffusionEquations1D)
    dudx = gradients
    # orientation == 1
    return equations_parabolic.diffusivity * dudx
end

# Dirichlet-type boundary condition for use with a parabolic solver in weak form
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations_parabolic::LaplaceDiffusionEquations1D)
    return boundary_condition.boundary_value_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations_parabolic::LaplaceDiffusionEquations1D)
    return flux_inner
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Divergence,
                                                                equations_parabolic::LaplaceDiffusionEquations1D)
    return boundary_condition.boundary_normal_flux_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Gradient,
                                                                equations_parabolic::LaplaceDiffusionEquations1D)
    return flux_inner
end
