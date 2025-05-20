@doc raw"""
    LaplaceDiffusionEntropyVariables1D(equations)
    LaplaceDiffusionEntropyVariables2D(equations)
    LaplaceDiffusionEntropyVariables3D(equations)

This represent a symmetrized Laplacian diffusion 
``\nabla \cdot (\kappa\frac{\partial u}{\partial w}\nabla w(u)))``, 
where ``w(u)`` denotes the mapping between conservative and entropy variables. 
Compared with `LaplaceDiffusion` (see [`LaplaceDiffusion1D`](@ref),
[`LaplaceDiffusion2D`](@ref), and [`LaplaceDiffusion3D`](@ref)), `LaplaceDiffusionEntropyVariables` is 
guaranteed to dissipate entropy.
"""
struct LaplaceDiffusionEntropyVariables{NDIMS, E, N, T} <:
       AbstractLaplaceDiffusion{NDIMS, N}
    diffusivity::T
    equations_hyperbolic::E
end

function varnames(variable_mapping, equations_parabolic::LaplaceDiffusionEntropyVariables)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

function gradient_variable_transformation(::LaplaceDiffusionEntropyVariables)
    cons2entropy
end

function cons2entropy(u, equations::LaplaceDiffusionEntropyVariables)
    cons2entropy(u, equations.equations_hyperbolic)
end

function entropy2cons(w, equations::LaplaceDiffusionEntropyVariables)
    entropy2cons(w, equations.equations_hyperbolic)
end

# This is used to compute the diffusivity tensor for LaplaceDiffusionEntropyVariables.
# This is the generic fallback using AD (assuming entropy2cons exists)
function jacobian_entropy2cons(w, equations)
    return equations.diffusivity * ForwardDiff.jacobian(w -> entropy2cons(w, equations), w)
end

# Dirichlet and Neumann boundary conditions for use with parabolic solvers in weak form.
# Note that these are general, so they apply to LaplaceDiffusionEntropyVariables in any 
# spatial dimension. 
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations_parabolic::LaplaceDiffusionEntropyVariables)
    return boundary_condition.boundary_value_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations_parabolic::LaplaceDiffusionEntropyVariables)
    return flux_inner
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Divergence,
                                                                equations_parabolic::LaplaceDiffusionEntropyVariables)
    return boundary_condition.boundary_normal_flux_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Gradient,
                                                                equations_parabolic::LaplaceDiffusionEntropyVariables)
    return flux_inner
end
