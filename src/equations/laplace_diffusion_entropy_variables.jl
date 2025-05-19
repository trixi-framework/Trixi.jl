@doc raw"""
    LaplaceDiffusionEntropyVariables1D(equations)
    LaplaceDiffusionEntropyVariables2D(equations)
    LaplaceDiffusionEntropyVariables3D(equations)

This represent an artificial viscosity term used to enforce entropy stability
``\nabla \cdot (\epsilon(u)\frac{\partial u}{\partial w}\nabla w(u)))``, 
where `w(u)` denotes the mapping between conservative and entropy variables. 
"""
struct LaplaceDiffusionEntropyVariables{NDIMS, E, N, T} <:
       AbstractLaplaceDiffusion{NDIMS, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusionEntropyVariables1D(diffusivity, equations_hyperbolic)
    LaplaceDiffusionEntropyVariables{1, typeof(equations_hyperbolic),
                                     nvariables(equations_hyperbolic),
                                     typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

function LaplaceDiffusionEntropyVariables2D(diffusivity, equations_hyperbolic)
    LaplaceDiffusionEntropyVariables{2, typeof(equations_hyperbolic),
                                     nvariables(equations_hyperbolic),
                                     typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

function LaplaceDiffusionEntropyVariables3D(diffusivity, equations_hyperbolic)
    LaplaceDiffusionEntropyVariables{3, typeof(equations_hyperbolic),
                                     nvariables(equations_hyperbolic),
                                     typeof(diffusivity)}(diffusivity, equations_hyperbolic)
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

# generic fallback, assuming entropy2cons exists
function jacobian_entropy2cons(w, equations)
    return equations.diffusivity * ForwardDiff.jacobian(w -> entropy2cons(w, equations), w)
end

# Note that here, `u` should be the transformed entropy variables, and 
# not the conservative variables.
function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{1})
    dudx = gradients
    diffusivity = jacobian_entropy2cons(u, equations)
    # if orientation == 1
    return SVector(diffusivity * dudx)
end

function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{2})
    dudx, dudy = gradients
    diffusivity = jacobian_entropy2cons(u, equations)
    if orientation == 1
        return SVector(diffusivity * dudx)
    else # if orientation == 2
        return SVector(diffusivity * dudy)
    end
end

function flux(u, gradients, orientation::Integer,
              equations::LaplaceDiffusionEntropyVariables{3})
    dudx, dudy, dudz = gradients
    diffusivity = jacobian_entropy2cons(u, equations)
    if orientation == 1
        return SVector(diffusivity * dudx)
    elseif orientation == 2
        return SVector(diffusivity * dudy)
    else # if orientation == 3
        return SVector(diffusivity * dudz)
    end
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
