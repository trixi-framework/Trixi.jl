@doc raw"""
    LaplaceDiffusion1D(diffusivity, equations)

`LaplaceDiffusion1D` represents a scalar diffusion term ``\nabla \cdot (\kappa\nabla u))``
with diffusivity ``\kappa`` applied to each solution component defined by `equations`.
This is intended for use as the parabolic part of a hyperbolic-parabolic system, where the 
hyperbolic part is defined by `equations`. For a purely parabolic diffusion equation 
without any hyperbolic part, see [`LinearDiffusionEquation1D`](@ref).
"""
struct LaplaceDiffusion1D{E, N, T} <: AbstractLaplaceDiffusion{1, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusion1D(diffusivity, equations_hyperbolic)
    return LaplaceDiffusion1D{typeof(equations_hyperbolic),
                              nvariables(equations_hyperbolic),
                              typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

# Together with our specialization of `Adapt.adapt_structure`,
# this allows to move semidiscretizations and their components including
# the equations to GPUs and adapt the floating point type, e.g.,
# to `Float32` to improve performance on GPUs.
function Base.similar(equations::LaplaceDiffusion1D, ::Type{NewRealT}) where {NewRealT}
    diffusivity = equations.diffusivity isa AbstractFloat ?
                  convert(NewRealT, equations.diffusivity) : equations.diffusivity
    return LaplaceDiffusion1D(diffusivity,
                              similar(equations.equations_hyperbolic, NewRealT))
end

function varnames(variable_mapping, equations_parabolic::LaplaceDiffusion1D)
    return varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

function flux(u, gradients, orientation::Integer,
              equations_parabolic::LaplaceDiffusion1D)
    dudx, = gradients # Extract first (and only) component from gradients
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

# Required for the 1D (TreeMesh) case
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  orientation,
                                                                  direction,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations_parabolic::AbstractLaplaceDiffusion)
    return boundary_condition.boundary_value_function(x, t, equations_parabolic)
end

# Required for the 1D (TreeMesh) case
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  orientation,
                                                                  direction,
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

# Required for the 1D (TreeMesh) case
@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                orientation,
                                                                direction,
                                                                x, t,
                                                                operator_type::Divergence,
                                                                equations_parabolic::AbstractLaplaceDiffusion)
    return boundary_condition.boundary_normal_flux_function(x, t, equations_parabolic)
end

# Required for the 1D (TreeMesh) case
@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                orientation,
                                                                direction,
                                                                x, t,
                                                                operator_type::Gradient,
                                                                equations_parabolic::AbstractLaplaceDiffusion)
    return flux_inner
end
