@doc raw"""
    LaplaceDiffusion3D(diffusivity, equations)

`LaplaceDiffusion3D` represents a scalar diffusion term ``\nabla \cdot (\kappa\nabla u))``
with diffusivity ``\kappa`` applied to each solution component defined by `equations`.
"""
struct LaplaceDiffusion3D{E, N, T} <: AbstractLaplaceDiffusion{3, N}
    diffusivity::T
    equations_hyperbolic::E
end

function LaplaceDiffusion3D(diffusivity, equations_hyperbolic)
    LaplaceDiffusion3D{typeof(equations_hyperbolic), nvariables(equations_hyperbolic),
                       typeof(diffusivity)}(diffusivity, equations_hyperbolic)
end

function varnames(variable_mapping, equations_parabolic::LaplaceDiffusion3D)
    varnames(variable_mapping, equations_parabolic.equations_hyperbolic)
end

# no orientation specified since the flux is vector-valued
function flux(u, gradients, orientation::Integer, equations_parabolic::LaplaceDiffusion3D)
    dudx, dudy, dudz = gradients
    if orientation == 1
        return SVector(equations_parabolic.diffusivity * dudx)
    elseif orientation == 2
        return SVector(equations_parabolic.diffusivity * dudy)
    else  # if orientation == 3
        return SVector(equations_parabolic.diffusivity * dudz)
    end
end

# TODO: parabolic; should this remain in the equations file, be moved to solvers, or live in the elixir?
# The penalization depends on the solver, but also depends explicitly on physical parameters,
# and would probably need to be specialized for every different equation.
function penalty(u_outer, u_inner, inv_h, equations_parabolic::LaplaceDiffusion3D,
                 dg::ViscousFormulationLocalDG)
    return dg.penalty_parameter * (u_outer - u_inner) * equations_parabolic.diffusivity
end

# Dirichlet-type boundary condition for use with a parabolic solver in weak form
@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations_parabolic::LaplaceDiffusion3D)
    return boundary_condition.boundary_value_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, u_inner,
                                                                  normal::AbstractVector,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations_parabolic::LaplaceDiffusion3D)
    return flux_inner
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Divergence,
                                                                equations_parabolic::LaplaceDiffusion3D)
    return boundary_condition.boundary_normal_flux_function(x, t, equations_parabolic)
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, u_inner,
                                                                normal::AbstractVector,
                                                                x, t,
                                                                operator_type::Gradient,
                                                                equations_parabolic::LaplaceDiffusion3D)
    return flux_inner
end
