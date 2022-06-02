@doc raw"""
    LaplaceDiffusion2D(diffusivity, equations)

`LaplaceDiffusion2D` represents a scalar diffusion term ``\nabla \cdot (\kappa\nabla u))``
with diffusivity ``\kappa`` applied to each solution component defined by `equations`.
"""
struct LaplaceDiffusion2D{E, N, T} <: AbstractLaplaceDiffusionEquations{2, N}
  diffusivity::T
  equations::E
end

LaplaceDiffusion2D(diffusivity, equations) =
  LaplaceDiffusion2D{typeof(equations), nvariables(equations), typeof(diffusivity)}(diffusivity, equations)

varnames(variable_mapping, equations_parabolic::LaplaceDiffusion2D) =
  varnames(variable_mapping, equations_parabolic.equations)

# no orientation specified since the flux is vector-valued
function flux(u, grad_u, equations::LaplaceDiffusion2D)
  dudx, dudy = grad_u
  return SVector(equations.diffusivity * dudx, equations.diffusivity * dudy)
end

# TODO: should this remain in the equations file, be moved to solvers, or live in the elixir?
# The penalization depends on the solver, but also depends explicitly on physical parameters,
# and would probably need to be specialized for every different equation.
function penalty(u_outer, u_inner, inv_h, equations::LaplaceDiffusion2D, dg::ViscousFormulationLocalDG)
  return dg.penalty_parameter * (u_outer - u_inner) * equations.diffusivity * inv_h
end

# Dirichlet-type boundary condition for use with a parabolic solver in weak form
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, normal::AbstractVector,
                                                                  x, t, operator_type::Gradient,
                                                                  equations::LaplaceDiffusion2D)
  return boundary_condition.boundary_value_function(x, t, equations)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, normal::AbstractVector,
                                                                  x, t, operator_type::Divergence,
                                                                  equations::LaplaceDiffusion2D)
  return u_inner
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, normal::AbstractVector,
                                                                x, t, operator_type::Divergence,
                                                                equations::LaplaceDiffusion2D)
  return boundary_condition.boundary_normal_flux_function(x, t, equations)
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, normal::AbstractVector,
                                                                x, t, operator_type::Gradient,
                                                                equations::LaplaceDiffusion2D)
  return flux_inner
end
