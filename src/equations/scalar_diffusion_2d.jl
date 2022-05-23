struct ScalarDiffusion2D{T} <: AbstractParabolicEquations{2, 1}
  diffusivity::T
end

# no orientation specified since the flux is vector-valued
function flux(u, grad_u, equations::ScalarDiffusion2D)
  dudx, dudy = grad_u
  return equations.diffusivity * dudx, equations.diffusivity * dudy
end

# Dirichlet-type boundary condition for use with a parabolic solver in weak form
@inline function (boundary_condition::BoundaryConditionDirichlet)(u_inner, normal::AbstractVector, x, t,
                                                                  operator_type::Gradient,
                                                                  equations::ScalarDiffusion2D)
  return boundary_condition.boundary_value_function(x, t, equations)
end

@inline function (boundary_condition::BoundaryConditionDirichlet)(flux_inner, normal::AbstractVector, x, t,
                                                                  operator_type::Divergence,
                                                                  equations::ScalarDiffusion2D)
  return flux_inner
end

# `boundary_value_function` should have signature `boundary_value_function(x, t, equations)`
# For Neumann BCs, this corresponds to kappa * (∇u ⋅ n)
struct BoundaryConditionNeumann{B}
  boundary_value_function::B
end

# Neumann type boundary condition for use with a parabolic solver in weak form
@inline function (boundary_condition::BoundaryConditionNeumann)(u_inner, normal::AbstractVector, x, t,
                                                                operator_type::Gradient,
                                                                equations::ScalarDiffusion2D)
  return u_inner
end

@inline function (boundary_condition::BoundaryConditionNeumann)(flux_inner, normal::AbstractVector, x, t,
                                                                operator_type::Divergence,
                                                                equations::ScalarDiffusion2D)
  return boundary_condition.boundary_value_function(x, t, equations)
end