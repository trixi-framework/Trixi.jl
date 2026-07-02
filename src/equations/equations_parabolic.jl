"""
    gradient_variable_transformation(equations::AbstractEquationsParabolic)

Return the mapping from conservative variables to the variables in which parabolic
gradients are computed. Defaults to [`cons2cons`](@ref), may be specialized to [`cons2prim`](@ref) 
or [`cons2entropy`](@ref) depending on the equation and type of `gradient_variables`. 
"""
gradient_variable_transformation(::AbstractEquationsParabolic) = cons2cons

# By default, the gradients are taken with respect to the conservative variables.
# this is reflected by the type parameter `GradientVariablesConservative` in the abstract
# type `AbstractEquationsParabolic{NDIMS, NVARS, GradientVariablesConservative}`.
struct GradientVariablesConservative end

include("laplace_diffusion.jl")

include("laplace_diffusion_entropy_variables.jl")

include("linear_diffusion_equation.jl")

include("compressible_navier_stokes.jl")

@inline function (boundary_condition::BoundaryConditionDoNothing)(flux_inner,
                                                                  u_inner,
                                                                  orientation::Integer,
                                                                  direction,
                                                                  x, t,
                                                                  operator_type::Gradient,
                                                                  equations::AbstractEquationsParabolic)
    return u_inner
end

@inline function (boundary_condition::BoundaryConditionDoNothing)(flux_inner,
                                                                  u_inner,
                                                                  orientation::Integer,
                                                                  direction,
                                                                  x, t,
                                                                  operator_type::Divergence,
                                                                  equations::AbstractEquationsParabolic)
    return flux_inner
end
