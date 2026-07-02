"""
    project_to_admissible_set(cell_average, lower_bound, variables, equations)

For scalar equations, the positivity-preserving limiter enforces `u > u_lower`, and
projection to the admissible set is a clipping operation. 

To ensure that `variables` is consistent with this assumption, users must set 
`variables = (first,)`. 
"""
@inline function project_to_admissible_set(cell_average, lower_bounds,
                                           variables::Tuple{typeof(first)},
                                           equations::AbstractEquations{NDIMS, 1}) where {NDIMS}
    # lower_bound and cell_average are SVectors of size 1
    return SVector(max(lower_bounds[1], cell_average[1]))
end

# use lispy tuple recursion (similar to implementation of limiter_zhang_shu!) to 
# check admissibility in a type-stable way. 
@inline function state_is_admissible(u, lower_bounds::NTuple{N, <:Real},
                                     variables::NTuple{N, Any}, equations) where {N}
    lower_bound = first(lower_bounds)
    variable = first(variables)
    remaining_lower_bounds = Base.tail(lower_bounds)
    remaining_variables = Base.tail(variables)

    satisfies_bound = variable(u, equations) >= lower_bound
    return satisfies_bound &&
           state_is_admissible(u, remaining_lower_bounds, remaining_variables, equations)
end

# terminate recursion
@inline function state_is_admissible(u, lower_bounds::Tuple{},
                                     variables::Tuple{}, equations)
    return true
end

include("admissible_projection_euler_1d.jl")
include("admissible_projection_euler_2d.jl")
