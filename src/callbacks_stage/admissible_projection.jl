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

include("admissible_projection_euler_1d.jl")
include("admissible_projection_euler_2d.jl")
