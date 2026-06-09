# for any scalar equation, the admissible set is assumed to be u > u_lower, and 
# projection to the admissible set is simply a clipping operation
@inline function project_to_admissible_set(cell_average, lower_bound, variables,
                                           equations::AbstractEquations{NDIMS, 1}) where {NDIMS}
    # lower_bound and cell_average are SVectors of size 1
    return SVector(max(lower_bound[1], cell_average[1]))
end

include("admissible_projection_euler_1d.jl")
include("admissible_projection_euler_2d.jl")
