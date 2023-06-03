# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@inline function get_one_sided_surface_node_vars(u, equations, solver::DG, j,
                                                 indices...)
    # There is a cut-off at `n == 10` inside of the method
    # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
    # in Julia `v1.5`, leading to type instabilities if
    # more than ten variables are used. That's why we use
    # `Val(...)` below.
    u_surface = SVector(ntuple(v -> u[j, v, indices...], Val(nvariables(equations))))
    return u_surface
end

# 2D unstructured DG implementation
include("mappings_geometry_curved_2d.jl")
include("mappings_geometry_straight_2d.jl")
include("containers_2d.jl")
include("sort_boundary_conditions.jl")
include("dg_2d.jl")
include("indicators_2d.jl")
end # @muladd
