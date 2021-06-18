# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


@inline function get_one_sided_surface_node_vars(u, equations, solver::DG, j, indices...)
  # There is a cut-off at `n == 10` inside of the method
  # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
  # in Julia `v1.5`, leading to type instabilities if
  # more than ten variables are used. That's why we use
  # `Val(...)` below.
  u_surface = SVector(ntuple(v -> u[j, v, indices...], Val(nvariables(equations))))
  return u_surface
end


@inline ndofs(mesh::UnstructuredQuadMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


# 2D unstructured DG implementation
include_fast("mappings_geometry_curved_2d.jl")
include_fast("mappings_geometry_straight_2d.jl")
include_fast("containers_2d.jl")
include_fast("sort_boundary_conditions.jl")
include_fast("dg_2d.jl")


end # @muladd
