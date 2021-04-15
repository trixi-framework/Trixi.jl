
# 2D unstructured DG implementation
include("dg_element_geometry_2d.jl")
include("containers_2d.jl")
include("dg_2d.jl")

@inline ndofs(mesh::UnstructuredQuadMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


@inline function get_one_sided_surface_node_vars(u, equations, solver::DG, j, indices...)
  # There is a cut-off at `n == 10` inside of the method
  # `ntuple(f::F, n::Integer) where F` in Base at ntuple.jl:17
  # in Julia `v1.5`, leading to type instabilities if
  # more than ten variables are used. That's why we use
  # `Val(...)` below.
  u_surface = SVector(ntuple(v -> u[j, v, indices...], Val(nvariables(equations))))
  return u_surface
end
