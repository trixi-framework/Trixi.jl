
# 2D unstructured DG implementation
include("dg_element_geometry_2d.jl")
include("containers_2d.jl")
include("dg_2d.jl")

@inline ndofs(mesh::UnstructuredQuadMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)
