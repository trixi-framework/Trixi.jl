
# 2D unstructured DG implementation
include("dg_element_geometry_2d.jl")
include("containers_2d.jl")
include("dg_2d.jl")

# TODO: each element, each interface, each boundary wrappers
@inline ndofs(mesh::UnstructuredQuadMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)
