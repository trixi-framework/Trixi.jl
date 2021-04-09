
include("dg_2d.jl")

# TODO: each element, each interface, each boundary wrappers
@inline ndofs(mesh::UnstructuredQuadMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)
