
# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
# function create_cache(mesh::UnstructuredMesh, equations::Trixi.AbstractEquations,
#                       dg::Trixi.DG, RealT)
#
#   # extract the elements and interfaces out of the mesh container and into cache
#   cache = (; mesh.elements, mesh.interfaces, mesh.boundaries)
#
#   return cache
# end

# @inline ndofs(mesh::UnstructuredMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)

include("containers_2d.jl")
include("dg_2d.jl")
