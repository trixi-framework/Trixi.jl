# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::StructuredMesh, equations::AbstractEquations, dg::DG, ::Any, ::Any)
  elements = init_elements(mesh, equations, dg.basis)

  cache = (; elements)

  return cache
end


function jacobian_volume(element, mesh::StructuredMesh, cache)
  return inv(cache.elements.inverse_jacobian[element])
end


@inline ndofs(mesh::StructuredMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
