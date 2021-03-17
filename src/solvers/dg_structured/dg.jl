# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::StructuredMesh, equations::AbstractEquations, dg::DG, _)
  elements = init_elements(mesh, equations, dg.basis)

  init_interfaces!(elements, mesh, equations, dg)

  cache = (; elements)

  # Add specialized parts of the cache required to compute the volume integral etc.
  # cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg)...)
  # cache = (;cache..., create_cache(mesh, equations, dg.mortar)...)

  return cache
end


@inline ndofs(mesh::StructuredMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
