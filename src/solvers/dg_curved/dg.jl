# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::CurvedMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  elements = init_elements(mesh, equations, dg.basis, uEltype)

  cache = (; elements)

  return cache
end


@inline ndofs(mesh::CurvedMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
