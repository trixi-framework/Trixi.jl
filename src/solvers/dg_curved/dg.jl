# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::CurvedMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  elements = init_elements(mesh, equations, dg.basis, uEltype)

  cache = (; elements)

  return cache
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             equations::AbstractEquations, mesh::CurvedMesh, dg::DG)
  boundary_conditions = ntuple(_ -> boundary_condition, 2*ndims(mesh))
  calc_boundary_flux!(cache, u, t, boundary_conditions,
                      equations::AbstractEquations, mesh::CurvedMesh, dg::DG)
end


@inline ndofs(mesh::CurvedMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")
