# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::StructuredMesh{RealT, NDIMS}, equations::AbstractEquations, dg::DG, _) where {RealT, NDIMS}
  elements = init_elements(mesh, equations, dg.basis, RealT)

  init_interfaces!(elements, mesh, equations, dg)

  cache = (; elements)

  # Add specialized parts of the cache required to compute the volume integral etc.
  # cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg)...)
  # cache = (;cache..., create_cache(mesh, equations, dg.mortar)...)

  return cache
end


function allocate_coefficients(mesh::StructuredMesh{RealT, NDIMS}, equations, dg::DG, cache) where {RealT, NDIMS}
  return zeros(real(dg), nvariables(equations), fill(nnodes(dg), NDIMS)..., prod(mesh.size))
end


function wrap_array(u_ode::AbstractArray, mesh::StructuredMesh, equations, dg::DG, cache)
  return u_ode # TODO remove this?
end

@inline ndofs(mesh::StructuredMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")