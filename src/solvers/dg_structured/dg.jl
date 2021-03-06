# function allocate_coefficients(mesh::StructuredMesh, equations, dg::DG, cache)
#     # We must allocate a `Vector` in order to be able to `resize!` it (AMR).
#     # cf. wrap_array
#     zeros(real(dg), nvariables(equations) * nnodes(dg)^ndims(mesh) * prod(size(mesh.elements)))
# end


function allocate_coefficients(mesh::StructuredMesh, equations, dg::DG, cache)
  zeros(real(dg), nvariables(equations), nnodes(dg), mesh.size...)
end


function wrap_array(u_ode::AbstractArray, mesh::StructuredMesh, equations, dg::DG, cache)
  return u_ode # TODO remove this?
end

@inline ndofs(mesh::StructuredMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_1d.jl")