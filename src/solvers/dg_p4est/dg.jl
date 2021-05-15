# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::P4estMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  elements = init_elements(mesh, equations, dg.basis, uEltype)
  interfaces = init_interfaces(mesh, equations, dg.basis, elements)

  cache = (; elements, interfaces)

  return cache
end


# Extract outward-pointing normal vector (contravariant vector ±Ja^i, i = index) as SVector
# Note that this vector is not normalized
@inline function get_normal_vector(direction, cache, indices...)
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  # If the mapping is orientation-reversing, the contravariant vectors' orientation
  # is reversed as well
  sign_jacobian = sign(inverse_jacobian[indices...])

  orientation = div(direction + 1, 2)
  normal = sign_jacobian * get_contravariant_vector(orientation, contravariant_vectors, indices...)

  # Contravariant vectors at interfaces in negative coordinate direction are pointing inwards
  # (after normalizing with sign(J) above)
  if direction in (1, 3, 5)
    normal *= -1
  end

  return normal
end


@inline ndofs(mesh::P4estMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include("containers.jl")
include("dg_2d.jl")
