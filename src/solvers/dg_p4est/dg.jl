# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::P4estMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  # Make sure to balance the p4est before creating any containers
  # in case someone has tampered with the p4est after creating the mesh
  p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, C_NULL)
  # Due to a bug in p4est, the forest needs to be rebalanced twice sometimes
  # See https://github.com/cburstedde/p4est/issues/112
  p4est_balance(mesh.p4est, P4EST_CONNECT_FACE, C_NULL)

  elements   = init_elements(mesh, equations, dg.basis, uEltype)
  interfaces = init_interfaces(mesh, equations, dg.basis, elements)
  boundaries = init_boundaries(mesh, equations, dg.basis, elements)
  mortars    = init_mortars(mesh, equations, dg.basis, elements)

  cache = (; elements, interfaces, boundaries, mortars)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.mortar, uEltype)...)

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


include_fast("containers.jl")
include_fast("dg_2d.jl")


end # @muladd
