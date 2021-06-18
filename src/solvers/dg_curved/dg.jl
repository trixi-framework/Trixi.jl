# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::CurvedMesh, equations::AbstractEquations, dg::DG, ::Any, ::Type{uEltype}) where {uEltype<:Real}
  elements = init_elements(mesh, equations, dg.basis, uEltype)

  cache = (; elements)

  # Add specialized parts of the cache required to compute the volume integral etc.
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

  return cache
end

# Extract contravariant vector Ja^i (i = index) as SVector
@inline function get_contravariant_vector(index, contravariant_vectors, indices...)
  SVector(ntuple(@inline(dim -> contravariant_vectors[dim, index, indices...]), Val(ndims(contravariant_vectors) - 3)))
end


@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t, orientation,
                                                  boundary_condition::BoundaryConditionPeriodic,
                                                  mesh::CurvedMesh, equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices, surface_node_indices, element)
  @assert isperiodic(mesh, orientation)
end


@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t, orientation,
                                                  boundary_condition,
                                                  mesh::CurvedMesh, equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices, surface_node_indices, element)
  @unpack node_coordinates, contravariant_vectors, inverse_jacobian = cache.elements
  @unpack surface_flux = surface_integral

  u_inner = get_node_vars(u, equations, dg, node_indices..., element)
  x = get_node_coords(node_coordinates, equations, dg, node_indices..., element)

  # If the mapping is orientation-reversing, the contravariant vectors' orientation
  # is reversed as well. The normal vector must be oriented in the direction
  # from `left_element` to `right_element`, or the numerical flux will be computed
  # incorrectly (downwind direction).
  sign_jacobian = sign(inverse_jacobian[node_indices..., element])

  # Contravariant vector Ja^i is the normal vector
  normal = sign_jacobian * get_contravariant_vector(orientation, contravariant_vectors,
                                                    node_indices..., element)

  # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
  # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
  flux = sign_jacobian * boundary_condition(u_inner, normal, direction, x, t, surface_flux, equations)

  for v in eachvariable(equations)
    surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
  end
end


@inline ndofs(mesh::CurvedMesh, dg::DG, cache) = nelements(cache.elements) * nnodes(dg)^ndims(mesh)


include_fast("containers.jl")
include_fast("dg_1d.jl")
include_fast("dg_2d.jl")
include_fast("dg_3d.jl")


end # @muladd
