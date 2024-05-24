# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::Union{StructuredMesh, StructuredMeshView},
                      equations::AbstractEquations, dg::DG, ::Any,
                      ::Type{uEltype}) where {uEltype <: Real}
    elements = init_elements(mesh, equations, dg.basis, uEltype)

    cache = (; elements)

    # Add specialized parts of the cache required to compute the volume integral etc.
    cache = (; cache...,
             create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

    return cache
end

# Extract contravariant vector Ja^i (i = index) as SVector
@inline function get_contravariant_vector(index, contravariant_vectors, indices...)
    SVector(ntuple(@inline(dim->contravariant_vectors[dim, index, indices...]),
                   Val(ndims(contravariant_vectors) - 3)))
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionPeriodic,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @assert isperiodic(mesh, orientation)
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
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
    normal = sign_jacobian *
             get_contravariant_vector(orientation, contravariant_vectors,
                                      node_indices..., element)

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian *
           boundary_condition(u_inner, normal, direction, x, t, surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = flux[v]
    end
end

@inline function get_inverse_jacobian(inverse_jacobian,
                                      mesh::Union{StructuredMesh, StructuredMeshView,
                                                  UnstructuredMesh2D, P4estMesh,
                                                  T8codeMesh},
                                      indices...)
    return inverse_jacobian[indices...]
end

include("containers.jl")
include("dg_1d.jl")
include("dg_2d.jl")
include("dg_3d.jl")

include("indicators_1d.jl")
include("indicators_2d.jl")
include("indicators_3d.jl")

include("subcell_limiters_2d.jl")
include("dg_2d_subcell_limiters.jl")

# Specialized implementations used to improve performance
include("dg_2d_compressible_euler.jl")
include("dg_3d_compressible_euler.jl")
end # @muladd
