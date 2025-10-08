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

# Dimension agnostic, i.e., valid for all 1D, 2D, and 3D `StructuredMesh`es.
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::StructuredMesh, equations, surface_integral,
                             dg::DG)
    @assert isperiodic(mesh)
end

function rhs!(backend, du, u, t,
              mesh::Union{StructuredMesh, StructuredMeshView{2}}, equations,
              boundary_conditions, source_terms::Source,
              dg::DG, cache) where {Source}
    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(backend, du, u, mesh,
                              have_nonconservative_terms(equations), equations,
                              dg.volume_integral, dg, cache)
    end

    # Calculate interface and boundary fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache, u, mesh,
                             have_nonconservative_terms(equations), equations,
                             dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux!(cache, u, t, boundary_conditions, mesh, equations,
                            dg.surface_integral, dg)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" apply_jacobian!(backend, du, mesh, equations, dg,
                                                     cache)

    # Calculate source terms
    @trixi_timeit timer() "source terms" begin
        calc_sources!(du, u, t, source_terms, equations, dg, cache)
    end

    return nothing
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionPeriodic,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::False,
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @assert isperiodic(mesh, orientation)
    return nothing
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition::BoundaryConditionPeriodic,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::True,
                                                  equations,
                                                  surface_integral, dg::DG, cache,
                                                  direction, node_indices,
                                                  surface_node_indices, element)
    @assert isperiodic(mesh, orientation)
    return nothing
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::False,
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

    return nothing
end

@inline function calc_boundary_flux_by_direction!(surface_flux_values, u, t,
                                                  orientation,
                                                  boundary_condition,
                                                  mesh::Union{StructuredMesh,
                                                              StructuredMeshView},
                                                  have_nonconservative_terms::True,
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
    flux, noncons_flux = boundary_condition(u_inner, normal, direction, x, t,
                                            surface_flux, equations)

    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_indices..., direction, element] = sign_jacobian *
                                                                              (flux[v] +
                                                                               0.5f0 *
                                                                               noncons_flux[v])
    end

    return nothing
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
