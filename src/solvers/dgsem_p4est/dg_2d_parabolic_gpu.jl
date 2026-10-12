# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# KernelAbstractions.jl implementation of the parabolic right-hand side for the
# `P4estMesh`. This mirrors `rhs_parabolic!(backend::Nothing, ...)` from
# `dg_2d_parabolic.jl`, but parallelizes over the individual nodes (volume
# terms) or surface nodes (interface and boundary terms) instead of over the
# elements, interfaces, and boundaries, following the structure of the
# hyperbolic implementation in `dg_2d_gpu.jl`: the functions launching the
# kernels are generic in the number of spatial dimensions, while the kernels
# and the `per_node!` functions they call dispatch on the mesh type `MeshT`.
# To reduce the number of kernel launches, some steps of the CPU implementation
# are fused:
# - Volume integrals compute their result per node and overwrite their output,
#   so no separate reset of `gradients` or `du` is required.
# - Prolongation to interfaces/boundaries is fused with the computation of the
#   interface/boundary fluxes, i.e., `cache.interfaces.u` and `cache.boundaries.u`
#   are not used.
# - Surface integrals are fused with the application of the Jacobian (and the
#   parabolic source terms).
#
# Currently, only conforming meshes are supported (no mortars).
function rhs_parabolic!(backend::Backend, du, u, t,
                        mesh::P4estMesh{2},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms_parabolic,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack parabolic_container = cache_parabolic
    @unpack u_transformed, gradients, flux_parabolic = parabolic_container

    # Convert conservative variables to a form more suitable for parabolic flux calculations
    @trixi_timeit_ext backend timer() "transform variables" begin
        transform_variables!(backend, u_transformed, u, mesh, equations_parabolic,
                             dg, cache)
    end

    # Compute the gradients of the transformed variables
    @trixi_timeit_ext backend timer() "calculate gradient" begin
        calc_gradient!(backend, gradients, u_transformed, t, mesh,
                       equations_parabolic, boundary_conditions_parabolic,
                       dg, parabolic_scheme, cache)
    end

    # Compute and store the parabolic fluxes
    @trixi_timeit_ext backend timer() "calculate parabolic fluxes" begin
        calc_parabolic_fluxes!(backend, flux_parabolic, gradients, u_transformed,
                               mesh, equations_parabolic, dg, cache)
    end

    # Calculate volume integral of the divergence of the parabolic fluxes.
    # This overwrites `du`, so no reset is required.
    @trixi_timeit_ext backend timer() "volume integral" begin
        calc_volume_integral!(backend, du, flux_parabolic, mesh,
                              equations_parabolic, dg, cache)
    end

    # Prolong the normal parabolic fluxes to the interfaces and calculate the
    # interface fluxes
    @trixi_timeit_ext backend timer() "prolong2interfaces + flux" begin
        prolong2interfaces_and_calc_interface_flux!(backend,
                                                    cache.elements.surface_flux_values,
                                                    flux_parabolic, mesh,
                                                    equations_parabolic,
                                                    dg, parabolic_scheme,
                                                    cache)
    end

    # Prolong the normal parabolic fluxes to the boundaries and calculate the
    # boundary fluxes
    @trixi_timeit_ext backend timer() "prolong2boundaries + flux" begin
        calc_boundary_flux_divergence!(backend, cache, t, flux_parabolic,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic, dg)
    end

    # Mortars are not yet supported on GPU backends
    @trixi_timeit_ext backend timer() "mortar flux" begin
        @assert isempty(eachmortar(dg, cache))
    end

    # Calculate surface integrals, apply the Jacobian from the mapping to the
    # reference element, and add the parabolic source terms
    @trixi_timeit_ext backend timer() "surface, Jacobian + source terms" begin
        calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend,
                                                                   du, u,
                                                                   gradients,
                                                                   t,
                                                                   source_terms_parabolic,
                                                                   mesh,
                                                                   equations_parabolic,
                                                                   dg.surface_integral,
                                                                   dg, cache)
    end

    return nothing
end

function calc_gradient!(backend::Backend, gradients, u_transformed, t,
                        mesh::P4estMesh{2},
                        equations_parabolic, boundary_conditions_parabolic,
                        dg::DG, parabolic_scheme, cache)
    # Calculate volume integral. This overwrites `gradients`, so no reset is required.
    @trixi_timeit_ext backend timer() "volume integral" begin
        calc_volume_integral_gradient!(backend, gradients, u_transformed,
                                       mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces and calculate interface fluxes
    @trixi_timeit_ext backend timer() "prolong2interfaces + flux" begin
        prolong2interfaces_and_calc_interface_flux_gradient!(backend,
                                                             cache.elements.surface_flux_values,
                                                             u_transformed, mesh,
                                                             equations_parabolic,
                                                             dg, parabolic_scheme,
                                                             cache)
    end

    # Prolong solution to boundaries and calculate boundary fluxes
    @trixi_timeit_ext backend timer() "prolong2boundaries + flux" begin
        calc_boundary_flux_gradient!(backend, cache, t, u_transformed,
                                     boundary_conditions_parabolic, mesh,
                                     equations_parabolic, dg)
    end

    # Mortars are not yet supported on GPU backends
    @trixi_timeit_ext backend timer() "mortar flux" begin
        @assert isempty(eachmortar(dg, cache))
    end

    # Calculate surface integrals and apply Jacobian from mapping to reference element
    @trixi_timeit_ext backend timer() "surface + Jacobian" begin
        calc_surface_integral_and_apply_jacobian_gradient!(backend, gradients, mesh,
                                                           equations_parabolic, dg,
                                                           cache)
    end

    return nothing
end

# The volume kernels parallelize over the volume nodes of all elements. The
# surface kernels parallelize over the surface nodes of all interfaces or
# boundaries, which are indexed in the same way, so that the
# `boundary_node_ndrange` of the hyperbolic GPU code is reused for interfaces.
@inline function volume_node_ndrange(mesh::P4estMesh, dg)
    return ntuple(_ -> nnodes(dg), Val(ndims(mesh)))
end

###############################################################################
# Volume terms

function transform_variables!(backend::Backend, u_transformed, u,
                              mesh::P4estMesh,
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u, mesh, equations_parabolic, dg, cache)
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
    end
    kernel! = transform_variables_KAkernel!(backend)
    kernel!(u_transformed, u, typeof(mesh), equations_parabolic, dg,
            ndrange = (volume_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

@kernel inbounds=true function transform_variables_KAkernel!(u_transformed, u,
                                                             MeshT::Type{<:P4estMesh{2}},
                                                             equations_parabolic,
                                                             dg)
    i, j, element = @index(Global, NTuple)
    transformation = gradient_variable_transformation(equations_parabolic)
    u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)
    u_transformed_node = transformation(u_node, equations_parabolic)
    set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg,
                   i, j, element)
end

function calc_volume_integral_gradient!(backend::Backend, gradients, u_transformed,
                                        mesh::P4estMesh,
                                        equations_parabolic::AbstractEquationsParabolic,
                                        dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack derivative_hat = dg.basis
    @unpack contravariant_vectors = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
        for gradient in gradients
            check_axes(gradient, mesh, equations_parabolic, dg, cache)
        end
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end
    kernel! = calc_volume_integral_gradient_KAkernel!(backend)
    kernel!(gradients, u_transformed, typeof(mesh), equations_parabolic, dg,
            derivative_hat, contravariant_vectors,
            ndrange = (volume_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

# In contrast to the CPU version, this computes the complete volume integral of
# a node at once and overwrites `gradients`.
@kernel inbounds=true function calc_volume_integral_gradient_KAkernel!(gradients,
                                                                       u_transformed,
                                                                       MeshT::Type{<:P4estMesh{2}},
                                                                       equations_parabolic,
                                                                       dg,
                                                                       derivative_hat,
                                                                       contravariant_vectors)
    i, j, element = @index(Global, NTuple)
    gradients_x, gradients_y = gradients

    # Gradients with respect to the reference coordinates
    gradients_reference_1 = zero(get_node_vars(u_transformed, equations_parabolic,
                                               dg, i, j, element))
    gradients_reference_2 = gradients_reference_1
    for ii in eachnode(dg)
        gradients_reference_1 = gradients_reference_1 +
                                derivative_hat[i, ii] *
                                get_node_vars(u_transformed, equations_parabolic,
                                              dg, ii, j, element)
    end
    for jj in eachnode(dg)
        gradients_reference_2 = gradients_reference_2 +
                                derivative_hat[j, jj] *
                                get_node_vars(u_transformed, equations_parabolic,
                                              dg, i, jj, element)
    end

    # Transform the reference coordinate gradients to physical gradients using
    # the contravariant vectors. Note that the contravariant vectors are
    # transposed compared with computations of flux divergences in
    # `calc_volume_integral!`. See
    # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190
    # for a more detailed discussion.
    Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    gradient_x_node = Ja11 * gradients_reference_1 + Ja21 * gradients_reference_2
    gradient_y_node = Ja12 * gradients_reference_1 + Ja22 * gradients_reference_2

    set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                   i, j, element)
    set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                   i, j, element)
end

function calc_parabolic_fluxes!(backend::Backend, flux_parabolic, gradients,
                                u_transformed, mesh::P4estMesh,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
        for gradient in gradients
            check_axes(gradient, mesh, equations_parabolic, dg, cache)
        end
        for flux_parabolic_i in flux_parabolic
            check_axes(flux_parabolic_i, mesh, equations_parabolic, dg, cache)
        end
    end
    kernel! = calc_parabolic_fluxes_KAkernel!(backend)
    kernel!(flux_parabolic, gradients, u_transformed, typeof(mesh),
            equations_parabolic, dg,
            ndrange = (volume_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

@kernel inbounds=true function calc_parabolic_fluxes_KAkernel!(flux_parabolic,
                                                               gradients,
                                                               u_transformed,
                                                               MeshT::Type{<:P4estMesh{2}},
                                                               equations_parabolic,
                                                               dg)
    i, j, element = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    gradients_x, gradients_y = gradients

    u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, element)
    gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                     i, j, element)
    gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                     i, j, element)

    flux_parabolic_node_x = flux(u_node, (gradients_1_node, gradients_2_node),
                                 1, equations_parabolic)
    flux_parabolic_node_y = flux(u_node, (gradients_1_node, gradients_2_node),
                                 2, equations_parabolic)
    set_node_vars!(flux_parabolic_x, flux_parabolic_node_x, equations_parabolic,
                   dg, i, j, element)
    set_node_vars!(flux_parabolic_y, flux_parabolic_node_y, equations_parabolic,
                   dg, i, j, element)
end

# This is the version used when calculating the divergence of the parabolic fluxes.
# Identical to the weak-form volume integral for the purely hyperbolic case,
# except that the fluxes are here already precomputed in `calc_parabolic_fluxes!`.
function calc_volume_integral!(backend::Backend, du, flux_parabolic::Tuple,
                               mesh::P4estMesh,
                               equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack derivative_hat = dg.basis
    @unpack contravariant_vectors = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(du, mesh, equations_parabolic, dg, cache)
        for flux_parabolic_i in flux_parabolic
            check_axes(flux_parabolic_i, mesh, equations_parabolic, dg, cache)
        end
        check_axes(cache.elements, equations_parabolic, dg, cache)
    end
    kernel! = calc_volume_integral_KAkernel!(backend)
    kernel!(du, flux_parabolic, typeof(mesh), equations_parabolic, dg,
            derivative_hat, contravariant_vectors,
            ndrange = (volume_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

# In contrast to the CPU version, this computes the complete volume integral of
# a node at once and overwrites `du`.
@kernel inbounds=true function calc_volume_integral_KAkernel!(du, flux_parabolic,
                                                              MeshT::Type{<:P4estMesh{2}},
                                                              equations_parabolic,
                                                              dg,
                                                              derivative_hat,
                                                              contravariant_vectors)
    i, j, element = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y = flux_parabolic

    du_node = zero(get_node_vars(du, equations_parabolic, dg, i, j, element))
    for ii in eachnode(dg)
        flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                              ii, j, element)
        flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                              ii, j, element)

        # Compute the contravariant flux by taking the scalar product of the
        # first contravariant vector Ja^1 and the flux vector
        Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors,
                                              ii, j, element)
        contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
        du_node = du_node + derivative_hat[i, ii] * contravariant_flux1
    end
    for jj in eachnode(dg)
        flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                              i, jj, element)
        flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                              i, jj, element)

        # Compute the contravariant flux by taking the scalar product of the
        # second contravariant vector Ja^2 and the flux vector
        Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors,
                                              i, jj, element)
        contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
        du_node = du_node + derivative_hat[j, jj] * contravariant_flux2
    end

    set_node_vars!(du, du_node, equations_parabolic, dg, i, j, element)
end

###############################################################################
# Interface terms

# This is the version used when calculating the gradient of the parabolic fluxes.
# It fuses `prolong2interfaces!` and `calc_interface_flux_gradient!` of the CPU
# version.
function prolong2interfaces_and_calc_interface_flux_gradient!(backend::Backend,
                                                              surface_flux_values,
                                                              u_transformed,
                                                              mesh::P4estMesh,
                                                              equations_parabolic,
                                                              dg::DG,
                                                              parabolic_scheme, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    ninterfaces(cache.interfaces) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
        check_axes(cache.interfaces, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh,
                                       equations_parabolic, dg, cache)
    end
    index_range = eachnode(dg)
    kernel! = prolong2interfaces_and_calc_interface_flux_gradient_KAkernel!(backend)
    kernel!(surface_flux_values, u_transformed, typeof(mesh), equations_parabolic,
            dg, parabolic_scheme, neighbor_ids, node_indices, contravariant_vectors,
            index_range,
            ndrange = (boundary_node_ndrange(mesh, dg)...,
                       ninterfaces(cache.interfaces)))
    return nothing
end

@kernel inbounds=true function prolong2interfaces_and_calc_interface_flux_gradient_KAkernel!(surface_flux_values,
                                                                                             u_transformed,
                                                                                             MeshT::Type{<:P4estMesh{2}},
                                                                                             equations_parabolic,
                                                                                             dg,
                                                                                             parabolic_scheme,
                                                                                             neighbor_ids,
                                                                                             node_indices,
                                                                                             contravariant_vectors,
                                                                                             index_range)
    i, interface = @index(Global, NTuple)
    prolong2interfaces_and_calc_interface_flux_gradient_per_node!(surface_flux_values,
                                                                  u_transformed,
                                                                  MeshT,
                                                                  equations_parabolic,
                                                                  dg,
                                                                  parabolic_scheme,
                                                                  neighbor_ids,
                                                                  node_indices,
                                                                  contravariant_vectors,
                                                                  index_range, i,
                                                                  interface)
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_gradient_per_node!(surface_flux_values,
                                                                                                u_transformed,
                                                                                                MeshT::Type{<:P4estMesh{2}},
                                                                                                equations_parabolic,
                                                                                                dg,
                                                                                                parabolic_scheme,
                                                                                                neighbor_ids,
                                                                                                node_indices,
                                                                                                contravariant_vectors,
                                                                                                index_range,
                                                                                                i,
                                                                                                interface)
    # This reuses the hyperbolic version, which also returns the outward normal
    # direction on the primary element
    u_ll, u_rr, normal_direction, primary_direction, primary_element,
    i_secondary, secondary_direction, secondary_element = get_interface_values(u_transformed,
                                                                               equations_parabolic,
                                                                               dg,
                                                                               neighbor_ids,
                                                                               node_indices,
                                                                               contravariant_vectors,
                                                                               index_range,
                                                                               i,
                                                                               interface)

    flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                           equations_parabolic, parabolic_scheme)

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
        # No sign flip required for gradient calculation because for parabolic terms,
        # the normals are not embedded in `flux_` for gradient computations.
        surface_flux_values[v, i_secondary, secondary_direction,
        secondary_element] = flux_[v]
    end
    return nothing
end

# This is the version used when calculating the divergence of the parabolic fluxes.
# It fuses `prolong2interfaces!` and `calc_interface_flux!` of the CPU version.
function prolong2interfaces_and_calc_interface_flux!(backend::Backend,
                                                     surface_flux_values,
                                                     flux_parabolic::Tuple,
                                                     mesh::P4estMesh,
                                                     equations_parabolic::AbstractEquationsParabolic,
                                                     dg::DG, parabolic_scheme, cache)
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    ninterfaces(cache.interfaces) == 0 && return nothing
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        for flux_parabolic_i in flux_parabolic
            check_axes(flux_parabolic_i, mesh, equations_parabolic, dg, cache)
        end
        check_axes(cache.interfaces, equations_parabolic, dg, cache)
        check_axes(cache.elements, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh,
                                       equations_parabolic, dg, cache)
    end
    index_range = eachnode(dg)
    kernel! = prolong2interfaces_and_calc_interface_flux_KAkernel!(backend)
    kernel!(surface_flux_values, flux_parabolic, typeof(mesh), equations_parabolic,
            dg, parabolic_scheme, neighbor_ids, node_indices, contravariant_vectors,
            index_range,
            ndrange = (boundary_node_ndrange(mesh, dg)...,
                       ninterfaces(cache.interfaces)))
    return nothing
end

@kernel inbounds=true function prolong2interfaces_and_calc_interface_flux_KAkernel!(surface_flux_values,
                                                                                    flux_parabolic,
                                                                                    MeshT::Type{<:P4estMesh{2}},
                                                                                    equations_parabolic,
                                                                                    dg,
                                                                                    parabolic_scheme,
                                                                                    neighbor_ids,
                                                                                    node_indices,
                                                                                    contravariant_vectors,
                                                                                    index_range)
    i, interface = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                         flux_parabolic_x,
                                                         flux_parabolic_y, MeshT,
                                                         equations_parabolic, dg,
                                                         parabolic_scheme,
                                                         neighbor_ids, node_indices,
                                                         contravariant_vectors,
                                                         index_range, i, interface)
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                                       flux_parabolic_x,
                                                                                       flux_parabolic_y,
                                                                                       MeshT::Type{<:P4estMesh{2}},
                                                                                       equations_parabolic,
                                                                                       dg,
                                                                                       parabolic_scheme,
                                                                                       neighbor_ids,
                                                                                       node_indices,
                                                                                       contravariant_vectors,
                                                                                       index_range,
                                                                                       i,
                                                                                       interface)
    (primary_direction, primary_element, i_primary, j_primary,
    i_secondary, secondary_direction, secondary_element,
    i_secondary_node, j_secondary_node) = get_interface_indices(neighbor_ids,
                                                                node_indices,
                                                                index_range, i,
                                                                interface)

    # Outward normal direction on the primary element
    normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                                            i_primary, j_primary, primary_element)

    # Parabolic flux of the primary element dotted with its outward normal direction
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                          i_primary, j_primary, primary_element)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                          i_primary, j_primary, primary_element)
    parabolic_flux_normal_ll = normal_direction[1] * flux1 + normal_direction[2] * flux2

    # Parabolic flux of the secondary element dotted with its outward normal
    # direction. Here, we assume that the normal direction on the secondary
    # element is the negative of the normal direction on the primary element,
    # and store the normal flux with respect to the primary normal direction.
    normal_direction_secondary = get_normal_direction(secondary_direction,
                                                      contravariant_vectors,
                                                      i_secondary_node,
                                                      j_secondary_node,
                                                      secondary_element)
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                          i_secondary_node, j_secondary_node, secondary_element)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                          i_secondary_node, j_secondary_node, secondary_element)
    parabolic_flux_normal_rr = -(normal_direction_secondary[1] * flux1 +
                                 normal_direction_secondary[2] * flux2)

    flux_ = flux_parabolic(parabolic_flux_normal_ll, parabolic_flux_normal_rr,
                           normal_direction, Divergence(),
                           equations_parabolic, parabolic_scheme)

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
        # Sign flip required for divergence calculation since the divergence
        # interface flux involves the normal direction.
        surface_flux_values[v, i_secondary, secondary_direction,
        secondary_element] = -flux_[v]
    end
    return nothing
end

###############################################################################
# Boundary terms
#
# The prolongation of the (transformed) solution or of the normal parabolic
# fluxes to the boundaries is fused with the boundary flux computation. As in
# the hyperbolic GPU code, one kernel is launched per boundary condition type.

function calc_boundary_flux_gradient!(backend::Backend, cache, t, u_transformed,
                                      boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                      mesh::P4estMesh, equations_parabolic, dg::DG)
    @assert isempty(eachboundary(dg, cache))
    return nothing
end

function calc_boundary_flux_divergence!(backend::Backend, cache, t, flux_parabolic,
                                        boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::P4estMesh, equations_parabolic, dg::DG)
    @assert isempty(eachboundary(dg, cache))
    return nothing
end

function calc_boundary_flux_gradient!(backend::Backend, cache, t, u_transformed,
                                      boundary_conditions_parabolic::UnstructuredSortedBoundaryTypes,
                                      mesh::P4estMesh, equations_parabolic, dg::DG)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions_parabolic
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(u_transformed, mesh, equations_parabolic, dg, cache)
    end
    calc_boundary_flux_by_type!(backend, cache, t, u_transformed,
                                boundary_condition_types, boundary_indices,
                                Gradient(), mesh, equations_parabolic, dg)
    return nothing
end

function calc_boundary_flux_divergence!(backend::Backend, cache, t, flux_parabolic,
                                        boundary_conditions_parabolic::UnstructuredSortedBoundaryTypes,
                                        mesh::P4estMesh, equations_parabolic, dg::DG)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions_parabolic
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        for flux_parabolic_i in flux_parabolic
            check_axes(flux_parabolic_i, mesh, equations_parabolic, dg, cache)
        end
    end
    calc_boundary_flux_by_type!(backend, cache, t, flux_parabolic,
                                boundary_condition_types, boundary_indices,
                                Divergence(), mesh, equations_parabolic, dg)
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming", launching one kernel
# per boundary condition type. Thus, the kernels are specialized on the
# boundary condition and do not need any dynamic dispatch.
function calc_boundary_flux_by_type!(backend::Backend, cache, t, u_or_flux,
                                     BCs::Tuple{}, BC_indices::Tuple{},
                                     operator_type, mesh::P4estMesh,
                                     equations_parabolic, dg::DG)
    return nothing
end

function calc_boundary_flux_by_type!(backend::Backend, cache, t, u_or_flux,
                                     BCs::Tuple{Any, Vararg{Any}},
                                     BC_indices::Tuple{AbstractVector{Int},
                                                       Vararg{AbstractVector{Int}}},
                                     operator_type, mesh::P4estMesh,
                                     equations_parabolic, dg::DG)
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)

    n_boundaries = length(boundary_condition_indices)
    if n_boundaries > 0
        @unpack neighbor_ids, node_indices = cache.boundaries
        @unpack node_coordinates, contravariant_vectors, surface_flux_values = cache.elements
        # Explicit bounds check, which allows us to assume inbounds access in the kernel
        @boundscheck begin
            check_axes(cache.boundaries, equations_parabolic, dg, cache)
            check_axes(cache.elements, equations_parabolic, dg, cache)
            check_axes_surface_flux_values(surface_flux_values, mesh,
                                           equations_parabolic, dg, cache)
        end
        index_range = eachnode(dg)
        kernel! = calc_boundary_flux_KAkernel!(backend)
        kernel!(surface_flux_values, u_or_flux, boundary_condition_indices,
                neighbor_ids, node_indices, t, boundary_condition, operator_type,
                index_range, typeof(mesh), equations_parabolic, dg,
                node_coordinates, contravariant_vectors,
                ndrange = (boundary_node_ndrange(mesh, dg)..., n_boundaries))
    end

    # Recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(backend, cache, t, u_or_flux,
                                Base.tail(BCs), Base.tail(BC_indices),
                                operator_type, mesh, equations_parabolic, dg)
    return nothing
end

@kernel inbounds=true function calc_boundary_flux_KAkernel!(surface_flux_values,
                                                            u_or_flux,
                                                            boundary_condition_indices,
                                                            neighbor_ids,
                                                            node_indices_arr,
                                                            t,
                                                            boundary_condition::BC,
                                                            operator_type,
                                                            index_range,
                                                            MeshT::Type{<:P4estMesh{2}},
                                                            equations_parabolic,
                                                            dg,
                                                            node_coordinates,
                                                            contravariant_vectors) where {BC}
    i, local_index = @index(Global, NTuple)
    # Use the local index to get the global boundary index from the pre-sorted list
    boundary = boundary_condition_indices[local_index]

    calc_boundary_flux_per_node!(surface_flux_values, u_or_flux, t,
                                 boundary_condition, operator_type, MeshT,
                                 equations_parabolic, dg, boundary, neighbor_ids,
                                 node_indices_arr, index_range, node_coordinates,
                                 contravariant_vectors, i)
end

Base.@propagate_inbounds function calc_boundary_flux_per_node!(surface_flux_values,
                                                               u_or_flux, t,
                                                               boundary_condition,
                                                               operator_type,
                                                               MeshT::Type{<:P4estMesh{2}},
                                                               equations_parabolic,
                                                               dg, boundary,
                                                               neighbor_ids,
                                                               node_indices_arr,
                                                               index_range,
                                                               node_coordinates,
                                                               contravariant_vectors,
                                                               i)
    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element = neighbor_ids[boundary]
    node_indices = node_indices_arr[boundary]
    direction = indices2direction(node_indices)

    i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

    i_node = delayed_index_2d(i_node_start, i_node_step, i)
    j_node = delayed_index_2d(j_node_start, j_node_step, i)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction, contravariant_vectors,
                                            i_node, j_node, element)

    # Prolong the inner value to the boundary node, which is stored in
    # `boundaries.u` in the CPU version
    u_inner = boundary_inner_value_parabolic(operator_type, u_or_flux,
                                             normal_direction, equations_parabolic,
                                             dg, i_node, j_node, element)

    # This assumes the gradient numerical flux at the boundary is the gradient variable,
    # which is consistent with BR1, LDG.
    flux_inner = u_inner

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations_parabolic, dg,
                        i_node, j_node, element)

    flux_ = boundary_condition(flux_inner, u_inner, normal_direction, x, t,
                               operator_type, equations_parabolic)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, direction, element] = flux_[v]
    end
    return nothing
end

# Inner values at the boundary node passed to the boundary condition:
# the (transformed) solution for the gradient computation ...
Base.@propagate_inbounds function boundary_inner_value_parabolic(::Gradient,
                                                                 u_transformed,
                                                                 normal_direction,
                                                                 equations_parabolic,
                                                                 dg, indices...)
    return get_node_vars(u_transformed, equations_parabolic, dg, indices...)
end

# ... and the parabolic flux dotted with the outward normal direction for the
# divergence computation
Base.@propagate_inbounds function boundary_inner_value_parabolic(::Divergence,
                                                                 flux_parabolic::NTuple{2,
                                                                                        Any},
                                                                 normal_direction,
                                                                 equations_parabolic,
                                                                 dg, indices...)
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg, indices...)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg, indices...)
    return normal_direction[1] * flux1 + normal_direction[2] * flux2
end

###############################################################################
# Surface integrals, Jacobian, and source terms

# This fuses `calc_surface_integral_gradient!` and `apply_jacobian_parabolic!`
# of the CPU version
function calc_surface_integral_and_apply_jacobian_gradient!(backend::Backend,
                                                            gradients,
                                                            mesh::P4estMesh,
                                                            equations_parabolic::AbstractEquationsParabolic,
                                                            dg::DGSEM{<:LobattoLegendreBasis},
                                                            cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, contravariant_vectors, inverse_jacobian = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        for gradient in gradients
            check_axes(gradient, mesh, equations_parabolic, dg, cache)
        end
        check_axes(cache.elements, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh,
                                       equations_parabolic, dg, cache)
    end
    NNODES = nnodes(dg)
    kernel! = calc_surface_integral_and_apply_jacobian_gradient_KAkernel!(backend)
    kernel!(gradients, typeof(mesh), equations_parabolic, inverse_weights[1],
            Val(NNODES), surface_flux_values, dg, contravariant_vectors,
            inverse_jacobian,
            ndrange = (volume_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

@kernel inbounds=true function calc_surface_integral_and_apply_jacobian_gradient_KAkernel!(gradients,
                                                                                           MeshT::Type{<:P4estMesh{2}},
                                                                                           equations_parabolic::AbstractEquationsParabolic{2},
                                                                                           factor,
                                                                                           ::Val{NNODES},
                                                                                           surface_flux_values,
                                                                                           dg::DGSEM,
                                                                                           contravariant_vectors,
                                                                                           inverse_jacobian) where {
                                                                                                                    NNODES
                                                                                                                    }
    i, j, element = @index(Global, NTuple)
    gradients_x, gradients_y = gradients
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # As in the hyperbolic kernel, we avoid branching on the position of the
    # node and select the surface flux values of the faces in x and y direction
    # with `ifelse` instead (zero for interior nodes).
    #
    # factor = inverse_weights[1]
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    x_node_interface = (i == 1) | (i == NNODES)
    y_node_interface = (j == 1) | (j == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    _zero = zero(eltype(gradients_x))
    surface_flux_x_face = SVector(ntuple(@inline(v->ifelse(x_node_interface,
                                                           @inbounds(surface_flux_values[v,
                                                                                         j,
                                                                                         x_face,
                                                                                         element]),
                                                           _zero)),
                                         Val(nvariables(equations_parabolic))))
    surface_flux_y_face = SVector(ntuple(@inline(v->ifelse(y_node_interface,
                                                           @inbounds(surface_flux_values[v,
                                                                                         i,
                                                                                         y_face,
                                                                                         element]),
                                                           _zero)),
                                         Val(nvariables(equations_parabolic))))

    # The surface contribution to the x-component of the gradients is the
    # surface flux times the x-component of the outward normal direction of
    # the face (and similarly for the y-component). The normal directions are
    # only used at surface nodes, where the surface fluxes are nonzero.
    normal_direction_x_face = get_normal_direction(x_face, contravariant_vectors,
                                                   i, j, element)
    normal_direction_y_face = get_normal_direction(y_face, contravariant_vectors,
                                                   i, j, element)
    surface_node_x = surface_flux_x_face * normal_direction_x_face[1] +
                     surface_flux_y_face * normal_direction_y_face[1]
    surface_node_y = surface_flux_x_face * normal_direction_x_face[2] +
                     surface_flux_y_face * normal_direction_y_face[2]

    # Apply Jacobian from mapping to reference element. In contrast to the
    # hyperbolic part, the sign of the inverse Jacobian is not flipped.
    jacobian_factor = inverse_jacobian[i, j, element]
    gradient_x_node = jacobian_factor *
                      (get_node_vars(gradients_x, equations_parabolic, dg,
                                     i, j, element) + factor * surface_node_x)
    gradient_y_node = jacobian_factor *
                      (get_node_vars(gradients_y, equations_parabolic, dg,
                                     i, j, element) + factor * surface_node_y)
    set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                   i, j, element)
    set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                   i, j, element)
end

# This fuses `calc_surface_integral!`, `apply_jacobian_parabolic!`, and
# `calc_sources_parabolic!` of the CPU version
function calc_surface_integral_and_apply_jacobian_and_calc_sources!(backend::Backend,
                                                                    du, u,
                                                                    gradients,
                                                                    t,
                                                                    source_terms_parabolic,
                                                                    mesh::P4estMesh,
                                                                    equations_parabolic::AbstractEquationsParabolic,
                                                                    surface_integral::SurfaceIntegralWeakForm,
                                                                    dg::DGSEM{<:LobattoLegendreBasis},
                                                                    cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, inverse_jacobian, node_coordinates = cache.elements
    # Explicit bounds check, which allows us to assume inbounds access in the kernel
    @boundscheck begin
        check_axes(du, mesh, equations_parabolic, dg, cache)
        check_axes(u, mesh, equations_parabolic, dg, cache)
        for gradient in gradients
            check_axes(gradient, mesh, equations_parabolic, dg, cache)
        end
        check_axes(cache.elements, equations_parabolic, dg, cache)
        check_axes_surface_flux_values(surface_flux_values, mesh,
                                       equations_parabolic, dg, cache)
    end
    NNODES = nnodes(dg)
    kernel! = calc_surface_integral_and_apply_jacobian_and_calc_sources_KAkernel!(backend)
    kernel!(du, u, gradients, t, source_terms_parabolic, node_coordinates,
            typeof(mesh), equations_parabolic, inverse_weights[1], Val(NNODES),
            surface_flux_values, dg, inverse_jacobian,
            ndrange = (volume_node_ndrange(mesh, dg)..., nelements(dg, cache)))
    return nothing
end

# This is distinguished from the hyperbolic kernel of the same name by the
# additional `gradients` argument. The type parameter `Source` forces
# specialization on the source terms, see the comment in `src/solvers/dg_gpu.jl`.
@kernel inbounds=true function calc_surface_integral_and_apply_jacobian_and_calc_sources_KAkernel!(du,
                                                                                                   u,
                                                                                                   gradients::Tuple,
                                                                                                   t,
                                                                                                   source_terms_parabolic::Source,
                                                                                                   node_coordinates,
                                                                                                   MeshT::Type{<:P4estMesh{2}},
                                                                                                   equations_parabolic::AbstractEquationsParabolic{2},
                                                                                                   factor,
                                                                                                   ::Val{NNODES},
                                                                                                   surface_flux_values,
                                                                                                   dg::DGSEM,
                                                                                                   inverse_jacobian) where {
                                                                                                                            NNODES,
                                                                                                                            Source
                                                                                                                            }
    i, j, element = @index(Global, NTuple)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # As in the hyperbolic kernel, we avoid branching on the position of the
    # node and select the surface flux values of the faces in x and y direction
    # with `ifelse` instead (zero for interior nodes).
    #
    # factor = inverse_weights[1]
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    x_node_interface = (i == 1) | (i == NNODES)
    y_node_interface = (j == 1) | (j == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    _zero = zero(eltype(du))
    surface_node = SVector(ntuple(@inline(v->ifelse(x_node_interface,
                                                    @inbounds(surface_flux_values[v, j,
                                                                                  x_face,
                                                                                  element]),
                                                    _zero) +
                                             ifelse(y_node_interface,
                                                    @inbounds(surface_flux_values[v, i,
                                                                                  y_face,
                                                                                  element]),
                                                    _zero)),
                                  Val(nvariables(equations_parabolic))))
    source_node = calc_source_terms_node(u, gradients, t,
                                         source_terms_parabolic,
                                         node_coordinates,
                                         equations_parabolic, dg,
                                         i, j, element)
    # Apply Jacobian from mapping to reference element. In contrast to the
    # hyperbolic part, the sign of the inverse Jacobian is not flipped.
    jacobian_factor = inverse_jacobian[i, j, element]
    du_local = get_node_vars(du, equations_parabolic, dg, i, j, element) +
               factor * surface_node
    du_node = source_node + jacobian_factor * du_local
    set_node_vars!(du, du_node, equations_parabolic, dg, i, j, element)
end

Base.@propagate_inbounds function calc_source_terms_node(u,
                                                         gradients::NTuple{2,
                                                                           Any},
                                                         t,
                                                         source_terms_parabolic,
                                                         node_coordinates,
                                                         equations_parabolic,
                                                         dg::DG, indices...)
    gradients_x, gradients_y = gradients
    u_local = get_node_vars(u, equations_parabolic, dg, indices...)
    gradients_x_local = get_node_vars(gradients_x, equations_parabolic, dg,
                                      indices...)
    gradients_y_local = get_node_vars(gradients_y, equations_parabolic, dg,
                                      indices...)
    x_local = get_node_coords(node_coordinates, equations_parabolic, dg, indices...)

    return source_terms_parabolic(u_local, (gradients_x_local, gradients_y_local),
                                  x_local, t, equations_parabolic)
end

@inline function calc_source_terms_node(u, gradients::NTuple{2, Any}, t,
                                        source_terms_parabolic::Nothing,
                                        node_coordinates,
                                        equations_parabolic,
                                        dg::DG, indices...)
    return zero(SVector{nvariables(equations_parabolic), eltype(u)})
end
end # @muladd
