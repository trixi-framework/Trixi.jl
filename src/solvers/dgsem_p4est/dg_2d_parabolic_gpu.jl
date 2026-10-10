# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# KernelAbstractions.jl implementation of the parabolic right-hand side for the
# 2D `P4estMesh`. This mirrors `rhs_parabolic!(backend::Nothing, ...)` from
# `dg_2d_parabolic.jl`, but parallelizes over the individual nodes (volume
# terms) or surface nodes (interface and boundary terms) instead of over the
# elements, interfaces, and boundaries. To reduce the number of kernel launches,
# some steps of the CPU implementation are fused:
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
        calc_volume_integral_divergence!(backend, du, flux_parabolic, mesh,
                                         equations_parabolic, dg, cache)
    end

    # Prolong the normal parabolic fluxes to the interfaces and calculate the
    # interface fluxes
    @trixi_timeit_ext backend timer() "prolong2interfaces + interface flux" begin
        prolong2interfaces_and_calc_interface_flux_divergence!(backend,
                                                               cache.elements.surface_flux_values,
                                                               flux_parabolic, mesh,
                                                               equations_parabolic,
                                                               dg, parabolic_scheme,
                                                               cache)
    end

    # Prolong the normal parabolic fluxes to the boundaries and calculate the
    # boundary fluxes
    @trixi_timeit_ext backend timer() "prolong2boundaries + boundary flux" begin
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
    @trixi_timeit_ext backend timer() "surface integral, Jacobian + source terms" begin
        calc_surface_integral_and_apply_jacobian_and_calc_sources_parabolic!(backend,
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
    @trixi_timeit_ext backend timer() "prolong2interfaces + interface flux" begin
        prolong2interfaces_and_calc_interface_flux_gradient!(backend,
                                                             cache.elements.surface_flux_values,
                                                             u_transformed, mesh,
                                                             equations_parabolic,
                                                             dg, parabolic_scheme,
                                                             cache)
    end

    # Prolong solution to boundaries and calculate boundary fluxes
    @trixi_timeit_ext backend timer() "prolong2boundaries + boundary flux" begin
        calc_boundary_flux_gradient!(backend, cache, t, u_transformed,
                                     boundary_conditions_parabolic, mesh,
                                     equations_parabolic, dg)
    end

    # Mortars are not yet supported on GPU backends
    @trixi_timeit_ext backend timer() "mortar flux" begin
        @assert isempty(eachmortar(dg, cache))
    end

    # Calculate surface integrals and apply Jacobian from mapping to reference element
    @trixi_timeit_ext backend timer() "surface integral + Jacobian" begin
        calc_surface_integral_and_apply_jacobian_gradient!(backend, gradients, mesh,
                                                           equations_parabolic, dg,
                                                           cache)
    end

    return nothing
end

###############################################################################
# Volume terms

function transform_variables!(backend::Backend, u_transformed, u,
                              mesh::P4estMesh{2},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    kernel! = transform_variables_KAkernel!(backend)
    kernel!(u_transformed, u, equations_parabolic, dg,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

@kernel function transform_variables_KAkernel!(u_transformed, u, equations_parabolic,
                                               dg)
    i, j, element = @index(Global, NTuple)
    transformation = gradient_variable_transformation(equations_parabolic)
    @inbounds begin
        u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)
        u_transformed_node = transformation(u_node, equations_parabolic)
        set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg,
                       i, j, element)
    end
end

function calc_volume_integral_gradient!(backend::Backend, gradients, u_transformed,
                                        mesh::P4estMesh{2},
                                        equations_parabolic::AbstractEquationsParabolic,
                                        dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack contravariant_vectors = cache.elements
    gradients_x, gradients_y = gradients
    kernel! = calc_volume_integral_gradient_KAkernel!(backend)
    kernel!(gradients_x, gradients_y, u_transformed, equations_parabolic, dg,
            contravariant_vectors,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

@kernel function calc_volume_integral_gradient_KAkernel!(gradients_x, gradients_y,
                                                         u_transformed,
                                                         equations_parabolic, dg,
                                                         contravariant_vectors)
    i, j, element = @index(Global, NTuple)
    @inbounds begin
        @unpack derivative_hat = dg.basis

        # Gradients with respect to the reference coordinates
        gradients_reference_1 = zero(get_node_vars(u_transformed, equations_parabolic,
                                                   dg, i, j, element))
        gradients_reference_2 = gradients_reference_1
        for l in eachnode(dg)
            gradients_reference_1 = gradients_reference_1 +
                                    derivative_hat[i, l] *
                                    get_node_vars(u_transformed, equations_parabolic,
                                                  dg, l, j, element)
        end
        for l in eachnode(dg)
            gradients_reference_2 = gradients_reference_2 +
                                    derivative_hat[j, l] *
                                    get_node_vars(u_transformed, equations_parabolic,
                                                  dg, i, l, element)
        end

        # Transform the reference coordinate gradients to physical gradients
        # using the contravariant vectors. Note that the contravariant vectors are
        # transposed compared with computations of flux divergences.
        Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
        gradient_x_node = Ja11 * gradients_reference_1 + Ja21 * gradients_reference_2
        gradient_y_node = Ja12 * gradients_reference_1 + Ja22 * gradients_reference_2

        set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                       i, j, element)
        set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                       i, j, element)
    end
end

function calc_parabolic_fluxes!(backend::Backend, flux_parabolic, gradients,
                                u_transformed, mesh::P4estMesh{2},
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, cache)
    nelements(dg, cache) == 0 && return nothing
    gradients_x, gradients_y = gradients
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    kernel! = calc_parabolic_fluxes_KAkernel!(backend)
    kernel!(flux_parabolic_x, flux_parabolic_y, gradients_x, gradients_y,
            u_transformed, equations_parabolic, dg,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

@kernel function calc_parabolic_fluxes_KAkernel!(flux_parabolic_x, flux_parabolic_y,
                                                 gradients_x, gradients_y,
                                                 u_transformed, equations_parabolic,
                                                 dg)
    i, j, element = @index(Global, NTuple)
    @inbounds begin
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
end

# Weak-form volume integral of the divergence of the (precomputed) parabolic fluxes.
# In contrast to the CPU version, this overwrites `du`.
function calc_volume_integral_divergence!(backend::Backend, du, flux_parabolic,
                                          mesh::P4estMesh{2},
                                          equations_parabolic::AbstractEquationsParabolic,
                                          dg::DGSEM, cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack contravariant_vectors = cache.elements
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    kernel! = calc_volume_integral_divergence_KAkernel!(backend)
    kernel!(du, flux_parabolic_x, flux_parabolic_y, equations_parabolic, dg,
            contravariant_vectors,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

@inline function contravariant_parabolic_flux(orientation, flux_parabolic_x,
                                              flux_parabolic_y, equations_parabolic,
                                              dg, contravariant_vectors, i, j, element)
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg, i, j, element)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg, i, j, element)
    Ja1, Ja2 = get_contravariant_vector(orientation, contravariant_vectors,
                                        i, j, element)
    return Ja1 * flux1 + Ja2 * flux2
end

@kernel function calc_volume_integral_divergence_KAkernel!(du, flux_parabolic_x,
                                                           flux_parabolic_y,
                                                           equations_parabolic, dg,
                                                           contravariant_vectors)
    i, j, element = @index(Global, NTuple)
    @inbounds begin
        @unpack derivative_hat = dg.basis

        du_node = zero(get_node_vars(du, equations_parabolic, dg, i, j, element))
        for l in eachnode(dg)
            contravariant_flux1 = contravariant_parabolic_flux(1, flux_parabolic_x,
                                                               flux_parabolic_y,
                                                               equations_parabolic, dg,
                                                               contravariant_vectors,
                                                               l, j, element)
            du_node = du_node + derivative_hat[i, l] * contravariant_flux1
        end
        for l in eachnode(dg)
            contravariant_flux2 = contravariant_parabolic_flux(2, flux_parabolic_x,
                                                               flux_parabolic_y,
                                                               equations_parabolic, dg,
                                                               contravariant_vectors,
                                                               i, l, element)
            du_node = du_node + derivative_hat[j, l] * contravariant_flux2
        end

        set_node_vars!(du, du_node, equations_parabolic, dg, i, j, element)
    end
end

###############################################################################
# Interface terms

# Compute the element/direction/node information of both sides of an interface
# for the surface node `i` (counted along the primary side).
@inline function interface_node_info_2d(neighbor_ids, node_indices, index_range,
                                        i, interface)
    primary_element = neighbor_ids[1, interface]
    primary_indices = node_indices[1, interface]
    primary_direction = indices2direction(primary_indices)

    i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                             index_range)
    j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                             index_range)
    i_primary = delayed_index_2d(i_primary_start, i_primary_step, i)
    j_primary = delayed_index_2d(j_primary_start, j_primary_step, i)

    secondary_element = neighbor_ids[2, interface]
    secondary_indices = node_indices[2, interface]
    secondary_direction = indices2direction(secondary_indices)

    i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                 index_range)
    j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                 index_range)
    i_secondary = delayed_index_2d(i_secondary_start, i_secondary_step, i)
    j_secondary = delayed_index_2d(j_secondary_start, j_secondary_step, i)

    # Surface node index on the secondary element (might run backwards)
    if i_secondary_step == 0
        node_secondary = j_secondary
    else
        node_secondary = i_secondary
    end

    return (primary_element, primary_direction, i_primary, j_primary,
            secondary_element, secondary_direction, i_secondary, j_secondary,
            node_secondary)
end

function prolong2interfaces_and_calc_interface_flux_gradient!(backend::Backend,
                                                              surface_flux_values,
                                                              u_transformed,
                                                              mesh::P4estMesh{2},
                                                              equations_parabolic,
                                                              dg::DG,
                                                              parabolic_scheme, cache)
    ninterfaces(cache.interfaces) == 0 && return nothing
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    kernel! = interface_flux_gradient_KAkernel!(backend)
    kernel!(surface_flux_values, u_transformed, equations_parabolic, dg,
            parabolic_scheme, neighbor_ids, node_indices, contravariant_vectors,
            eachnode(dg),
            ndrange = (nnodes(dg), ninterfaces(cache.interfaces)))
    return nothing
end

@kernel function interface_flux_gradient_KAkernel!(surface_flux_values, u_transformed,
                                                   equations_parabolic, dg,
                                                   parabolic_scheme, neighbor_ids,
                                                   node_indices, contravariant_vectors,
                                                   index_range)
    i, interface = @index(Global, NTuple)
    @inbounds begin
        (primary_element, primary_direction, i_primary, j_primary,
        secondary_element, secondary_direction, i_secondary, j_secondary,
        node_secondary) = interface_node_info_2d(neighbor_ids, node_indices,
                                                 index_range, i, interface)

        u_ll = get_node_vars(u_transformed, equations_parabolic, dg,
                             i_primary, j_primary, primary_element)
        u_rr = get_node_vars(u_transformed, equations_parabolic, dg,
                             i_secondary, j_secondary, secondary_element)

        normal_direction = get_normal_direction(primary_direction,
                                                contravariant_vectors,
                                                i_primary, j_primary, primary_element)

        flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                               equations_parabolic, parabolic_scheme)

        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
            # No sign flip required for gradient calculation because for parabolic terms,
            # the normals are not embedded in `flux_` for gradient computations.
            surface_flux_values[v, node_secondary, secondary_direction,
            secondary_element] = flux_[v]
        end
    end
end

# Parabolic flux in the outward normal direction `normal_direction` at a volume node
@inline function normal_parabolic_flux(flux_parabolic_x, flux_parabolic_y,
                                       normal_direction, equations_parabolic, dg,
                                       i, j, element)
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg, i, j, element)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg, i, j, element)
    return normal_direction[1] * flux1 + normal_direction[2] * flux2
end

function prolong2interfaces_and_calc_interface_flux_divergence!(backend::Backend,
                                                                surface_flux_values,
                                                                flux_parabolic,
                                                                mesh::P4estMesh{2},
                                                                equations_parabolic,
                                                                dg::DG,
                                                                parabolic_scheme,
                                                                cache)
    ninterfaces(cache.interfaces) == 0 && return nothing
    @unpack neighbor_ids, node_indices = cache.interfaces
    @unpack contravariant_vectors = cache.elements
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    kernel! = interface_flux_divergence_KAkernel!(backend)
    kernel!(surface_flux_values, flux_parabolic_x, flux_parabolic_y,
            equations_parabolic, dg, parabolic_scheme, neighbor_ids, node_indices,
            contravariant_vectors, eachnode(dg),
            ndrange = (nnodes(dg), ninterfaces(cache.interfaces)))
    return nothing
end

@kernel function interface_flux_divergence_KAkernel!(surface_flux_values,
                                                     flux_parabolic_x,
                                                     flux_parabolic_y,
                                                     equations_parabolic, dg,
                                                     parabolic_scheme, neighbor_ids,
                                                     node_indices,
                                                     contravariant_vectors,
                                                     index_range)
    i, interface = @index(Global, NTuple)
    @inbounds begin
        (primary_element, primary_direction, i_primary, j_primary,
        secondary_element, secondary_direction, i_secondary, j_secondary,
        node_secondary) = interface_node_info_2d(neighbor_ids, node_indices,
                                                 index_range, i, interface)

        # Outward normal directions on the primary and secondary element
        normal_direction_primary = get_normal_direction(primary_direction,
                                                        contravariant_vectors,
                                                        i_primary, j_primary,
                                                        primary_element)
        normal_direction_secondary = get_normal_direction(secondary_direction,
                                                          contravariant_vectors,
                                                          i_secondary, j_secondary,
                                                          secondary_element)

        # Normal parabolic fluxes with respect to the primary normal direction,
        # which is the negative of the secondary normal direction
        parabolic_flux_normal_ll = normal_parabolic_flux(flux_parabolic_x,
                                                         flux_parabolic_y,
                                                         normal_direction_primary,
                                                         equations_parabolic, dg,
                                                         i_primary, j_primary,
                                                         primary_element)
        parabolic_flux_normal_rr = -normal_parabolic_flux(flux_parabolic_x,
                                                          flux_parabolic_y,
                                                          normal_direction_secondary,
                                                          equations_parabolic, dg,
                                                          i_secondary, j_secondary,
                                                          secondary_element)

        flux_ = flux_parabolic(parabolic_flux_normal_ll, parabolic_flux_normal_rr,
                               normal_direction_primary, Divergence(),
                               equations_parabolic, parabolic_scheme)

        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
            # Sign flip required for divergence calculation since the divergence
            # interface flux involves the normal direction.
            surface_flux_values[v, node_secondary, secondary_direction,
            secondary_element] = -flux_[v]
        end
    end
end

###############################################################################
# Boundary terms

function calc_boundary_flux_gradient!(backend::Backend, cache, t, u_transformed,
                                      boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                      mesh::P4estMesh{2}, equations_parabolic, dg::DG)
    @assert isempty(eachboundary(dg, cache))
    return nothing
end

function calc_boundary_flux_divergence!(backend::Backend, cache, t, flux_parabolic,
                                        boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::P4estMesh{2}, equations_parabolic,
                                        dg::DG)
    @assert isempty(eachboundary(dg, cache))
    return nothing
end

function calc_boundary_flux_gradient!(backend::Backend, cache, t, u_transformed,
                                      boundary_conditions_parabolic::UnstructuredSortedBoundaryTypes,
                                      mesh::P4estMesh{2}, equations_parabolic, dg::DG)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions_parabolic
    calc_boundary_flux_parabolic_by_type!(backend, cache, t, u_transformed,
                                          boundary_condition_types, boundary_indices,
                                          Gradient(), mesh, equations_parabolic, dg)
    return nothing
end

function calc_boundary_flux_divergence!(backend::Backend, cache, t, flux_parabolic,
                                        boundary_conditions_parabolic::UnstructuredSortedBoundaryTypes,
                                        mesh::P4estMesh{2}, equations_parabolic,
                                        dg::DG)
    @unpack boundary_condition_types, boundary_indices = boundary_conditions_parabolic
    calc_boundary_flux_parabolic_by_type!(backend, cache, t, flux_parabolic,
                                          boundary_condition_types, boundary_indices,
                                          Divergence(), mesh, equations_parabolic, dg)
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming", launching one kernel
# per boundary condition type. Thus, the kernels are specialized on the
# boundary condition and do not need any dynamic dispatch.
function calc_boundary_flux_parabolic_by_type!(backend::Backend, cache, t, u_or_flux,
                                               BCs::Tuple{}, BC_indices::Tuple{},
                                               operator_type, mesh::P4estMesh{2},
                                               equations_parabolic, dg::DG)
    return nothing
end

function calc_boundary_flux_parabolic_by_type!(backend::Backend, cache, t, u_or_flux,
                                               BCs::Tuple{Any, Vararg{Any}},
                                               BC_indices::Tuple{AbstractVector{Int},
                                                                 Vararg{AbstractVector{Int}}},
                                               operator_type, mesh::P4estMesh{2},
                                               equations_parabolic, dg::DG)
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)

    n_boundaries = length(boundary_condition_indices)
    if n_boundaries > 0
        @unpack neighbor_ids, node_indices = cache.boundaries
        @unpack node_coordinates, contravariant_vectors, surface_flux_values = cache.elements
        kernel! = calc_boundary_flux_parabolic_KAkernel!(backend)
        kernel!(surface_flux_values, u_or_flux, boundary_condition_indices,
                neighbor_ids, node_indices, t, boundary_condition, operator_type,
                eachnode(dg), equations_parabolic, dg, node_coordinates,
                contravariant_vectors,
                ndrange = (nnodes(dg), n_boundaries))
    end

    calc_boundary_flux_parabolic_by_type!(backend, cache, t, u_or_flux,
                                          Base.tail(BCs), Base.tail(BC_indices),
                                          operator_type, mesh, equations_parabolic, dg)
    return nothing
end

# Values at the boundary node passed to the boundary condition:
# the (transformed) solution for the gradient computation ...
@inline function boundary_inner_value_parabolic(::Gradient, u_transformed,
                                                normal_direction, equations_parabolic,
                                                dg, i, j, element)
    return get_node_vars(u_transformed, equations_parabolic, dg, i, j, element)
end

# ... and the parabolic flux in the outward normal direction for the divergence computation
@inline function boundary_inner_value_parabolic(::Divergence, flux_parabolic,
                                                normal_direction, equations_parabolic,
                                                dg, i, j, element)
    flux_parabolic_x, flux_parabolic_y = flux_parabolic
    return normal_parabolic_flux(flux_parabolic_x, flux_parabolic_y, normal_direction,
                                 equations_parabolic, dg, i, j, element)
end

@kernel function calc_boundary_flux_parabolic_KAkernel!(surface_flux_values, u_or_flux,
                                                        boundary_condition_indices,
                                                        neighbor_ids, node_indices_arr,
                                                        t, boundary_condition,
                                                        operator_type, index_range,
                                                        equations_parabolic, dg,
                                                        node_coordinates,
                                                        contravariant_vectors)
    node_index, local_index = @index(Global, NTuple)
    @inbounds begin
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary = boundary_condition_indices[local_index]

        # Get information on the adjacent element
        element = neighbor_ids[boundary]
        node_indices = node_indices_arr[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)
        i_node = delayed_index_2d(i_node_start, i_node_step, node_index)
        j_node = delayed_index_2d(j_node_start, j_node_step, node_index)

        # Outward-pointing normal direction (not normalized)
        normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                i_node, j_node, element)

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
            surface_flux_values[v, node_index, direction, element] = flux_[v]
        end
    end
end

###############################################################################
# Surface integrals, Jacobian, and source terms

function calc_surface_integral_and_apply_jacobian_gradient!(backend::Backend,
                                                            gradients,
                                                            mesh::P4estMesh{2},
                                                            equations_parabolic::AbstractEquationsParabolic,
                                                            dg::DGSEM{<:LobattoLegendreBasis},
                                                            cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, contravariant_vectors, inverse_jacobian = cache.elements
    gradients_x, gradients_y = gradients
    kernel! = calc_surface_integral_and_apply_jacobian_gradient_KAkernel!(backend)
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    factor = inverse_weights[1]
    kernel!(gradients_x, gradients_y, surface_flux_values, contravariant_vectors,
            inverse_jacobian, factor, equations_parabolic, dg,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

# Surface integral contribution of the face `direction` at the volume node (i, j)
# and surface node `l`
@inline function surface_integral_gradient_node(gradient_x_node, gradient_y_node,
                                                surface_flux_values,
                                                contravariant_vectors, factor,
                                                equations_parabolic, dg, direction,
                                                l, i, j, element)
    normal_direction_x, normal_direction_y = get_normal_direction(direction,
                                                                  contravariant_vectors,
                                                                  i, j, element)
    surface_flux = get_node_vars(surface_flux_values, equations_parabolic, dg,
                                 l, direction, element)
    gradient_x_node = gradient_x_node + surface_flux * factor * normal_direction_x
    gradient_y_node = gradient_y_node + surface_flux * factor * normal_direction_y
    return gradient_x_node, gradient_y_node
end

@kernel function calc_surface_integral_and_apply_jacobian_gradient_KAkernel!(gradients_x,
                                                                             gradients_y,
                                                                             surface_flux_values,
                                                                             contravariant_vectors,
                                                                             inverse_jacobian,
                                                                             factor,
                                                                             equations_parabolic,
                                                                             dg)
    i, j, element = @index(Global, NTuple)
    @inbounds begin
        n = nnodes(dg)
        gradient_x_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                        i, j, element)
        gradient_y_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                        i, j, element)

        # surface at -x
        if i == 1
            gradient_x_node, gradient_y_node = surface_integral_gradient_node(gradient_x_node,
                                                                              gradient_y_node,
                                                                              surface_flux_values,
                                                                              contravariant_vectors,
                                                                              factor,
                                                                              equations_parabolic,
                                                                              dg, 1, j,
                                                                              i, j,
                                                                              element)
        end
        # surface at +x
        if i == n
            gradient_x_node, gradient_y_node = surface_integral_gradient_node(gradient_x_node,
                                                                              gradient_y_node,
                                                                              surface_flux_values,
                                                                              contravariant_vectors,
                                                                              factor,
                                                                              equations_parabolic,
                                                                              dg, 2, j,
                                                                              i, j,
                                                                              element)
        end
        # surface at -y
        if j == 1
            gradient_x_node, gradient_y_node = surface_integral_gradient_node(gradient_x_node,
                                                                              gradient_y_node,
                                                                              surface_flux_values,
                                                                              contravariant_vectors,
                                                                              factor,
                                                                              equations_parabolic,
                                                                              dg, 3, i,
                                                                              i, j,
                                                                              element)
        end
        # surface at +y
        if j == n
            gradient_x_node, gradient_y_node = surface_integral_gradient_node(gradient_x_node,
                                                                              gradient_y_node,
                                                                              surface_flux_values,
                                                                              contravariant_vectors,
                                                                              factor,
                                                                              equations_parabolic,
                                                                              dg, 4, i,
                                                                              i, j,
                                                                              element)
        end

        # Apply Jacobian from mapping to reference element. In contrast to the
        # hyperbolic part, the sign of the inverse Jacobian is not flipped.
        jacobian_factor = inverse_jacobian[i, j, element]
        set_node_vars!(gradients_x, jacobian_factor * gradient_x_node,
                       equations_parabolic, dg, i, j, element)
        set_node_vars!(gradients_y, jacobian_factor * gradient_y_node,
                       equations_parabolic, dg, i, j, element)
    end
end

function calc_surface_integral_and_apply_jacobian_and_calc_sources_parabolic!(backend::Backend,
                                                                              du, u,
                                                                              gradients,
                                                                              t,
                                                                              source_terms_parabolic,
                                                                              mesh::P4estMesh{2},
                                                                              equations_parabolic::AbstractEquationsParabolic,
                                                                              surface_integral::SurfaceIntegralWeakForm,
                                                                              dg::DGSEM{<:LobattoLegendreBasis},
                                                                              cache)
    nelements(dg, cache) == 0 && return nothing
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, inverse_jacobian, node_coordinates = cache.elements
    gradients_x, gradients_y = gradients
    kernel! = calc_surface_integral_and_apply_jacobian_and_calc_sources_parabolic_KAkernel!(backend)
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    factor = inverse_weights[1]
    kernel!(du, u, gradients_x, gradients_y, t, source_terms_parabolic,
            surface_flux_values, inverse_jacobian, node_coordinates, factor,
            equations_parabolic, dg,
            ndrange = (nnodes(dg), nnodes(dg), nelements(dg, cache)))
    return nothing
end

@inline function calc_source_terms_parabolic_node(u, gradients_x, gradients_y, t,
                                                  source_terms_parabolic,
                                                  node_coordinates,
                                                  equations_parabolic, dg, i, j,
                                                  element)
    u_local = get_node_vars(u, equations_parabolic, dg, i, j, element)
    gradients_x_local = get_node_vars(gradients_x, equations_parabolic, dg,
                                      i, j, element)
    gradients_y_local = get_node_vars(gradients_y, equations_parabolic, dg,
                                      i, j, element)
    x_local = get_node_coords(node_coordinates, equations_parabolic, dg,
                              i, j, element)
    return source_terms_parabolic(u_local, (gradients_x_local, gradients_y_local),
                                  x_local, t, equations_parabolic)
end

@kernel function calc_surface_integral_and_apply_jacobian_and_calc_sources_parabolic_KAkernel!(du,
                                                                                               u,
                                                                                               gradients_x,
                                                                                               gradients_y,
                                                                                               t,
                                                                                               source_terms_parabolic,
                                                                                               surface_flux_values,
                                                                                               inverse_jacobian,
                                                                                               node_coordinates,
                                                                                               factor,
                                                                                               equations_parabolic,
                                                                                               dg)
    i, j, element = @index(Global, NTuple)
    @inbounds begin
        n = nnodes(dg)
        du_node = get_node_vars(du, equations_parabolic, dg, i, j, element)

        # Note that all fluxes have been computed with outward-pointing normal vectors.
        # surface at -x
        if i == 1
            du_node = du_node +
                      factor * get_node_vars(surface_flux_values, equations_parabolic,
                                    dg, j, 1, element)
        end
        # surface at +x
        if i == n
            du_node = du_node +
                      factor * get_node_vars(surface_flux_values, equations_parabolic,
                                    dg, j, 2, element)
        end
        # surface at -y
        if j == 1
            du_node = du_node +
                      factor * get_node_vars(surface_flux_values, equations_parabolic,
                                    dg, i, 3, element)
        end
        # surface at +y
        if j == n
            du_node = du_node +
                      factor * get_node_vars(surface_flux_values, equations_parabolic,
                                    dg, i, 4, element)
        end

        # Apply Jacobian from mapping to reference element. In contrast to the
        # hyperbolic part, the sign of the inverse Jacobian is not flipped.
        du_node = inverse_jacobian[i, j, element] * du_node

        # Add parabolic source terms
        if !(source_terms_parabolic isa Nothing)
            du_node = du_node +
                      calc_source_terms_parabolic_node(u, gradients_x, gradients_y, t,
                                                       source_terms_parabolic,
                                                       node_coordinates,
                                                       equations_parabolic, dg,
                                                       i, j, element)
        end

        set_node_vars!(du, du_node, equations_parabolic, dg, i, j, element)
    end
end
end # @muladd
