# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# KernelAbstractions.jl kernels of the parabolic right-hand side for the 3D
# `P4estMesh`. The functions launching the kernels (`rhs_parabolic!`,
# `calc_gradient!`, `calc_volume_integral!`, ...) are generic in the number of
# spatial dimensions and live in `dg_2d_parabolic_gpu.jl`, which also explains
# the fusion of the steps of the CPU implementation. This file only contains
# the 3D versions of the kernels and of the `per_node!` functions they call,
# which dispatch on the mesh type `MeshT`, as in the hyperbolic GPU code in
# `dg_3d_gpu.jl`.
#
# Currently, only conforming meshes are supported (no mortars).

###############################################################################
# Volume terms

@kernel inbounds=true function transform_variables_KAkernel!(u_transformed, u,
                                                             MeshT::Type{<:P4estMesh{3}},
                                                             equations_parabolic,
                                                             dg)
    i, j, k, element = @index(Global, NTuple)
    transformation = gradient_variable_transformation(equations_parabolic)
    u_node = get_node_vars(u, equations_parabolic, dg, i, j, k, element)
    u_transformed_node = transformation(u_node, equations_parabolic)
    set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg,
                   i, j, k, element)
end

# In contrast to the CPU version, this computes the complete volume integral of
# a node at once and overwrites `gradients`.
@kernel inbounds=true function calc_volume_integral_gradient_KAkernel!(gradients,
                                                                       u_transformed,
                                                                       MeshT::Type{<:P4estMesh{3}},
                                                                       equations_parabolic,
                                                                       dg,
                                                                       derivative_hat,
                                                                       contravariant_vectors)
    i, j, k, element = @index(Global, NTuple)
    gradients_x, gradients_y, gradients_z = gradients

    # Gradients with respect to the reference coordinates
    gradients_reference_1 = zero(get_node_vars(u_transformed, equations_parabolic,
                                               dg, i, j, k, element))
    gradients_reference_2 = gradients_reference_1
    gradients_reference_3 = gradients_reference_1

    for ii in eachnode(dg)
        gradients_reference_1 = gradients_reference_1 +
                                derivative_hat[i, ii] *
                                get_node_vars(u_transformed, equations_parabolic,
                                              dg, ii, j, k, element)
    end

    for jj in eachnode(dg)
        gradients_reference_2 = gradients_reference_2 +
                                derivative_hat[j, jj] *
                                get_node_vars(u_transformed, equations_parabolic,
                                              dg, i, jj, k, element)
    end

    for kk in eachnode(dg)
        gradients_reference_3 = gradients_reference_3 +
                                derivative_hat[k, kk] *
                                get_node_vars(u_transformed, equations_parabolic,
                                              dg, i, j, kk, element)
    end

    # Transform the reference coordinate gradients to physical gradients using
    # the contravariant vectors. Note that the contravariant vectors are
    # transposed compared with computations of flux divergences in
    # `calc_volume_integral!`. See
    # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190
    # for a more detailed discussion.
    Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                i, j, k, element)
    Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                i, j, k, element)
    Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                i, j, k, element)
    gradient_x_node = Ja11 * gradients_reference_1 +
                      Ja21 * gradients_reference_2 +
                      Ja31 * gradients_reference_3
    gradient_y_node = Ja12 * gradients_reference_1 +
                      Ja22 * gradients_reference_2 +
                      Ja32 * gradients_reference_3
    gradient_z_node = Ja13 * gradients_reference_1 +
                      Ja23 * gradients_reference_2 +
                      Ja33 * gradients_reference_3

    set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                   i, j, k, element)
    set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                   i, j, k, element)
    set_node_vars!(gradients_z, gradient_z_node, equations_parabolic, dg,
                   i, j, k, element)
end

@kernel inbounds=true function calc_parabolic_fluxes_KAkernel!(flux_parabolic,
                                                               gradients,
                                                               u_transformed,
                                                               MeshT::Type{<:P4estMesh{3}},
                                                               equations_parabolic,
                                                               dg)
    i, j, k, element = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
    gradients_x, gradients_y, gradients_z = gradients

    u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, k, element)
    gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg,
                                     i, j, k, element)
    gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg,
                                     i, j, k, element)
    gradients_3_node = get_node_vars(gradients_z, equations_parabolic, dg,
                                     i, j, k, element)
    gradients_node = (gradients_1_node, gradients_2_node, gradients_3_node)

    flux_parabolic_node_x = flux(u_node, gradients_node, 1, equations_parabolic)
    flux_parabolic_node_y = flux(u_node, gradients_node, 2, equations_parabolic)
    flux_parabolic_node_z = flux(u_node, gradients_node, 3, equations_parabolic)
    set_node_vars!(flux_parabolic_x, flux_parabolic_node_x, equations_parabolic,
                   dg, i, j, k, element)
    set_node_vars!(flux_parabolic_y, flux_parabolic_node_y, equations_parabolic,
                   dg, i, j, k, element)
    set_node_vars!(flux_parabolic_z, flux_parabolic_node_z, equations_parabolic,
                   dg, i, j, k, element)
end

# In contrast to the CPU version, this computes the complete volume integral of
# a node at once and overwrites `du`.
@kernel inbounds=true function calc_volume_integral_KAkernel!(du, flux_parabolic,
                                                              MeshT::Type{<:P4estMesh{3}},
                                                              equations_parabolic,
                                                              dg,
                                                              derivative_hat,
                                                              contravariant_vectors)
    i, j, k, element = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic

    du_node = zero(get_node_vars(du, equations_parabolic, dg, i, j, k, element))

    for ii in eachnode(dg)
        flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                              ii, j, k, element)
        flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                              ii, j, k, element)
        flux3 = get_node_vars(flux_parabolic_z, equations_parabolic, dg,
                              ii, j, k, element)

        # Compute the contravariant flux by taking the scalar product of the
        # first contravariant vector Ja^1 and the flux vector
        Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                    ii, j, k, element)
        contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2 + Ja13 * flux3
        du_node = du_node + derivative_hat[i, ii] * contravariant_flux1
    end

    for jj in eachnode(dg)
        flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                              i, jj, k, element)
        flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                              i, jj, k, element)
        flux3 = get_node_vars(flux_parabolic_z, equations_parabolic, dg,
                              i, jj, k, element)

        # Compute the contravariant flux by taking the scalar product of the
        # second contravariant vector Ja^2 and the flux vector
        Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                    i, jj, k, element)
        contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2 + Ja23 * flux3
        du_node = du_node + derivative_hat[j, jj] * contravariant_flux2
    end

    for kk in eachnode(dg)
        flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                              i, j, kk, element)
        flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                              i, j, kk, element)
        flux3 = get_node_vars(flux_parabolic_z, equations_parabolic, dg,
                              i, j, kk, element)

        # Compute the contravariant flux by taking the scalar product of the
        # third contravariant vector Ja^3 and the flux vector
        Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                    i, j, kk, element)
        contravariant_flux3 = Ja31 * flux1 + Ja32 * flux2 + Ja33 * flux3
        du_node = du_node + derivative_hat[k, kk] * contravariant_flux3
    end

    set_node_vars!(du, du_node, equations_parabolic, dg, i, j, k, element)
end

###############################################################################
# Interface terms

@kernel inbounds=true function prolong2interfaces_and_calc_interface_flux_gradient_KAkernel!(surface_flux_values,
                                                                                             u_transformed,
                                                                                             MeshT::Type{<:P4estMesh{3}},
                                                                                             equations_parabolic,
                                                                                             dg,
                                                                                             parabolic_scheme,
                                                                                             neighbor_ids,
                                                                                             node_indices,
                                                                                             contravariant_vectors,
                                                                                             index_range)
    i, j, interface = @index(Global, NTuple)
    prolong2interfaces_and_calc_interface_flux_gradient_per_node!(surface_flux_values,
                                                                  u_transformed,
                                                                  MeshT,
                                                                  equations_parabolic,
                                                                  dg,
                                                                  parabolic_scheme,
                                                                  neighbor_ids,
                                                                  node_indices,
                                                                  contravariant_vectors,
                                                                  index_range, i, j,
                                                                  interface)
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_gradient_per_node!(surface_flux_values,
                                                                                                u_transformed,
                                                                                                MeshT::Type{<:P4estMesh{3}},
                                                                                                equations_parabolic,
                                                                                                dg,
                                                                                                parabolic_scheme,
                                                                                                neighbor_ids,
                                                                                                node_indices,
                                                                                                contravariant_vectors,
                                                                                                index_range,
                                                                                                i,
                                                                                                j,
                                                                                                interface)
    # This reuses the hyperbolic version, which also returns the outward normal
    # direction on the primary element
    u_ll, u_rr, normal_direction, primary_direction, primary_element,
    i_secondary, j_secondary, secondary_direction, secondary_element = get_interface_values(u_transformed,
                                                                                            equations_parabolic,
                                                                                            dg,
                                                                                            neighbor_ids,
                                                                                            node_indices,
                                                                                            contravariant_vectors,
                                                                                            index_range,
                                                                                            i,
                                                                                            j,
                                                                                            interface)

    flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                           equations_parabolic, parabolic_scheme)

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, j, primary_direction, primary_element] = flux_[v]
        # No sign flip required for gradient calculation because for parabolic terms,
        # the normals are not embedded in `flux_` for gradient computations.
        surface_flux_values[v, i_secondary, j_secondary, secondary_direction,
        secondary_element] = flux_[v]
    end
    return nothing
end

@kernel inbounds=true function prolong2interfaces_and_calc_interface_flux_KAkernel!(surface_flux_values,
                                                                                    flux_parabolic,
                                                                                    MeshT::Type{<:P4estMesh{3}},
                                                                                    equations_parabolic,
                                                                                    dg,
                                                                                    parabolic_scheme,
                                                                                    neighbor_ids,
                                                                                    node_indices,
                                                                                    contravariant_vectors,
                                                                                    index_range)
    i, j, interface = @index(Global, NTuple)
    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
    prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                         flux_parabolic_x,
                                                         flux_parabolic_y,
                                                         flux_parabolic_z, MeshT,
                                                         equations_parabolic, dg,
                                                         parabolic_scheme,
                                                         neighbor_ids, node_indices,
                                                         contravariant_vectors,
                                                         index_range, i, j, interface)
end

Base.@propagate_inbounds function prolong2interfaces_and_calc_interface_flux_per_node!(surface_flux_values,
                                                                                       flux_parabolic_x,
                                                                                       flux_parabolic_y,
                                                                                       flux_parabolic_z,
                                                                                       MeshT::Type{<:P4estMesh{3}},
                                                                                       equations_parabolic,
                                                                                       dg,
                                                                                       parabolic_scheme,
                                                                                       neighbor_ids,
                                                                                       node_indices,
                                                                                       contravariant_vectors,
                                                                                       index_range,
                                                                                       i,
                                                                                       j,
                                                                                       interface)
    (primary_direction, primary_element, i_primary, j_primary, k_primary,
    i_secondary, j_secondary, secondary_direction, secondary_element,
    i_secondary_node, j_secondary_node, k_secondary_node) = get_interface_indices(neighbor_ids,
                                                                                  node_indices,
                                                                                  index_range,
                                                                                  i, j,
                                                                                  interface)

    # Outward normal direction on the primary element
    normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                                            i_primary, j_primary, k_primary,
                                            primary_element)

    # Parabolic flux of the primary element dotted with its outward normal direction
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                          i_primary, j_primary, k_primary, primary_element)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                          i_primary, j_primary, k_primary, primary_element)
    flux3 = get_node_vars(flux_parabolic_z, equations_parabolic, dg,
                          i_primary, j_primary, k_primary, primary_element)
    parabolic_flux_normal_ll = normal_direction[1] * flux1 +
                               normal_direction[2] * flux2 +
                               normal_direction[3] * flux3

    # Parabolic flux of the secondary element dotted with its outward normal
    # direction. Here, we assume that the normal direction on the secondary
    # element is the negative of the normal direction on the primary element,
    # and store the normal flux with respect to the primary normal direction.
    normal_direction_secondary = get_normal_direction(secondary_direction,
                                                      contravariant_vectors,
                                                      i_secondary_node,
                                                      j_secondary_node,
                                                      k_secondary_node,
                                                      secondary_element)
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg,
                          i_secondary_node, j_secondary_node, k_secondary_node,
                          secondary_element)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg,
                          i_secondary_node, j_secondary_node, k_secondary_node,
                          secondary_element)
    flux3 = get_node_vars(flux_parabolic_z, equations_parabolic, dg,
                          i_secondary_node, j_secondary_node, k_secondary_node,
                          secondary_element)
    parabolic_flux_normal_rr = -(normal_direction_secondary[1] * flux1 +
                                 normal_direction_secondary[2] * flux2 +
                                 normal_direction_secondary[3] * flux3)

    flux_ = flux_parabolic(parabolic_flux_normal_ll, parabolic_flux_normal_rr,
                           normal_direction, Divergence(),
                           equations_parabolic, parabolic_scheme)

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, j, primary_direction, primary_element] = flux_[v]
        # Sign flip required for divergence calculation since the divergence
        # interface flux involves the normal direction.
        surface_flux_values[v, i_secondary, j_secondary, secondary_direction,
        secondary_element] = -flux_[v]
    end
    return nothing
end

###############################################################################
# Boundary terms

@kernel inbounds=true function calc_boundary_flux_KAkernel!(surface_flux_values,
                                                            u_or_flux,
                                                            boundary_condition_indices,
                                                            neighbor_ids,
                                                            node_indices_arr,
                                                            t,
                                                            boundary_condition::BC,
                                                            operator_type,
                                                            index_range,
                                                            MeshT::Type{<:P4estMesh{3}},
                                                            equations_parabolic,
                                                            dg,
                                                            node_coordinates,
                                                            contravariant_vectors) where {BC}
    i, j, local_index = @index(Global, NTuple)
    # Use the local index to get the global boundary index from the pre-sorted list
    boundary = boundary_condition_indices[local_index]

    calc_boundary_flux_per_node!(surface_flux_values, u_or_flux, t,
                                 boundary_condition, operator_type, MeshT,
                                 equations_parabolic, dg, boundary, neighbor_ids,
                                 node_indices_arr, index_range, node_coordinates,
                                 contravariant_vectors, i, j)
end

Base.@propagate_inbounds function calc_boundary_flux_per_node!(surface_flux_values,
                                                               u_or_flux, t,
                                                               boundary_condition,
                                                               operator_type,
                                                               MeshT::Type{<:P4estMesh{3}},
                                                               equations_parabolic,
                                                               dg, boundary,
                                                               neighbor_ids,
                                                               node_indices_arr,
                                                               index_range,
                                                               node_coordinates,
                                                               contravariant_vectors,
                                                               i, j)
    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element = neighbor_ids[boundary]
    node_indices = node_indices_arr[boundary]
    direction = indices2direction(node_indices)
    n = length(index_range)

    i_node_start, i_node_step_i, i_node_step_j = index_to_start_step_3d(node_indices[1],
                                                                        index_range)
    j_node_start, j_node_step_i, j_node_step_j = index_to_start_step_3d(node_indices[2],
                                                                        index_range)
    k_node_start, k_node_step_i, k_node_step_j = index_to_start_step_3d(node_indices[3],
                                                                        index_range)
    i_node = delayed_index_3d(i_node_start, i_node_step_i, i_node_step_j, i, j, n)
    j_node = delayed_index_3d(j_node_start, j_node_step_i, j_node_step_j, i, j, n)
    k_node = delayed_index_3d(k_node_start, k_node_step_i, k_node_step_j, i, j, n)

    # Outward-pointing normal direction (not normalized)
    normal_direction = get_normal_direction(direction, contravariant_vectors,
                                            i_node, j_node, k_node, element)

    # Prolong the inner value to the boundary node, which is stored in
    # `boundaries.u` in the CPU version
    u_inner = boundary_inner_value_parabolic(operator_type, u_or_flux,
                                             normal_direction, equations_parabolic,
                                             dg, i_node, j_node, k_node, element)

    # This assumes the gradient numerical flux at the boundary is the gradient variable,
    # which is consistent with BR1, LDG.
    flux_inner = u_inner

    # Coordinates at boundary node
    x = get_node_coords(node_coordinates, equations_parabolic, dg,
                        i_node, j_node, k_node, element)

    flux_ = boundary_condition(flux_inner, u_inner, normal_direction, x, t,
                               operator_type, equations_parabolic)

    # Copy flux to element storage in the correct orientation
    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, j, direction, element] = flux_[v]
    end
    return nothing
end

# The parabolic flux dotted with the outward normal direction for the
# divergence computation (the gradient version is generic in the number of
# spatial dimensions, see `dg_2d_parabolic_gpu.jl`)
Base.@propagate_inbounds function boundary_inner_value_parabolic(::Divergence,
                                                                 flux_parabolic::NTuple{3,
                                                                                        Any},
                                                                 normal_direction,
                                                                 equations_parabolic,
                                                                 dg, indices...)
    flux_parabolic_x, flux_parabolic_y, flux_parabolic_z = flux_parabolic
    flux1 = get_node_vars(flux_parabolic_x, equations_parabolic, dg, indices...)
    flux2 = get_node_vars(flux_parabolic_y, equations_parabolic, dg, indices...)
    flux3 = get_node_vars(flux_parabolic_z, equations_parabolic, dg, indices...)
    return normal_direction[1] * flux1 + normal_direction[2] * flux2 +
           normal_direction[3] * flux3
end

###############################################################################
# Surface integrals, Jacobian, and source terms

@kernel inbounds=true function calc_surface_integral_and_apply_jacobian_gradient_KAkernel!(gradients,
                                                                                           MeshT::Type{<:P4estMesh{3}},
                                                                                           equations_parabolic::AbstractEquationsParabolic{3},
                                                                                           factor,
                                                                                           ::Val{NNODES},
                                                                                           surface_flux_values,
                                                                                           dg::DGSEM,
                                                                                           contravariant_vectors,
                                                                                           inverse_jacobian) where {
                                                                                                                    NNODES
                                                                                                                    }
    i, j, k, element = @index(Global, NTuple)
    gradients_x, gradients_y, gradients_z = gradients
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # As in the hyperbolic kernel, we avoid branching on the position of the
    # node and select the surface flux values of the faces in x, y, and z
    # direction with `ifelse` instead (zero for interior nodes).
    #
    # factor = inverse_weights[1]
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    x_node_interface = (i == 1) | (i == NNODES)
    y_node_interface = (j == 1) | (j == NNODES)
    z_node_interface = (k == 1) | (k == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    z_face = ifelse(k == 1, 5, 6)
    _zero = zero(eltype(gradients_x))
    surface_flux_x_face = SVector(ntuple(@inline(v->ifelse(x_node_interface,
                                                           @inbounds(surface_flux_values[v,
                                                                                         j,
                                                                                         k,
                                                                                         x_face,
                                                                                         element]),
                                                           _zero)),
                                         Val(nvariables(equations_parabolic))))
    surface_flux_y_face = SVector(ntuple(@inline(v->ifelse(y_node_interface,
                                                           @inbounds(surface_flux_values[v,
                                                                                         i,
                                                                                         k,
                                                                                         y_face,
                                                                                         element]),
                                                           _zero)),
                                         Val(nvariables(equations_parabolic))))
    surface_flux_z_face = SVector(ntuple(@inline(v->ifelse(z_node_interface,
                                                           @inbounds(surface_flux_values[v,
                                                                                         i,
                                                                                         j,
                                                                                         z_face,
                                                                                         element]),
                                                           _zero)),
                                         Val(nvariables(equations_parabolic))))

    # The surface contribution to the x-component of the gradients is the
    # surface flux times the x-component of the outward normal direction of
    # the face (and similarly for the y- and z-components). The normal
    # directions are only used at surface nodes, where the surface fluxes are
    # nonzero.
    normal_direction_x_face = get_normal_direction(x_face, contravariant_vectors,
                                                   i, j, k, element)
    normal_direction_y_face = get_normal_direction(y_face, contravariant_vectors,
                                                   i, j, k, element)
    normal_direction_z_face = get_normal_direction(z_face, contravariant_vectors,
                                                   i, j, k, element)
    surface_node_x = surface_flux_x_face * normal_direction_x_face[1] +
                     surface_flux_y_face * normal_direction_y_face[1] +
                     surface_flux_z_face * normal_direction_z_face[1]
    surface_node_y = surface_flux_x_face * normal_direction_x_face[2] +
                     surface_flux_y_face * normal_direction_y_face[2] +
                     surface_flux_z_face * normal_direction_z_face[2]
    surface_node_z = surface_flux_x_face * normal_direction_x_face[3] +
                     surface_flux_y_face * normal_direction_y_face[3] +
                     surface_flux_z_face * normal_direction_z_face[3]

    # Apply Jacobian from mapping to reference element. In contrast to the
    # hyperbolic part, the sign of the inverse Jacobian is not flipped.
    jacobian_factor = inverse_jacobian[i, j, k, element]
    gradient_x_node = jacobian_factor *
                      (get_node_vars(gradients_x, equations_parabolic, dg,
                                     i, j, k, element) + factor * surface_node_x)
    gradient_y_node = jacobian_factor *
                      (get_node_vars(gradients_y, equations_parabolic, dg,
                                     i, j, k, element) + factor * surface_node_y)
    gradient_z_node = jacobian_factor *
                      (get_node_vars(gradients_z, equations_parabolic, dg,
                                     i, j, k, element) + factor * surface_node_z)
    set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                   i, j, k, element)
    set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                   i, j, k, element)
    set_node_vars!(gradients_z, gradient_z_node, equations_parabolic, dg,
                   i, j, k, element)
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
                                                                                                   MeshT::Type{<:P4estMesh{3}},
                                                                                                   equations_parabolic::AbstractEquationsParabolic{3},
                                                                                                   factor,
                                                                                                   ::Val{NNODES},
                                                                                                   surface_flux_values,
                                                                                                   dg::DGSEM,
                                                                                                   inverse_jacobian) where {
                                                                                                                            NNODES,
                                                                                                                            Source
                                                                                                                            }
    i, j, k, element = @index(Global, NTuple)
    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # As in the hyperbolic kernel, we avoid branching on the position of the
    # node and select the surface flux values of the faces in x, y, and z
    # direction with `ifelse` instead (zero for interior nodes).
    #
    # factor = inverse_weights[1]
    # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    x_node_interface = (i == 1) | (i == NNODES)
    y_node_interface = (j == 1) | (j == NNODES)
    z_node_interface = (k == 1) | (k == NNODES)
    x_face = ifelse(i == 1, 1, 2)
    y_face = ifelse(j == 1, 3, 4)
    z_face = ifelse(k == 1, 5, 6)
    _zero = zero(eltype(du))
    surface_node = SVector(ntuple(@inline(v->ifelse(x_node_interface,
                                                    @inbounds(surface_flux_values[v, j,
                                                                                  k,
                                                                                  x_face,
                                                                                  element]),
                                                    _zero) +
                                             ifelse(y_node_interface,
                                                    @inbounds(surface_flux_values[v, i,
                                                                                  k,
                                                                                  y_face,
                                                                                  element]),
                                                    _zero) +
                                             ifelse(z_node_interface,
                                                    @inbounds(surface_flux_values[v, i,
                                                                                  j,
                                                                                  z_face,
                                                                                  element]),
                                                    _zero)),
                                  Val(nvariables(equations_parabolic))))
    source_node = calc_source_terms_node(u, gradients, t,
                                         source_terms_parabolic,
                                         node_coordinates,
                                         equations_parabolic, dg,
                                         i, j, k, element)
    # Apply Jacobian from mapping to reference element. In contrast to the
    # hyperbolic part, the sign of the inverse Jacobian is not flipped.
    jacobian_factor = inverse_jacobian[i, j, k, element]
    du_local = get_node_vars(du, equations_parabolic, dg, i, j, k, element) +
               factor * surface_node
    du_node = source_node + jacobian_factor * du_local
    set_node_vars!(du, du_node, equations_parabolic, dg, i, j, k, element)
end

Base.@propagate_inbounds function calc_source_terms_node(u,
                                                         gradients::NTuple{3,
                                                                           Any},
                                                         t,
                                                         source_terms_parabolic,
                                                         node_coordinates,
                                                         equations_parabolic,
                                                         dg::DG, indices...)
    gradients_x, gradients_y, gradients_z = gradients
    u_local = get_node_vars(u, equations_parabolic, dg, indices...)
    gradients_x_local = get_node_vars(gradients_x, equations_parabolic, dg,
                                      indices...)
    gradients_y_local = get_node_vars(gradients_y, equations_parabolic, dg,
                                      indices...)
    gradients_z_local = get_node_vars(gradients_z, equations_parabolic, dg,
                                      indices...)
    x_local = get_node_coords(node_coordinates, equations_parabolic, dg, indices...)

    return source_terms_parabolic(u_local,
                                  (gradients_x_local, gradients_y_local,
                                   gradients_z_local),
                                  x_local, t, equations_parabolic)
end

@inline function calc_source_terms_node(u, gradients::NTuple{3, Any}, t,
                                        source_terms_parabolic::Nothing,
                                        node_coordinates,
                                        equations_parabolic,
                                        dg::DG, indices...)
    return zero(SVector{nvariables(equations_parabolic), eltype(u)})
end
end # @muladd
