# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# This method is called when a SemidiscretizationHyperbolicParabolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_parabolic(mesh::P4estMesh{2},
                                equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, parabolic_scheme, RealT, uEltype)
    balance!(mesh)

    elements = init_elements(mesh, equations_hyperbolic, dg.basis, uEltype)
    interfaces = init_interfaces(mesh, equations_hyperbolic, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations_hyperbolic, dg.basis, elements)

    viscous_container = init_viscous_container_2d(nvariables(equations_hyperbolic),
                                                  nnodes(dg.basis), nelements(elements),
                                                  uEltype)

    cache = (; elements, interfaces, boundaries, viscous_container)

    return cache
end

#=
Reusing `rhs_parabolic!` for `TreeMesh`es is not easily possible as
for `P4estMesh`es we call

    ```
    prolong2mortars_divergence!(cache, flux_viscous, mesh, equations_parabolic,
                                dg.mortar, dg.surface_integral, dg)

    calc_mortar_flux_divergence!(cache_parabolic.elements.surface_flux_values,
                                 mesh, equations_parabolic, dg.mortar,
                                 dg.surface_integral, dg, cache)
    ```
instead of
    ```
    prolong2mortars!(cache, flux_viscous, mesh, equations_parabolic,
                     dg.mortar, dg.surface_integral, dg)

    calc_mortar_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                      equations_parabolic,
                      dg.mortar, dg.surface_integral, dg, cache)
    ```
=#
function rhs_parabolic!(du, u, t, mesh::Union{P4estMesh{2}, P4estMesh{3}},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack viscous_container = cache_parabolic
    @unpack u_transformed, gradients, flux_viscous = viscous_container

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, parabolic_scheme, cache, cache_parabolic)
    end

    # Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradient!(gradients, u_transformed, t, mesh, equations_parabolic,
                       boundary_conditions_parabolic, dg, cache, cache_parabolic)
    end

    # Compute and store the viscous fluxes
    @trixi_timeit timer() "calculate viscous fluxes" begin
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_parabolic, dg, cache, cache_parabolic)
    end

    # The remainder of this function is essentially a regular rhs! for parabolic
    # equations (i.e., it computes the divergence of the viscous fluxes)
    #
    # OBS! In `calc_viscous_fluxes!`, the viscous flux values at the volume nodes of each element have
    # been computed and stored in `fluxes_viscous`. In the following, we *reuse* (abuse) the
    # `interfaces` and `boundaries` containers in `cache_parabolic` to interpolate and store the
    # *fluxes* at the element surfaces, as opposed to interpolating and storing the *solution* (as it
    # is done in the hyperbolic operator). That is, `interfaces.u`/`boundaries.u` store *viscous flux values*
    # and *not the solution*.  The advantage is that a) we do not need to allocate more storage, b) we
    # do not need to recreate the existing data structure only with a different name, and c) we do not
    # need to interpolate solutions *and* gradients to the surfaces.

    # TODO: parabolic; reconsider current data structure reuse strategy

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_viscous, mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg, cache)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                             equations_parabolic, dg, cache_parabolic)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache_parabolic, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg, cache)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache_parabolic, t,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic,
                                       dg.surface_integral, dg)
    end

    # Prolong solution to mortars (specialized for AbstractEquationsParabolic)
    # !!! NOTE: we reuse the hyperbolic cache here since it contains "mortars" and "u_threaded". See https://github.com/trixi-framework/Trixi.jl/issues/1674 for a discussion
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars_divergence!(cache, flux_viscous, mesh, equations_parabolic,
                                    dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes (specialized for AbstractEquationsParabolic)
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux_divergence!(cache_parabolic.elements.surface_flux_values,
                                     mesh, equations_parabolic, dg.mortar,
                                     dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache_parabolic)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache_parabolic)
    end

    return nothing
end

function calc_gradient!(gradients, u_transformed, t,
                        mesh::P4estMesh{2}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG,
                        cache, cache_parabolic)
    gradients_x, gradients_y = gradients

    # Reset du
    @trixi_timeit timer() "reset gradients" begin
        reset_du!(gradients_x, dg, cache)
        reset_du!(gradients_y, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        (; derivative_dhat) = dg.basis
        (; contravariant_vectors) = cache.elements

        @threaded for element in eachelement(dg, cache)

            # Calculate gradients with respect to reference coordinates in one element
            for j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j,
                                       element)

                for ii in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i],
                                               u_node,
                                               equations_parabolic, dg, ii, j, element)
                end

                for jj in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j],
                                               u_node,
                                               equations_parabolic, dg, i, jj, element)
                end
            end

            # now that the reference coordinate gradients are computed, transform them node-by-node to physical gradients
            # using the contravariant vectors
            for j in eachnode(dg), i in eachnode(dg)
                Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                      element)
                Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                      element)

                gradients_reference_1 = get_node_vars(gradients_x, equations_parabolic,
                                                      dg,
                                                      i, j, element)
                gradients_reference_2 = get_node_vars(gradients_y, equations_parabolic,
                                                      dg,
                                                      i, j, element)

                # note that the contravariant vectors are transposed compared with computations of flux
                # divergences in `calc_volume_integral!`. See
                # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190
                # for a more detailed discussion.
                gradient_x_node = Ja11 * gradients_reference_1 +
                                  Ja21 * gradients_reference_2
                gradient_y_node = Ja12 * gradients_reference_1 +
                                  Ja22 * gradients_reference_2

                set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg, i,
                               j,
                               element)
                set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg, i,
                               j,
                               element)
            end
        end
    end

    # Prolong solution to interfaces.
    # This reuses `prolong2interfaces` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, u_transformed, mesh,
                            equations_parabolic, dg.surface_integral, dg)
    end

    # Calculate interface fluxes for the gradient.
    # This reuses `calc_interface_flux!` for the purely hyperbolic case.
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache_parabolic.elements.surface_flux_values,
                             mesh, False(), # False() = no nonconservative terms
                             equations_parabolic, dg.surface_integral, dg,
                             cache_parabolic)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache_parabolic, u_transformed, mesh,
                            equations_parabolic, dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_gradients!(cache_parabolic, t, boundary_conditions_parabolic,
                                      mesh, equations_parabolic, dg.surface_integral,
                                      dg)
    end

    # Prolong solution to mortars. This resues the hyperbolic version of `prolong2mortars`
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u_transformed, mesh, equations_parabolic,
                         dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes. This reuses the hyperbolic version of `calc_mortar_flux`,
    # along with a specialization on `calc_mortar_flux!(fstar, ...)` and `mortar_fluxes_to_elements!` for
    # AbstractEquationsParabolic.
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache_parabolic.elements.surface_flux_values,
                          mesh, False(), # False() = no nonconservative terms
                          equations_parabolic,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        (; boundary_interpolation) = dg.basis
        (; surface_flux_values) = cache_parabolic.elements
        (; contravariant_vectors) = cache.elements

        # Access the factors only once before beginning the loop to increase performance.
        # We also use explicit assignments instead of `+=` to let `@muladd` turn these
        # into FMAs (see comment at the top of the file).
        factor_1 = boundary_interpolation[1, 1]
        factor_2 = boundary_interpolation[nnodes(dg), 2]
        @threaded for element in eachelement(dg, cache)
            for l in eachnode(dg)
                for v in eachvariable(equations_parabolic)

                    # Compute x-component of gradients

                    # surface at -x
                    normal_direction_x, _ = get_normal_direction(1,
                                                                 contravariant_vectors,
                                                                 1, l, element)
                    gradients_x[v, 1, l, element] = (gradients_x[v, 1, l, element] +
                                                     surface_flux_values[v, l, 1,
                                                                         element] *
                                                     factor_1 * normal_direction_x)

                    # surface at +x
                    normal_direction_x, _ = get_normal_direction(2,
                                                                 contravariant_vectors,
                                                                 nnodes(dg), l, element)
                    gradients_x[v, nnodes(dg), l, element] = (gradients_x[v, nnodes(dg),
                                                                          l,
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  2,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_x)

                    # surface at -y
                    normal_direction_x, _ = get_normal_direction(3,
                                                                 contravariant_vectors,
                                                                 l, 1, element)
                    gradients_x[v, l, 1, element] = (gradients_x[v, l, 1, element] +
                                                     surface_flux_values[v, l, 3,
                                                                         element] *
                                                     factor_1 * normal_direction_x)

                    # surface at +y
                    normal_direction_x, _ = get_normal_direction(4,
                                                                 contravariant_vectors,
                                                                 l, nnodes(dg), element)
                    gradients_x[v, l, nnodes(dg), element] = (gradients_x[v, l,
                                                                          nnodes(dg),
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  4,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_x)

                    # Compute y-component of gradients

                    # surface at -x
                    _, normal_direction_y = get_normal_direction(1,
                                                                 contravariant_vectors,
                                                                 1, l, element)
                    gradients_y[v, 1, l, element] = (gradients_y[v, 1, l, element] +
                                                     surface_flux_values[v, l, 1,
                                                                         element] *
                                                     factor_1 * normal_direction_y)

                    # surface at +x
                    _, normal_direction_y = get_normal_direction(2,
                                                                 contravariant_vectors,
                                                                 nnodes(dg), l, element)
                    gradients_y[v, nnodes(dg), l, element] = (gradients_y[v, nnodes(dg),
                                                                          l,
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  2,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_y)

                    # surface at -y
                    _, normal_direction_y = get_normal_direction(3,
                                                                 contravariant_vectors,
                                                                 l, 1, element)
                    gradients_y[v, l, 1, element] = (gradients_y[v, l, 1, element] +
                                                     surface_flux_values[v, l, 3,
                                                                         element] *
                                                     factor_1 * normal_direction_y)

                    # surface at +y
                    _, normal_direction_y = get_normal_direction(4,
                                                                 contravariant_vectors,
                                                                 l, nnodes(dg), element)
                    gradients_y[v, l, nnodes(dg), element] = (gradients_y[v, l,
                                                                          nnodes(dg),
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  4,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_y)
                end
            end
        end
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients_x, mesh, equations_parabolic, dg,
                                  cache_parabolic)
        apply_jacobian_parabolic!(gradients_y, mesh, equations_parabolic, dg,
                                  cache_parabolic)
    end

    return nothing
end

# This version is called during `calc_gradients!` and must be specialized because the
# flux for the gradient is {u}, which doesn't depend on the outward normal. Thus,
# you don't need to scale by 2 (e.g., the scaling factor in the normals (and in the
# contravariant vectors) along large/small elements across a non-conforming
# interface in 2D) and flip the sign when storing the mortar fluxes back
# into `surface_flux_values`.
@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                            equations::AbstractEquationsParabolic,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache, mortar, fstar, u_buffer)
    @unpack neighbor_ids, node_indices = cache.mortars
    # Copy solution small to small
    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    for position in 1:2
        element = neighbor_ids[position, mortar]
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, i, small_direction, element] = fstar[position][v,
                                                                                      i]
            end
        end
    end

    # Project small fluxes to large element.
    multiply_dimensionwise!(u_buffer,
                            mortar_l2.reverse_upper, fstar[2],
                            mortar_l2.reverse_lower, fstar[1])

    # Copy interpolated flux values from buffer to large element face in the
    # correct orientation.
    # Note that the index of the small sides will always run forward but
    # the index of the large side might need to run backwards for flipped sides.
    large_element = neighbor_ids[3, mortar]
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)

    if :i_backward in large_indices
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, end + 1 - i, large_direction, large_element] = u_buffer[v,
                                                                                               i]
            end
        end
    else
        for i in eachnode(dg)
            for v in eachvariable(equations)
                surface_flux_values[v, i, large_direction, large_element] = u_buffer[v,
                                                                                     i]
            end
        end
    end

    return nothing
end

# This version is used for parabolic gradient computations
@inline function calc_interface_flux!(surface_flux_values, mesh::P4estMesh{2},
                                      nonconservative_terms::False,
                                      equations::AbstractEquationsParabolic,
                                      surface_integral, dg::DG, cache,
                                      interface_index, normal_direction,
                                      primary_node_index, primary_direction_index,
                                      primary_element_index,
                                      secondary_node_index, secondary_direction_index,
                                      secondary_element_index)
    @unpack u = cache.interfaces
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_node_index,
                                       interface_index)

    flux_ = 0.5f0 * (u_ll + u_rr) # we assume that the gradient computations utilize a central flux

    # Note that we don't flip the sign on the secondary flux. This is because for parabolic terms,
    # the normals are not embedded in `flux_` for the parabolic gradient computations.
    for v in eachvariable(equations)
        surface_flux_values[v, primary_node_index, primary_direction_index, primary_element_index] = flux_[v]
        surface_flux_values[v, secondary_node_index, secondary_direction_index, secondary_element_index] = flux_[v]
    end
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_volume_integral!(du, flux_viscous,
                               mesh::P4estMesh{2},
                               equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
    (; derivative_dhat) = dg.basis
    (; contravariant_vectors) = cache.elements
    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for element in eachelement(dg, cache)
        # Calculate volume terms in one element
        for j in eachnode(dg), i in eachnode(dg)
            flux1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j,
                                  element)
            flux2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j,
                                  element)

            # Compute the contravariant flux by taking the scalar product of the
            # first contravariant vector Ja^1 and the flux vector
            Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                  element)
            contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
            for ii in eachnode(dg)
                multiply_add_to_node_vars!(du, derivative_dhat[ii, i],
                                           contravariant_flux1,
                                           equations_parabolic, dg, ii, j, element)
            end

            # Compute the contravariant flux by taking the scalar product of the
            # second contravariant vector Ja^2 and the flux vector
            Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                  element)
            contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
            for jj in eachnode(dg)
                multiply_add_to_node_vars!(du, derivative_dhat[jj, j],
                                           contravariant_flux2,
                                           equations_parabolic, dg, i, jj, element)
            end
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache_parabolic, flux_viscous,
                             mesh::P4estMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
    (; interfaces) = cache_parabolic
    (; contravariant_vectors) = cache_parabolic.elements
    index_range = eachnode(dg)
    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for interface in eachinterface(dg, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        primary_element = interfaces.neighbor_ids[1, interface]
        primary_indices = interfaces.node_indices[1, interface]
        primary_direction = indices2direction(primary_indices)

        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start
        for i in eachnode(dg)

            # this is the outward normal direction on the primary element
            normal_direction = get_normal_direction(primary_direction,
                                                    contravariant_vectors,
                                                    i_primary, j_primary,
                                                    primary_element)

            for v in eachvariable(equations_parabolic)
                # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
                flux_viscous = SVector(flux_viscous_x[v, i_primary, j_primary,
                                                      primary_element],
                                       flux_viscous_y[v, i_primary, j_primary,
                                                      primary_element])

                interfaces.u[1, v, i, interface] = dot(flux_viscous, normal_direction)
            end
            i_primary += i_primary_step
            j_primary += j_primary_step
        end

        # Copy solution data from the secondary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        secondary_element = interfaces.neighbor_ids[2, interface]
        secondary_indices = interfaces.node_indices[2, interface]
        secondary_direction = indices2direction(secondary_indices)

        i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
                                                                     index_range)
        j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
                                                                     index_range)

        i_secondary = i_secondary_start
        j_secondary = j_secondary_start
        for i in eachnode(dg)
            # This is the outward normal direction on the secondary element.
            # Here, we assume that normal_direction on the secondary element is
            # the negative of normal_direction on the primary element.
            normal_direction = get_normal_direction(secondary_direction,
                                                    contravariant_vectors,
                                                    i_secondary, j_secondary,
                                                    secondary_element)

            for v in eachvariable(equations_parabolic)
                # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
                flux_viscous = SVector(flux_viscous_x[v, i_secondary, j_secondary,
                                                      secondary_element],
                                       flux_viscous_y[v, i_secondary, j_secondary,
                                                      secondary_element])
                # store the normal flux with respect to the primary normal direction
                interfaces.u[2, v, i, interface] = -dot(flux_viscous, normal_direction)
            end
            i_secondary += i_secondary_step
            j_secondary += j_secondary_step
        end
    end

    return nothing
end

# This version is used for divergence flux computations
function calc_interface_flux!(surface_flux_values,
                              mesh::P4estMesh{2}, equations_parabolic,
                              dg::DG, cache_parabolic)
    (; neighbor_ids, node_indices) = cache_parabolic.interfaces
    (; contravariant_vectors) = cache_parabolic.elements
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachinterface(dg, cache_parabolic)
        # Get element and side index information on the primary element
        primary_element = neighbor_ids[1, interface]
        primary_indices = node_indices[1, interface]
        primary_direction_index = indices2direction(primary_indices)

        # Create the local i,j indexing on the primary element used to pull normal direction information
        i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1],
                                                                 index_range)
        j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2],
                                                                 index_range)

        i_primary = i_primary_start
        j_primary = j_primary_start

        # Get element and side index information on the secondary element
        secondary_element = neighbor_ids[2, interface]
        secondary_indices = node_indices[2, interface]
        secondary_direction_index = indices2direction(secondary_indices)

        # Initiate the secondary index to be used in the surface for loop.
        # This index on the primary side will always run forward but
        # the secondary index might need to run backwards for flipped sides.
        if :i_backward in secondary_indices
            node_secondary = index_end
            node_secondary_step = -1
        else
            node_secondary = 1
            node_secondary_step = 1
        end

        for node in eachnode(dg)
            # We prolong the viscous flux dotted with respect the outward normal on the
            # primary element. We assume a BR-1 type of flux.
            viscous_flux_normal_ll, viscous_flux_normal_rr = get_surface_node_vars(cache_parabolic.interfaces.u,
                                                                                   equations_parabolic,
                                                                                   dg,
                                                                                   node,
                                                                                   interface)

            flux = 0.5f0 * (viscous_flux_normal_ll + viscous_flux_normal_rr)

            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, node, primary_direction_index, primary_element] = flux[v]
                surface_flux_values[v, node_secondary, secondary_direction_index, secondary_element] = -flux[v]
            end

            # Increment primary element indices to pull the normal direction
            i_primary += i_primary_step
            j_primary += j_primary_step
            # Increment the surface node index along the secondary element
            node_secondary += node_secondary_step
        end
    end

    return nothing
end

function prolong2mortars_divergence!(cache, flux_viscous::Vector{Array{uEltype, 4}},
                                     mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                     equations,
                                     mortar_l2::LobattoLegendreMortarL2,
                                     surface_integral,
                                     dg::DGSEM) where {uEltype <: Real}
    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for mortar in eachmortar(dg, cache)
        # Copy solution data from the small elements using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        small_indices = node_indices[1, mortar]
        direction_index = indices2direction(small_indices)

        i_small_start, i_small_step = index_to_start_step_2d(small_indices[1],
                                                             index_range)
        j_small_start, j_small_step = index_to_start_step_2d(small_indices[2],
                                                             index_range)

        for position in 1:2
            i_small = i_small_start
            j_small = j_small_start
            element = neighbor_ids[position, mortar]
            for i in eachnode(dg)
                normal_direction = get_normal_direction(direction_index,
                                                        contravariant_vectors,
                                                        i_small, j_small, element)

                for v in eachvariable(equations)
                    flux_viscous = SVector(flux_viscous_x[v, i_small, j_small, element],
                                           flux_viscous_y[v, i_small, j_small, element])

                    cache.mortars.u[1, v, position, i, mortar] = dot(flux_viscous,
                                                                     normal_direction)
                end
                i_small += i_small_step
                j_small += j_small_step
            end
        end

        # Buffer to copy solution values of the large element in the correct orientation
        # before interpolating
        u_buffer = cache.u_threaded[Threads.threadid()]

        # Copy solution of large element face to buffer in the
        # correct orientation
        large_indices = node_indices[2, mortar]
        direction_index = indices2direction(large_indices)

        i_large_start, i_large_step = index_to_start_step_2d(large_indices[1],
                                                             index_range)
        j_large_start, j_large_step = index_to_start_step_2d(large_indices[2],
                                                             index_range)

        i_large = i_large_start
        j_large = j_large_start
        element = neighbor_ids[3, mortar]
        for i in eachnode(dg)
            normal_direction = get_normal_direction(direction_index,
                                                    contravariant_vectors,
                                                    i_large, j_large, element)

            for v in eachvariable(equations)
                flux_viscous = SVector(flux_viscous_x[v, i_large, j_large, element],
                                       flux_viscous_y[v, i_large, j_large, element])

                # We prolong the viscous flux dotted with respect the outward normal
                # on the small element. We scale by -1/2 here because the normal
                # direction on the large element is negative 2x that of the small
                # element (these normal directions are "scaled" by the surface Jacobian)
                u_buffer[v, i] = -0.5f0 * dot(flux_viscous, normal_direction)
            end
            i_large += i_large_step
            j_large += j_large_step
        end

        # Interpolate large element face data from buffer to small face locations
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, mortar),
                                mortar_l2.forward_lower,
                                u_buffer)
        multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, mortar),
                                mortar_l2.forward_upper,
                                u_buffer)
    end

    return nothing
end

# We specialize `calc_mortar_flux!` for the divergence part of
# the parabolic terms.
function calc_mortar_flux_divergence!(surface_flux_values,
                                      mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                      equations::AbstractEquationsParabolic,
                                      mortar_l2::LobattoLegendreMortarL2,
                                      surface_integral, dg::DG, cache)
    @unpack neighbor_ids, node_indices = cache.mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_upper_threaded, fstar_lower_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = (fstar_lower_threaded[Threads.threadid()],
                 fstar_upper_threaded[Threads.threadid()])

        for position in 1:2
            for node in eachnode(dg)
                for v in eachvariable(equations)
                    viscous_flux_normal_ll = cache.mortars.u[1, v, position, node,
                                                             mortar]
                    viscous_flux_normal_rr = cache.mortars.u[2, v, position, node,
                                                             mortar]

                    # TODO: parabolic; only BR1 at the moment
                    fstar[position][v, node] = 0.5f0 * (viscous_flux_normal_ll +
                                                viscous_flux_normal_rr)
                end
            end
        end

        # Buffer to interpolate flux values of the large element to before
        # copying in the correct orientation
        u_buffer = cache.u_threaded[Threads.threadid()]

        # this reuses the hyperbolic version of `mortar_fluxes_to_elements!`
        mortar_fluxes_to_elements!(surface_flux_values,
                                   mesh, equations, mortar_l2, dg, cache,
                                   mortar, fstar, u_buffer)
    end

    return nothing
end

# We structure `calc_interface_flux!` similarly to "calc_mortar_flux!" for
# hyperbolic  equations with no nonconservative terms.
# The reasoning is that parabolic fluxes are treated like conservative
# terms (e.g., we compute a viscous conservative "flux") and thus no
# non-conservative terms are present.
@inline function calc_mortar_flux!(fstar,
                                   mesh::Union{P4estMesh{2}, T8codeMesh{2}},
                                   nonconservative_terms::False,
                                   equations::AbstractEquationsParabolic,
                                   surface_integral, dg::DG, cache,
                                   mortar_index, position_index, normal_direction,
                                   node_index)
    @unpack u = cache.mortars
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, node_index,
                                       mortar_index)

    # TODO: parabolic; only BR1 at the moment
    flux_ = 0.5f0 * (u_ll + u_rr)

    # Copy flux to buffer
    set_node_vars!(fstar[position_index], flux_, equations, dg, node_index)
end

# TODO: parabolic, finish implementing `calc_boundary_flux_gradients!` and `calc_boundary_flux_divergence!`
function prolong2boundaries!(cache_parabolic, flux_viscous,
                             mesh::P4estMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
    (; boundaries) = cache_parabolic
    (; contravariant_vectors) = cache_parabolic.elements
    index_range = eachnode(dg)

    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for boundary in eachboundary(dg, cache_parabolic)
        # Copy solution data from the element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        element = boundaries.neighbor_ids[boundary]
        node_indices = boundaries.node_indices[boundary]
        direction = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for i in eachnode(dg)
            # this is the outward normal direction on the primary element
            normal_direction = get_normal_direction(direction, contravariant_vectors,
                                                    i_node, j_node, element)

            for v in eachvariable(equations_parabolic)
                flux_viscous = SVector(flux_viscous_x[v, i_node, j_node, element],
                                       flux_viscous_y[v, i_node, j_node, element])

                boundaries.u[v, i, boundary] = dot(flux_viscous, normal_direction)
            end
            i_node += i_node_step
            j_node += j_node_step
        end
    end
    return nothing
end

function calc_boundary_flux_gradients!(cache, t,
                                       boundary_condition::Union{BoundaryConditionPeriodic,
                                                                 BoundaryConditionDoNothing},
                                       mesh::P4estMesh, equations, surface_integral,
                                       dg::DG)
    @assert isempty(eachboundary(dg, cache))
end

# Function barrier for type stability
function calc_boundary_flux_gradients!(cache, t, boundary_conditions, mesh::P4estMesh,
                                       equations, surface_integral, dg::DG)
    (; boundary_condition_types, boundary_indices) = boundary_conditions

    calc_boundary_flux_by_type!(cache, t, boundary_condition_types, boundary_indices,
                                Gradient(), mesh, equations, surface_integral, dg)
    return nothing
end

function calc_boundary_flux_divergence!(cache, t, boundary_conditions, mesh::P4estMesh,
                                        equations, surface_integral, dg::DG)
    (; boundary_condition_types, boundary_indices) = boundary_conditions

    calc_boundary_flux_by_type!(cache, t, boundary_condition_types, boundary_indices,
                                Divergence(), mesh, equations, surface_integral, dg)
    return nothing
end

# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(cache, t, BCs::NTuple{N, Any},
                                     BC_indices::NTuple{N, Vector{Int}},
                                     operator_type,
                                     mesh::P4estMesh,
                                     equations, surface_integral, dg::DG) where {N}
    # Extract the boundary condition type and index vector
    boundary_condition = first(BCs)
    boundary_condition_indices = first(BC_indices)
    # Extract the remaining types and indices to be processed later
    remaining_boundary_conditions = Base.tail(BCs)
    remaining_boundary_condition_indices = Base.tail(BC_indices)

    # process the first boundary condition type
    calc_boundary_flux!(cache, t, boundary_condition, boundary_condition_indices,
                        operator_type, mesh, equations, surface_integral, dg)

    # recursively call this method with the unprocessed boundary types
    calc_boundary_flux_by_type!(cache, t, remaining_boundary_conditions,
                                remaining_boundary_condition_indices,
                                operator_type,
                                mesh, equations, surface_integral, dg)

    return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     operator_type, mesh::P4estMesh, equations,
                                     surface_integral, dg::DG)
    nothing
end

function calc_boundary_flux!(cache, t,
                             boundary_condition_parabolic, # works with Dict types
                             boundary_condition_indices,
                             operator_type, mesh::P4estMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG)
    (; boundaries) = cache
    (; node_coordinates, surface_flux_values) = cache.elements
    (; contravariant_vectors) = cache.elements
    index_range = eachnode(dg)

    @threaded for local_index in eachindex(boundary_condition_indices)
        # Use the local index to get the global boundary index from the pre-sorted list
        boundary_index = boundary_condition_indices[local_index]

        # Get information on the adjacent element, compute the surface fluxes,
        # and store them
        element = boundaries.neighbor_ids[boundary_index]
        node_indices = boundaries.node_indices[boundary_index]
        direction_index = indices2direction(node_indices)

        i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
        j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

        i_node = i_node_start
        j_node = j_node_start
        for node_index in eachnode(dg)
            # Extract solution data from boundary container
            u_inner = get_node_vars(boundaries.u, equations_parabolic, dg, node_index,
                                    boundary_index)

            # Outward-pointing normal direction (not normalized)
            normal_direction = get_normal_direction(direction_index,
                                                    contravariant_vectors,
                                                    i_node, j_node, element)

            # TODO: revisit if we want more general boundary treatments.
            # This assumes the gradient numerical flux at the boundary is the gradient variable,
            # which is consistent with BR1, LDG.
            flux_inner = u_inner

            # Coordinates at boundary node
            x = get_node_coords(node_coordinates, equations_parabolic, dg, i_node,
                                j_node,
                                element)

            flux_ = boundary_condition_parabolic(flux_inner, u_inner, normal_direction,
                                                 x, t, operator_type,
                                                 equations_parabolic)

            # Copy flux to element storage in the correct orientation
            for v in eachvariable(equations_parabolic)
                surface_flux_values[v, node_index, direction_index, element] = flux_[v]
            end

            i_node += i_node_step
            j_node += j_node_step
        end
    end
end

function apply_jacobian_parabolic!(du, mesh::P4estMesh{2},
                                   equations::AbstractEquationsParabolic,
                                   dg::DG, cache)
    @unpack inverse_jacobian = cache.elements

    @threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            factor = inverse_jacobian[i, j, element]

            for v in eachvariable(equations)
                du[v, i, j, element] *= factor
            end
        end
    end

    return nothing
end
end # @muladd
