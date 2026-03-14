# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

function rhs_parabolic!(du, u, t, mesh::StructuredMesh{2},
                        equations_parabolic::AbstractEquationsParabolic,
                        boundary_conditions_parabolic, source_terms_parabolic,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack viscous_container = cache_parabolic
    @unpack u_transformed, gradients, flux_viscous = viscous_container

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, cache)
    end

    # Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradient!(gradients, u_transformed, t, mesh,
                       equations_parabolic, boundary_conditions_parabolic,
                       dg, parabolic_scheme, cache)
    end

    # Compute and store the viscous fluxes
    @trixi_timeit timer() "calculate viscous fluxes" begin
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_parabolic, dg, cache)
    end

    # The remainder of this function is essentially a regular rhs! for parabolic
    # equations (i.e., it computes the divergence of the viscous fluxes).
    #
    # OBS! In `calc_viscous_fluxes!`, the viscous flux values at the volume nodes of each element have
    # been computed and stored in `flux_viscous`. In the following, we *reuse* (abuse) the
    # `interfaces_u` container in `cache.elements` to interpolate and store the
    # *fluxes* at the element surfaces, as opposed to interpolating and storing the *solution* (as it
    # is done in the hyperbolic operator). That is, `interfaces_u` stores *viscous flux values*
    # (normal components) and *not the solution*. The advantage is that a) we do not need to allocate
    # more storage, b) we do not need to recreate the existing data structure only with a different
    # name, and c) we do not need to interpolate solutions *and* gradients to the surfaces.

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" set_zero!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_viscous, mesh, equations_parabolic, dg, cache)
    end

    # Prolong viscous fluxes (as normal components) to interfaces.
    # Uses Tuple dispatch to avoid ambiguity with the hyperbolic `prolong2interfaces!`.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, flux_viscous, mesh, equations_parabolic, dg)
    end

    # Calculate interface fluxes for the divergence
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache.elements.surface_flux_values, mesh,
                             equations_parabolic, dg, parabolic_scheme, cache)
    end

    # Calculate boundary fluxes for the divergence.
    # `prolong2boundaries!` is not required for `StructuredMesh` since boundary values
    # are stored in `interfaces_u` after `prolong2interfaces!` above.
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache, t,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic,
                                       dg.surface_integral, dg)
    end

    # Calculate surface integrals.
    # Reuses `calc_surface_integral!` for the purely hyperbolic StructuredMesh case.
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    end

    @trixi_timeit timer() "source terms parabolic" begin
        calc_sources_parabolic!(du, u, gradients, t, source_terms_parabolic,
                                equations_parabolic, dg, cache)
    end

    return nothing
end

function calc_gradient!(gradients, u_transformed, t,
                        mesh::StructuredMesh{2},
                        equations_parabolic, boundary_conditions_parabolic,
                        dg::DG, parabolic_scheme, cache)
    # Reset gradients
    @trixi_timeit timer() "reset gradients" begin
        reset_gradients!(gradients, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral_gradient!(gradients, u_transformed,
                                       mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces.
    # This reuses `prolong2interfaces!` for the purely hyperbolic StructuredMesh case.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache, u_transformed, mesh,
                            equations_parabolic, dg)
    end

    # Calculate interface fluxes for the gradient.
    # `prolong2boundaries!` is not required for `StructuredMesh` — boundary values
    # are already available in `interfaces_u` after the prolong step above.
    @trixi_timeit timer() "interface flux" begin
        @unpack surface_flux_values = cache.elements
        calc_interface_flux_gradient!(surface_flux_values, mesh, equations_parabolic,
                                      dg, parabolic_scheme, cache)
    end

    # Calculate boundary fluxes for the gradient
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_gradient!(cache, t, boundary_conditions_parabolic,
                                     mesh, equations_parabolic, dg.surface_integral,
                                     dg)
    end

    # Calculate surface integrals for the gradient
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral_gradient!(gradients, mesh, equations_parabolic,
                                        dg, cache)
    end

    # Apply Jacobian from mapping to reference element.
    # Uses the AbstractMesh{2} dispatch which already covers StructuredMesh.
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients, mesh, equations_parabolic, dg,
                                  cache)
    end

    return nothing
end

# Outer loop: dispatch over elements using left_neighbors (x- and y-interfaces)
function calc_interface_flux_gradient!(surface_flux_values,
                                       mesh::StructuredMesh{2},
                                       equations_parabolic,
                                       dg::DG, parabolic_scheme, cache)
    @unpack elements = cache

    @threaded for element in eachelement(dg, cache)
        # Interfaces in x-direction (orientation = 1)
        calc_interface_flux_gradient!(surface_flux_values,
                                      elements.left_neighbors[1, element],
                                      element, 1, mesh,
                                      equations_parabolic, dg, parabolic_scheme, cache)

        # Interfaces in y-direction (orientation = 2)
        calc_interface_flux_gradient!(surface_flux_values,
                                      elements.left_neighbors[2, element],
                                      element, 2, mesh,
                                      equations_parabolic, dg, parabolic_scheme, cache)
    end

    return nothing
end

# Inner kernel: compute gradient flux at a single interface
@inline function calc_interface_flux_gradient!(surface_flux_values, left_element,
                                               right_element, orientation,
                                               mesh::StructuredMesh{2},
                                               equations_parabolic, dg::DG,
                                               parabolic_scheme, cache)
    # Skip boundary elements (left_element == 0 at domain boundaries)
    if left_element <= 0
        return nothing
    end

    @unpack interfaces_u, contravariant_vectors, inverse_jacobian = cache.elements

    right_direction = 2 * orientation
    left_direction = right_direction - 1

    for i in eachnode(dg)
        u_ll = get_node_vars(interfaces_u, equations_parabolic, dg, i,
                             right_direction, left_element)
        u_rr = get_node_vars(interfaces_u, equations_parabolic, dg, i,
                             left_direction, right_element)

        # The normal direction is from left_element to right_element.
        # Use the right element's left-face node to compute the contravariant vector.
        # The sign of inverse_jacobian determines whether the mapping is
        # orientation-reversing; if so, we flip the normal to maintain correct orientation.
        if orientation == 1
            sign_jac = sign(inverse_jacobian[1, i, right_element])
            normal_direction = sign_jac *
                               get_contravariant_vector(1, contravariant_vectors,
                                                        1, i, right_element)
        else # orientation == 2
            sign_jac = sign(inverse_jacobian[i, 1, right_element])
            normal_direction = sign_jac *
                               get_contravariant_vector(2, contravariant_vectors,
                                                        i, 1, right_element)
        end

        flux_ = flux_parabolic(u_ll, u_rr, normal_direction, Gradient(),
                               equations_parabolic, parabolic_scheme)

        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, i, right_direction, left_element] = flux_[v]
            # No sign flip required for gradient: the normal direction is not embedded
            # in flux_ for gradient computations
            surface_flux_values[v, i, left_direction, right_element] = flux_[v]
        end
    end

    return nothing
end

# Do nothing for periodic meshes
function calc_boundary_flux_gradient!(cache, t,
                                      boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                      mesh::StructuredMesh{2},
                                      equations_parabolic, surface_integral, dg::DG)
    @assert isperiodic(mesh)
    return nothing
end

function calc_boundary_flux_gradient!(cache, t,
                                      boundary_conditions_parabolic::NamedTuple,
                                      mesh::StructuredMesh{2},
                                      equations_parabolic, surface_integral, dg::DG)
    @unpack surface_flux_values, node_coordinates, contravariant_vectors,
    inverse_jacobian, interfaces_u = cache.elements
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2)
        # Negative x-direction (direction 1)
        direction = 1
        element = linear_indices[begin, cell_y]
        for j in eachnode(dg)
            calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                                      boundary_conditions_parabolic[direction],
                                                      1, # orientation
                                                      direction, j,
                                                      mesh, equations_parabolic,
                                                      surface_integral, dg,
                                                      interfaces_u, node_coordinates,
                                                      contravariant_vectors,
                                                      inverse_jacobian, element)
        end

        # Positive x-direction (direction 2)
        direction = 2
        element = linear_indices[end, cell_y]
        for j in eachnode(dg)
            calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                                      boundary_conditions_parabolic[direction],
                                                      1, # orientation
                                                      direction, j,
                                                      mesh, equations_parabolic,
                                                      surface_integral, dg,
                                                      interfaces_u, node_coordinates,
                                                      contravariant_vectors,
                                                      inverse_jacobian, element)
        end
    end

    for cell_x in axes(mesh, 1)
        # Negative y-direction (direction 3)
        direction = 3
        element = linear_indices[cell_x, begin]
        for i in eachnode(dg)
            calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                                      boundary_conditions_parabolic[direction],
                                                      2, # orientation
                                                      direction, i,
                                                      mesh, equations_parabolic,
                                                      surface_integral, dg,
                                                      interfaces_u, node_coordinates,
                                                      contravariant_vectors,
                                                      inverse_jacobian, element)
        end

        # Positive y-direction (direction 4)
        direction = 4
        element = linear_indices[cell_x, end]
        for i in eachnode(dg)
            calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                                      boundary_conditions_parabolic[direction],
                                                      2, # orientation
                                                      direction, i,
                                                      mesh, equations_parabolic,
                                                      surface_integral, dg,
                                                      interfaces_u, node_coordinates,
                                                      contravariant_vectors,
                                                      inverse_jacobian, element)
        end
    end

    return nothing
end

# Compute the gradient boundary flux at a single boundary face node.
# `face_node_index` is the index along the face (y-index for x-faces, x-index for y-faces).
@inline function calc_boundary_flux_by_direction_gradient!(surface_flux_values, t,
                                                           boundary_condition,
                                                           orientation, direction,
                                                           face_node_index,
                                                           mesh::StructuredMesh{2},
                                                           equations_parabolic,
                                                           surface_integral, dg::DG,
                                                           interfaces_u,
                                                           node_coordinates,
                                                           contravariant_vectors,
                                                           inverse_jacobian, element)
    # Map face node index and direction to volume node indices
    if orientation == 1 # x-faces
        if direction == 1 # -x face, leftmost column
            node_i, node_j = 1, face_node_index
        else              # +x face, rightmost column
            node_i, node_j = nnodes(dg), face_node_index
        end
    else # y-faces
        if direction == 3 # -y face, bottom row
            node_i, node_j = face_node_index, 1
        else              # +y face, top row
            node_i, node_j = face_node_index, nnodes(dg)
        end
    end

    # Boundary values for StructuredMesh are stored in interfaces_u
    u_inner = get_node_vars(interfaces_u, equations_parabolic, dg, face_node_index,
                            direction, element)
    x = get_node_coords(node_coordinates, equations_parabolic, dg, node_i, node_j,
                        element)

    # If the mapping is orientation-reversing, the contravariant vectors' orientation
    # is reversed as well. The outward normal must be correctly oriented.
    sign_jac = sign(inverse_jacobian[node_i, node_j, element])
    outward_normal = sign_jac *
                     get_normal_direction(direction, contravariant_vectors, node_i,
                                          node_j,
                                          element)

    flux_ = boundary_condition(u_inner, u_inner, outward_normal, x, t, Gradient(),
                               equations_parabolic)

    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, face_node_index, direction, element] = flux_[v]
    end

    return nothing
end

# Accumulate gradient surface-integral corrections from surface_flux_values into gradients.
# The factor `inverse_weights[1]` is the LGL boundary weight (= 1/w_boundary for LGL).
# We apply the outward-pointing normal (including the orientation-reversing sign correction)
# to decompose the scalar surface flux into physical x- and y-gradient contributions.
function calc_surface_integral_gradient!(gradients,
                                         mesh::StructuredMesh{2},
                                         equations_parabolic::AbstractEquationsParabolic,
                                         dg::DGSEM, cache)
    @unpack inverse_weights = dg.basis
    @unpack surface_flux_values, contravariant_vectors, inverse_jacobian = cache.elements

    gradients_x, gradients_y = gradients

    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor = inverse_weights[1] # For LGL basis: Identical to weighted boundary interpolation at x = ±1
    @threaded for element in eachelement(dg, cache)
        for l in eachnode(dg)
            for v in eachvariable(equations_parabolic)

                # Compute x-component of gradients

                # surface at -x (direction 1, i=1)
                sign_jac = sign(inverse_jacobian[1, l, element])
                normal_direction_x, _ = sign_jac *
                                        get_normal_direction(1, contravariant_vectors,
                                                             1,
                                                             l, element)
                gradients_x[v, 1, l, element] = (gradients_x[v, 1, l, element] +
                                                 surface_flux_values[v, l, 1, element] *
                                                 factor * normal_direction_x)

                # surface at +x (direction 2, i=N)
                sign_jac = sign(inverse_jacobian[nnodes(dg), l, element])
                normal_direction_x, _ = sign_jac *
                                        get_normal_direction(2, contravariant_vectors,
                                                             nnodes(dg), l, element)
                gradients_x[v, nnodes(dg), l, element] = (gradients_x[v, nnodes(dg), l,
                                                                      element] +
                                                          surface_flux_values[v, l, 2,
                                                                              element] *
                                                          factor * normal_direction_x)

                # surface at -y (direction 3, j=1)
                sign_jac = sign(inverse_jacobian[l, 1, element])
                normal_direction_x, _ = sign_jac *
                                        get_normal_direction(3, contravariant_vectors,
                                                             l,
                                                             1, element)
                gradients_x[v, l, 1, element] = (gradients_x[v, l, 1, element] +
                                                 surface_flux_values[v, l, 3, element] *
                                                 factor * normal_direction_x)

                # surface at +y (direction 4, j=N)
                sign_jac = sign(inverse_jacobian[l, nnodes(dg), element])
                normal_direction_x, _ = sign_jac *
                                        get_normal_direction(4, contravariant_vectors,
                                                             l,
                                                             nnodes(dg), element)
                gradients_x[v, l, nnodes(dg), element] = (gradients_x[v, l, nnodes(dg),
                                                                      element] +
                                                          surface_flux_values[v, l, 4,
                                                                              element] *
                                                          factor * normal_direction_x)

                # Compute y-component of gradients

                # surface at -x (direction 1, i=1)
                sign_jac = sign(inverse_jacobian[1, l, element])
                _, normal_direction_y = sign_jac *
                                        get_normal_direction(1, contravariant_vectors,
                                                             1,
                                                             l, element)
                gradients_y[v, 1, l, element] = (gradients_y[v, 1, l, element] +
                                                 surface_flux_values[v, l, 1, element] *
                                                 factor * normal_direction_y)

                # surface at +x (direction 2, i=N)
                sign_jac = sign(inverse_jacobian[nnodes(dg), l, element])
                _, normal_direction_y = sign_jac *
                                        get_normal_direction(2, contravariant_vectors,
                                                             nnodes(dg), l, element)
                gradients_y[v, nnodes(dg), l, element] = (gradients_y[v, nnodes(dg), l,
                                                                      element] +
                                                          surface_flux_values[v, l, 2,
                                                                              element] *
                                                          factor * normal_direction_y)

                # surface at -y (direction 3, j=1)
                sign_jac = sign(inverse_jacobian[l, 1, element])
                _, normal_direction_y = sign_jac *
                                        get_normal_direction(3, contravariant_vectors,
                                                             l,
                                                             1, element)
                gradients_y[v, l, 1, element] = (gradients_y[v, l, 1, element] +
                                                 surface_flux_values[v, l, 3, element] *
                                                 factor * normal_direction_y)

                # surface at +y (direction 4, j=N)
                sign_jac = sign(inverse_jacobian[l, nnodes(dg), element])
                _, normal_direction_y = sign_jac *
                                        get_normal_direction(4, contravariant_vectors,
                                                             l,
                                                             nnodes(dg), element)
                gradients_y[v, l, nnodes(dg), element] = (gradients_y[v, l, nnodes(dg),
                                                                      element] +
                                                          surface_flux_values[v, l, 4,
                                                                              element] *
                                                          factor * normal_direction_y)
            end
        end
    end

    return nothing
end

# Prolong viscous fluxes (as normal components) to interfaces for the divergence step.
# Specialization `flux_viscous::Tuple` is needed to avoid ambiguity with the hyperbolic
# version of `prolong2interfaces!` which takes `u::AbstractArray`.
# We store dot(F_visc, outward_normal) into interfaces_u for later use in
# `calc_interface_flux!` and `calc_boundary_flux_divergence!`.
function prolong2interfaces!(cache, flux_viscous::Tuple,
                             mesh::StructuredMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             dg::DG)
    @unpack interfaces_u, contravariant_vectors, inverse_jacobian = cache.elements
    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for element in eachelement(dg, cache)
        for i in eachnode(dg)
            # Direction 1: negative x-face, volume node at (1, i)
            sign_jac = sign(inverse_jacobian[1, i, element])
            normal = sign_jac *
                     get_normal_direction(1, contravariant_vectors, 1, i, element)
            for v in eachvariable(equations_parabolic)
                fvisc = SVector(flux_viscous_x[v, 1, i, element],
                                flux_viscous_y[v, 1, i, element])
                interfaces_u[v, i, 1, element] = dot(fvisc, normal)
            end

            # Direction 2: positive x-face, volume node at (N, i)
            sign_jac = sign(inverse_jacobian[nnodes(dg), i, element])
            normal = sign_jac *
                     get_normal_direction(2, contravariant_vectors, nnodes(dg), i,
                                          element)
            for v in eachvariable(equations_parabolic)
                fvisc = SVector(flux_viscous_x[v, nnodes(dg), i, element],
                                flux_viscous_y[v, nnodes(dg), i, element])
                interfaces_u[v, i, 2, element] = dot(fvisc, normal)
            end

            # Direction 3: negative y-face, volume node at (i, 1)
            sign_jac = sign(inverse_jacobian[i, 1, element])
            normal = sign_jac *
                     get_normal_direction(3, contravariant_vectors, i, 1, element)
            for v in eachvariable(equations_parabolic)
                fvisc = SVector(flux_viscous_x[v, i, 1, element],
                                flux_viscous_y[v, i, 1, element])
                interfaces_u[v, i, 3, element] = dot(fvisc, normal)
            end

            # Direction 4: positive y-face, volume node at (i, N)
            sign_jac = sign(inverse_jacobian[i, nnodes(dg), element])
            normal = sign_jac *
                     get_normal_direction(4, contravariant_vectors, i, nnodes(dg),
                                          element)
            for v in eachvariable(equations_parabolic)
                fvisc = SVector(flux_viscous_x[v, i, nnodes(dg), element],
                                flux_viscous_y[v, i, nnodes(dg), element])
                interfaces_u[v, i, 4, element] = dot(fvisc, normal)
            end
        end
    end

    return nothing
end

# Outer loop: compute divergence interface fluxes for all interior interfaces
function calc_interface_flux!(surface_flux_values,
                              mesh::StructuredMesh{2},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, parabolic_scheme, cache)
    @unpack elements = cache

    @threaded for element in eachelement(dg, cache)
        # Interfaces in x-direction (orientation = 1)
        calc_interface_flux!(surface_flux_values,
                             elements.left_neighbors[1, element],
                             element, 1, mesh,
                             equations_parabolic, dg, parabolic_scheme, cache)

        # Interfaces in y-direction (orientation = 2)
        calc_interface_flux!(surface_flux_values,
                             elements.left_neighbors[2, element],
                             element, 2, mesh,
                             equations_parabolic, dg, parabolic_scheme, cache)
    end

    return nothing
end

# Inner kernel: compute divergence flux at a single interior interface.
# `interfaces_u` stores the outward-oriented dot(F_visc, n̂):
#   - at the right face of left_element (right_direction): stored as  F·n̂
#   - at the left face of right_element (left_direction): stored as  F·(-n̂) = -F·n̂
# so flux_visc_rr = -interfaces_u[..., left_direction, right_element] gives F_rr·n̂.
@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation,
                                      mesh::StructuredMesh{2},
                                      equations_parabolic::AbstractEquationsParabolic,
                                      dg::DG, parabolic_scheme, cache)
    if left_element <= 0
        return nothing
    end

    @unpack interfaces_u, contravariant_vectors, inverse_jacobian = cache.elements

    right_direction = 2 * orientation
    left_direction = right_direction - 1

    for i in eachnode(dg)
        # Retrieve the pre-computed normal flux components.
        # right_direction face of left_element: flux stored as  +F_ll·n̂
        # left_direction face of right_element: flux stored as  -F_rr·n̂ → negate to get +F_rr·n̂
        viscous_flux_normal_ll = get_node_vars(interfaces_u, equations_parabolic, dg, i,
                                               right_direction, left_element)
        viscous_flux_normal_rr = -get_node_vars(interfaces_u, equations_parabolic, dg,
                                                i,
                                                left_direction, right_element)

        # Compute the outward normal at the interface (right element's left face)
        if orientation == 1
            sign_jac = sign(inverse_jacobian[1, i, right_element])
            normal_direction = sign_jac *
                               get_contravariant_vector(1, contravariant_vectors,
                                                        1, i, right_element)
        else # orientation == 2
            sign_jac = sign(inverse_jacobian[i, 1, right_element])
            normal_direction = sign_jac *
                               get_contravariant_vector(2, contravariant_vectors,
                                                        i, 1, right_element)
        end

        flux_ = flux_parabolic(viscous_flux_normal_ll, viscous_flux_normal_rr,
                               normal_direction, Divergence(),
                               equations_parabolic, parabolic_scheme)

        # Store the same flux value at both sides.
        # `calc_surface_integral!` for StructuredMesh subtracts at odd directions
        # (1, 3) and adds at even directions (2, 4), giving the correct net sign:
        #   left_element  (right_direction, even): +flux_/J
        #   right_element (left_direction, odd):   -flux_/J
        for v in eachvariable(equations_parabolic)
            surface_flux_values[v, i, right_direction, left_element] = flux_[v]
            surface_flux_values[v, i, left_direction, right_element] = flux_[v]
        end
    end

    return nothing
end

# Do nothing for periodic meshes
function calc_boundary_flux_divergence!(cache, t,
                                        boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::StructuredMesh{2},
                                        equations_parabolic, surface_integral, dg::DG)
    @assert isperiodic(mesh)
    return nothing
end

function calc_boundary_flux_divergence!(cache, t,
                                        boundary_conditions_parabolic::NamedTuple,
                                        mesh::StructuredMesh{2},
                                        equations_parabolic, surface_integral, dg::DG)
    @unpack surface_flux_values, node_coordinates, contravariant_vectors,
    inverse_jacobian, interfaces_u = cache.elements
    linear_indices = LinearIndices(size(mesh))

    for cell_y in axes(mesh, 2)
        # Negative x-direction (direction 1)
        direction = 1
        element = linear_indices[begin, cell_y]
        for j in eachnode(dg)
            calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                        boundary_conditions_parabolic[direction],
                                                        1, # orientation
                                                        direction, j,
                                                        mesh, equations_parabolic,
                                                        surface_integral, dg,
                                                        interfaces_u, node_coordinates,
                                                        contravariant_vectors,
                                                        inverse_jacobian, element)
        end

        # Positive x-direction (direction 2)
        direction = 2
        element = linear_indices[end, cell_y]
        for j in eachnode(dg)
            calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                        boundary_conditions_parabolic[direction],
                                                        1, # orientation
                                                        direction, j,
                                                        mesh, equations_parabolic,
                                                        surface_integral, dg,
                                                        interfaces_u, node_coordinates,
                                                        contravariant_vectors,
                                                        inverse_jacobian, element)
        end
    end

    for cell_x in axes(mesh, 1)
        # Negative y-direction (direction 3)
        direction = 3
        element = linear_indices[cell_x, begin]
        for i in eachnode(dg)
            calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                        boundary_conditions_parabolic[direction],
                                                        2, # orientation
                                                        direction, i,
                                                        mesh, equations_parabolic,
                                                        surface_integral, dg,
                                                        interfaces_u, node_coordinates,
                                                        contravariant_vectors,
                                                        inverse_jacobian, element)
        end

        # Positive y-direction (direction 4)
        direction = 4
        element = linear_indices[cell_x, end]
        for i in eachnode(dg)
            calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                        boundary_conditions_parabolic[direction],
                                                        2, # orientation
                                                        direction, i,
                                                        mesh, equations_parabolic,
                                                        surface_integral, dg,
                                                        interfaces_u, node_coordinates,
                                                        contravariant_vectors,
                                                        inverse_jacobian, element)
        end
    end

    return nothing
end

# Compute the divergence boundary flux at a single boundary face node.
# The sign convention must be consistent with `calc_surface_integral!` for StructuredMesh:
#   - odd directions (1, 3): surface integral subtracts the stored value → store -flux_
#   - even directions (2, 4): surface integral adds the stored value → store +flux_
# Combined with `apply_jacobian_parabolic!` (multiplies by +1/J), the net boundary
# contribution is +flux_/J at all faces (correct for the parabolic divergence).
@inline function calc_boundary_flux_by_direction_divergence!(surface_flux_values, t,
                                                             boundary_condition,
                                                             orientation, direction,
                                                             face_node_index,
                                                             mesh::StructuredMesh{2},
                                                             equations_parabolic,
                                                             surface_integral, dg::DG,
                                                             interfaces_u,
                                                             node_coordinates,
                                                             contravariant_vectors,
                                                             inverse_jacobian, element)
    # Map face node index and direction to volume node indices
    if orientation == 1 # x-faces
        if direction == 1 # -x face
            node_i, node_j = 1, face_node_index
        else              # +x face
            node_i, node_j = nnodes(dg), face_node_index
        end
    else # y-faces
        if direction == 3 # -y face
            node_i, node_j = face_node_index, 1
        else              # +y face
            node_i, node_j = face_node_index, nnodes(dg)
        end
    end

    # interfaces_u stores the pre-computed dot(F_visc, outward_normal) for each face
    flux_inner = get_node_vars(interfaces_u, equations_parabolic, dg, face_node_index,
                               direction, element)
    x = get_node_coords(node_coordinates, equations_parabolic, dg, node_i, node_j,
                        element)

    sign_jac = sign(inverse_jacobian[node_i, node_j, element])
    outward_normal = sign_jac *
                     get_normal_direction(direction, contravariant_vectors, node_i,
                                          node_j,
                                          element)

    flux_ = boundary_condition(flux_inner, nothing, outward_normal, x, t, Divergence(),
                               equations_parabolic)

    # Apply the sign correction to align with the `calc_surface_integral!` convention:
    # subtract at odd directions (1,3), add at even directions (2,4).
    sign_dir = isodd(direction) ? -one(eltype(flux_)) : one(eltype(flux_))
    for v in eachvariable(equations_parabolic)
        surface_flux_values[v, face_node_index, direction, element] = sign_dir *
                                                                      flux_[v]
    end

    return nothing
end
end # @muladd
