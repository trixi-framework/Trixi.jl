# !!! warning "Experimental implementation (curvilinear FDSBP)"
#     This is an experimental feature and may change in future releases.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# initialize all the values in the container of a general FD block (either straight sided or curved)
# OBS! Requires the SBP derivative matrix in order to compute metric terms that are free-stream preserving
function init_element!(elements, element, basis::AbstractDerivativeOperator,
                       corners_or_surface_curves)
    calc_node_coordinates!(elements.node_coordinates, element, get_nodes(basis),
                           corners_or_surface_curves)

    calc_metric_terms!(elements.jacobian_matrix, element, basis,
                       elements.node_coordinates)

    calc_inverse_jacobian!(elements.inverse_jacobian, element, elements.jacobian_matrix)

    calc_contravariant_vectors!(elements.contravariant_vectors, element,
                                elements.jacobian_matrix)

    calc_normal_directions!(elements.normal_directions, element,
                            elements.jacobian_matrix)

    return elements
end

# initialize all the values in the container of a general FD block (either straight sided or curved)
# for a set of upwind finite difference operators
# OBS! (Maybe) Requires the biased derivative matrices in order to compute proper metric terms
# that are free-stream preserving. If this is not necessary, then we do not need this specialized container.
function init_element!(elements, element, basis::SummationByPartsOperators.UpwindOperators, corners_or_surface_curves)

    calc_node_coordinates!(elements.node_coordinates, element, get_nodes(basis), corners_or_surface_curves)

    # Create the metric terms and contravariant vectors with the D⁺ operator
    calc_metric_terms!(elements.jacobian_matrix, element, basis.plus, elements.node_coordinates)
    calc_contravariant_vectors!(elements.contravariant_vectors_plus, element, elements.jacobian_matrix)
    calc_normal_directions!(elements.normal_directions_plus, element, elements.jacobian_matrix)

    # Create the metric terms and contravariant vectors with the D⁻ operator
    calc_metric_terms!(elements.jacobian_matrix, element, basis.minus, elements.node_coordinates)
    calc_contravariant_vectors!(elements.contravariant_vectors_minus, element, elements.jacobian_matrix)
    calc_normal_directions!(elements.normal_directions_minus, element, elements.jacobian_matrix)

    # Create the metric terms, Jacobian information, and normals for analysis
    # and the SATs with the central D operator.
    calc_metric_terms!(elements.jacobian_matrix, element, basis.central, elements.node_coordinates)
    calc_inverse_jacobian!(elements.inverse_jacobian, element, elements.jacobian_matrix)
    calc_contravariant_vectors!(elements.contravariant_vectors, element, elements.jacobian_matrix)
    calc_normal_directions!(elements.normal_directions, element, elements.jacobian_matrix)

    return elements
  end

# construct the metric terms for a FDSBP element "block". Directly use the derivative matrix
# applied to the node coordinates.
# TODO: FD; How to make this work for the upwind solver because basis has three available derivative matrices
function calc_metric_terms!(jacobian_matrix, element, D_SBP::AbstractDerivativeOperator,
                            node_coordinates)

    # storage format:
    #   jacobian_matrix[1,1,:,:,:] <- X_xi
    #   jacobian_matrix[1,2,:,:,:] <- X_eta
    #   jacobian_matrix[2,1,:,:,:] <- Y_xi
    #   jacobian_matrix[2,2,:,:,:] <- Y_eta

    # TODO: for now the upwind version needs to specify which matrix is used.
    # requires more testing / debugging but for now use the central SBP matrix

    # Compute the xi derivatives by applying D on the left
    # This is basically the same as
    # jacobian_matrix[1, 1, :, :, element] = Matrix(D_SBP) * node_coordinates[1, :, :, element]
    # but uses only matrix-vector products instead of a matrix-matrix product.
    for j in eachnode(D_SBP)
        mul!(view(jacobian_matrix, 1, 1, :, j, element), D_SBP,
             view(node_coordinates, 1, :, j, element))
    end
    # jacobian_matrix[2, 1, :, :, element] = Matrix(D_SBP) * node_coordinates[2, :, :, element]
    for j in eachnode(D_SBP)
        mul!(view(jacobian_matrix, 2, 1, :, j, element), D_SBP,
             view(node_coordinates, 2, :, j, element))
    end

    # Compute the eta derivatives by applying transpose of D on the right
    # jacobian_matrix[1, 2, :, :, element] = node_coordinates[1, :, :, element] * Matrix(D_SBP)'
    for i in eachnode(D_SBP)
        mul!(view(jacobian_matrix, 1, 2, i, :, element), D_SBP,
             view(node_coordinates, 1, i, :, element))
    end
    # jacobian_matrix[2, 2, :, :, element] = node_coordinates[2, :, :, element] * Matrix(D_SBP)'
    for i in eachnode(D_SBP)
        mul!(view(jacobian_matrix, 2, 2, i, :, element), D_SBP,
             view(node_coordinates, 2, i, :, element))
    end

    return jacobian_matrix
end

# construct the normal direction vectors (but not actually normalized) for a curved sided FDSBP element "block"
# normalization occurs on the fly during the surface flux computation
# OBS! This assumes that the boundary points are included.
function calc_normal_directions!(normal_directions, element, jacobian_matrix)

    # normal directions on the boundary for the left (local side 4) and right (local side 2)
    N = size(jacobian_matrix, 4)
    for j in 1:N
        # +x side or side 2 in the local indexing
        X_xi = jacobian_matrix[1, 1, N, j, element]
        X_eta = jacobian_matrix[1, 2, N, j, element]
        Y_xi = jacobian_matrix[2, 1, N, j, element]
        Y_eta = jacobian_matrix[2, 2, N, j, element]
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, j, 2, element] = sign(Jtemp) * (Y_eta)
        normal_directions[2, j, 2, element] = sign(Jtemp) * (-X_eta)

        # -x side or side 4 in the local indexing
        X_xi = jacobian_matrix[1, 1, 1, j, element]
        X_eta = jacobian_matrix[1, 2, 1, j, element]
        Y_xi = jacobian_matrix[2, 1, 1, j, element]
        Y_eta = jacobian_matrix[2, 2, 1, j, element]
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, j, 4, element] = -sign(Jtemp) * (Y_eta)
        normal_directions[2, j, 4, element] = -sign(Jtemp) * (-X_eta)
    end

    # normal directions on the boundary for the top (local side 3) and bottom (local side 1)
    N = size(jacobian_matrix, 3)
    for i in 1:N
        # -y side or side 1 in the local indexing
        X_xi = jacobian_matrix[1, 1, i, 1, element]
        X_eta = jacobian_matrix[1, 2, i, 1, element]
        Y_xi = jacobian_matrix[2, 1, i, 1, element]
        Y_eta = jacobian_matrix[2, 2, i, 1, element]
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, i, 1, element] = -sign(Jtemp) * (-Y_xi)
        normal_directions[2, i, 1, element] = -sign(Jtemp) * (X_xi)

        # +y side or side 3 in the local indexing
        X_xi = jacobian_matrix[1, 1, i, N, element]
        X_eta = jacobian_matrix[1, 2, i, N, element]
        Y_xi = jacobian_matrix[2, 1, i, N, element]
        Y_eta = jacobian_matrix[2, 2, i, N, element]
        Jtemp = X_xi * Y_eta - X_eta * Y_xi
        normal_directions[1, i, 3, element] = sign(Jtemp) * (-Y_xi)
        normal_directions[2, i, 3, element] = sign(Jtemp) * (X_xi)
    end

    return normal_directions
end

# Compute the rotation, if any, for the local coordinate system on each `element`
# with respect to the standard x-axis. This is necessary because if the local
# element axis is rotated it influences the computation of the +/- directions
# for the flux vector splitting of the upwind scheme.
# Local principle x-axis is computed from the four corners of a particular `element`,
# see page 35 of the "Verdict Library" for details.
# From the local axis the `atan` function is used to determine the value of the local rotation.
#
# - C. J. Stimpson, C. D. Ernst, P. Knupp, P. P. Pébay, and D. Thompson (2007)
#   The Verdict Geometric Quality Library
#   Technical Report. Sandia National Laboraties
#   [SAND2007-1751](https://coreform.com/papers/verdict_quality_library.pdf)
# - `atan(y, x)` function (https://docs.julialang.org/en/v1/base/math/#Base.atan-Tuple%7BNumber%7D)
# TODO: remove me. unnecessary
function calc_element_rotations!(elements, mesh::UnstructuredMesh2D)

    for element in 1:size(elements.node_coordinates, 4)
        # Pull the corners for the current element
        corner_points = mesh.corners[:, mesh.element_node_ids[:, element]]

        # Compute the principle x-axis of a right-handed quadrilateral element
        local_x_axis = (  (corner_points[:, 2] - corner_points[:, 1])
                        + (corner_points[:, 3] - corner_points[:, 4]) )

        # Two argument `atan` function retuns the angle in radians in [−pi, pi]
        # between the positive x-axis and the point (x, y)
        local_angle = atan(local_x_axis[2], local_x_axis[1])

        # Possible reference rotations of the local axis for a given quadrilateral element in the mesh
        #                   0°, 90°, 180°, -90° (or 270°), -180°
        reference_angles = [0.0, 0.5*pi, pi, -0.5*pi, -pi]

        elements.rotations[element] = argmin( abs.(reference_angles .- local_angle) ) - 1
    end

    return nothing
end
end # @muladd
