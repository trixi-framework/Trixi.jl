# !!! warning "Experimental implementation (curvilinear FDSBP)"
#     This is an experimental feature and may change in future releases.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# Alternative constructor to reuse the element container from `UnstructuredMesh2D`
# OBS! New element constructors require the derivative operator to compute the metric terms.
function init_elements!(elements::UnstructuredElementContainer2D, mesh, basis::AbstractDerivativeOperator)
  four_corners = zeros(eltype(mesh.corners), 4, 2)

  # loop through elements and call the correct constructor based on whether the element is curved
  for element in eachelement(elements)
    if mesh.element_is_curved[element]
      init_element!(elements, element, basis, view(mesh.surface_curves, :, element))
    else # straight sided element
      for i in 1:4, j in 1:2
        # pull the (x,y) values of these corners out of the global corners array
        four_corners[i, j] = mesh.corners[j, mesh.element_node_ids[i, element]]
      end
      init_element!(elements, element, basis, four_corners)
    end
  end
end


# initialize all the values in the container of a general FD block (either straight sided or curved)
# OBS! Requires the SBP derivative matrix in order to compute metric terms that are free-stream preserving
function init_element!(elements, element, basis::AbstractDerivativeOperator, corners_or_surface_curves)

  calc_node_coordinates!(elements.node_coordinates, element, get_nodes(basis), corners_or_surface_curves)

  calc_metric_terms!(elements.jacobian_matrix, element, basis, elements.node_coordinates)

  calc_inverse_jacobian!(elements.inverse_jacobian, element, elements.jacobian_matrix)

  calc_contravariant_vectors!(elements.contravariant_vectors, element, elements.jacobian_matrix)

  calc_normal_directions!(elements.normal_directions, element, elements.jacobian_matrix)

  return elements
end


# construct the metric terms for a FDSBP element "block". Directly use the derivative matrix
# applied to the node coordinates.
# TODO: FD performance; this is slow and allocates. Rewrite to use `mul!`
# TODO: FD; How to make this work for the upwind solver because basis has three available derivative matrices
function calc_metric_terms!(jacobian_matrix, element, basis::AbstractDerivativeOperator, node_coordinates)

  # storage format:
  #   jacobian_matrix[1,1,:,:,:] <- X_xi
  #   jacobian_matrix[1,2,:,:,:] <- X_eta
  #   jacobian_matrix[2,1,:,:,:] <- Y_xi
  #   jacobian_matrix[2,2,:,:,:] <- Y_eta

  # Compute the xi derivatives by applying D on the left
  jacobian_matrix[1, 1, :, :, element] = Matrix(basis) * node_coordinates[1, :, :, element]
  jacobian_matrix[2, 1, :, :, element] = Matrix(basis) * node_coordinates[2, :, :, element]

  # Compute the eta derivatives by applying tranpose of D on the right
  jacobian_matrix[1, 2, :, :, element] = node_coordinates[1, :, :, element] * Matrix(basis)'
  jacobian_matrix[2, 2, :, :, element] = node_coordinates[2, :, :, element] * Matrix(basis)'

  return jacobian_matrix
end

# construct the normal direction vectors (but not actually normalized) for a curved sided FDSBP element "block"
# normalization occurs on the fly during the surface flux computation
# OBS! This assumes that the boundary points are included.
function calc_normal_directions!(normal_directions, element, jacobian_matrix)

  # normal directions on the boundary for the left (local side 4) and right (local side 2)
  N_ = size(jacobian_matrix, 3)
  for j in 1:N_
    # +x side or side 2 in the local indexing
    X_xi  = jacobian_matrix[1, 1, N_, j, element]
    X_eta = jacobian_matrix[1, 2, N_, j, element]
    Y_xi  = jacobian_matrix[2, 1, N_, j, element]
    Y_eta = jacobian_matrix[2, 2, N_, j, element]
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    normal_directions[1, j, 2, element] = sign(Jtemp) * ( Y_eta)
    normal_directions[2, j, 2, element] = sign(Jtemp) * (-X_eta)

    # -x side or side 4 in the local indexing
    X_xi  = jacobian_matrix[1, 1, 1, j, element]
    X_eta = jacobian_matrix[1, 2, 1, j, element]
    Y_xi  = jacobian_matrix[2, 1, 1, j, element]
    Y_eta = jacobian_matrix[2, 2, 1, j, element]
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    normal_directions[1, j, 4, element] = -sign(Jtemp) * ( Y_eta)
    normal_directions[2, j, 4, element] = -sign(Jtemp) * (-X_eta)
  end

  # normal directions on the boundary for the top (local side 3) and bottom (local side 1)
  N_ = size(jacobian_matrix, 4)
  for i in 1:N_
    # -y side or side 1 in the local indexing
    X_xi  = jacobian_matrix[1, 1, i, 1, element]
    X_eta = jacobian_matrix[1, 2, i, 1, element]
    Y_xi  = jacobian_matrix[2, 1, i, 1, element]
    Y_eta = jacobian_matrix[2, 2, i, 1, element]
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    normal_directions[1, i, 1, element] = -sign(Jtemp) * (-Y_xi)
    normal_directions[2, i, 1, element] = -sign(Jtemp) * ( X_xi)

    # +y side or side 3 in the local indexing
    X_xi  = jacobian_matrix[1, 1, i, N_, element]
    X_eta = jacobian_matrix[1, 2, i, N_, element]
    Y_xi  = jacobian_matrix[2, 1, i, N_, element]
    Y_eta = jacobian_matrix[2, 2, i, N_, element]
    Jtemp = X_xi * Y_eta - X_eta * Y_xi
    normal_directions[1, i, 3, element] = sign(Jtemp) * (-Y_xi)
    normal_directions[2, i, 3, element] = sign(Jtemp) * ( X_xi)
  end

  return normal_directions
end


end # @muladd