# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


function rhs!(du, u, t,
              mesh::StructuredMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, u, t, boundary_conditions, mesh, equations, dg.surface_integral, dg)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


@inline function weak_form_kernel!(du, u,
                                   element, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                                   nonconservative_terms::False, equations,
                                   dg::DGSEM, cache, alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_dhat = dg.basis
  @unpack contravariant_vectors = cache.elements

  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    flux1 = flux(u_node, 1, equations)
    flux2 = flux(u_node, 2, equations)

    # Compute the contravariant flux by taking the scalar product of the
    # first contravariant vector Ja^1 and the flux vector
    Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
    for ii in eachnode(dg)
      multiply_add_to_node_vars!(du, alpha * derivative_dhat[ii, i], contravariant_flux1, equations, dg, ii, j, element)
    end

    # Compute the contravariant flux by taking the scalar product of the
    # second contravariant vector Ja^2 and the flux vector
    Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
    for jj in eachnode(dg)
      multiply_add_to_node_vars!(du, alpha * derivative_dhat[jj, j], contravariant_flux2, equations, dg, i, jj, element)
    end
  end

  return nothing
end


@inline function flux_differencing_kernel!(du, u,
                                           element, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                                           nonconservative_terms::False, equations,
                                           volume_flux, dg::DGSEM, cache, alpha=true)
  @unpack derivative_split = dg.basis
  @unpack contravariant_vectors = cache.elements

  # Calculate volume integral in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-point flux
    # computations.

    # x direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      # pull the contravariant vectors and compute the average
      Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors, ii, j, element)
      Ja1_avg = 0.5 * (Ja1_node + Ja1_node_ii)
      # compute the contravariant sharp flux in the direction of the
      # averaged contravariant vector
      fluxtilde1 = volume_flux(u_node, u_node_ii, Ja1_avg, equations)
      multiply_add_to_node_vars!(du, alpha * derivative_split[i, ii], fluxtilde1, equations, dg, i,  j, element)
      multiply_add_to_node_vars!(du, alpha * derivative_split[ii, i], fluxtilde1, equations, dg, ii, j, element)
    end

    # y direction
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      # pull the contravariant vectors and compute the average
      Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors, i, jj, element)
      Ja2_avg = 0.5 * (Ja2_node + Ja2_node_jj)
      # compute the contravariant sharp flux in the direction of the
      # averaged contravariant vector
      fluxtilde2 = volume_flux(u_node, u_node_jj, Ja2_avg, equations)
      multiply_add_to_node_vars!(du, alpha * derivative_split[j, jj], fluxtilde2, equations, dg, i, j,  element)
      multiply_add_to_node_vars!(du, alpha * derivative_split[jj, j], fluxtilde2, equations, dg, i, jj, element)
    end
  end
end

@inline function flux_differencing_kernel!(du, u,
                                           element, mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                                           nonconservative_terms::True, equations,
                                           volume_flux, dg::DGSEM, cache, alpha=true)
  @unpack derivative_split = dg.basis
  @unpack contravariant_vectors = cache.elements
  symmetric_flux, nonconservative_flux = volume_flux

  # Apply the symmetric flux as usual
  flux_differencing_kernel!(du, u, element, mesh, False(), equations, symmetric_flux, dg, cache, alpha)

  # Calculate the remaining volume terms using the nonsymmetric generalized flux
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # pull the contravariant vectors in each coordinate direction
    Ja1_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja2_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)

    # The diagonal terms are zero since the diagonal of `derivative_split`
    # is zero. We ignore this for now.
    # In general, nonconservative fluxes can depend on both the contravariant
    # vectors (normal direction) at the current node and the averaged ones.
    # Thus, we need to pass both to the nonconservative flux.

    # x direction
    integral_contribution = zero(u_node)
    for ii in eachnode(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      # pull the contravariant vectors and compute the average
      Ja1_node_ii = get_contravariant_vector(1, contravariant_vectors, ii, j, element)
      Ja1_avg = 0.5 * (Ja1_node + Ja1_node_ii)
      # Compute the contravariant nonconservative flux.
      fluxtilde1 = nonconservative_flux(u_node, u_node_ii, Ja1_node, Ja1_avg, equations)
      integral_contribution = integral_contribution + derivative_split[i, ii] * fluxtilde1
    end

    # y direction
    for jj in eachnode(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      # pull the contravariant vectors and compute the average
      Ja2_node_jj = get_contravariant_vector(2, contravariant_vectors, i, jj, element)
      Ja2_avg = 0.5 * (Ja2_node + Ja2_node_jj)
      # compute the contravariant nonconservative flux in the direction of the
      # averaged contravariant vector
      fluxtilde2 = nonconservative_flux(u_node, u_node_jj, Ja2_node, Ja2_avg, equations)
      integral_contribution = integral_contribution + derivative_split[j, jj] * fluxtilde2
    end

    # The factor 0.5 cancels the factor 2 in the flux differencing form
    multiply_add_to_node_vars!(du, alpha * 0.5, integral_contribution, equations, dg, i, j, element)
  end
end


# Computing the normal vector for the FV method on curvilinear subcells.
# To fulfill free-stream preservation we use the explicit formula B.53 in Appendix B.4
# by Hennemann, Rueda-Ramirez, Hindenlang, Gassner (2020)
# "A provably entropy stable subcell shock capturing approach for high order split form DG for the compressible Euler equations"
# [arXiv: 2008.12044v2](https://arxiv.org/pdf/2008.12044)
@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u,
                              mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                              nonconservative_terms::False, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
  @unpack contravariant_vectors = cache.elements
  @unpack weights, derivative_matrix = dg.basis

  # Performance improvement if the metric terms of the subcell FV method are only computed
  # once at the beginning of the simulation, instead of at every Runge-Kutta stage
  fstar1_L[:, 1,            :] .= zero(eltype(fstar1_L))
  fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
  fstar1_R[:, 1,            :] .= zero(eltype(fstar1_R))
  fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

  for j in eachnode(dg)
    normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)

    for i in 2:nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i-1, j, element)
      u_rr = get_node_vars(u, equations, dg, i,   j, element)

      for m in 1:nnodes(dg)
        normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
      end

      # Compute the contravariant flux
      contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

      set_node_vars!(fstar1_L, contravariant_flux, equations, dg, i, j)
      set_node_vars!(fstar1_R, contravariant_flux, equations, dg, i, j)
    end
  end

  fstar2_L[:, :, 1           ] .= zero(eltype(fstar2_L))
  fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
  fstar2_R[:, :, 1           ] .= zero(eltype(fstar2_R))
  fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

  for i in eachnode(dg)
    normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)

    for j in 2:nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i, j-1, element)
      u_rr = get_node_vars(u, equations, dg, i, j,   element)

      for m in 1:nnodes(dg)
        normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # normal vector and the flux vector
      contravariant_flux = volume_flux_fv(u_ll, u_rr, normal_direction, equations)

      set_node_vars!(fstar2_L, contravariant_flux, equations, dg, i, j)
      set_node_vars!(fstar2_R, contravariant_flux, equations, dg, i, j)
    end
  end

  return nothing
end

# Calculate the finite volume fluxes inside curvilinear elements (**with non-conservative terms**).
@inline function calcflux_fv!(fstar1_L, fstar1_R, fstar2_L, fstar2_R, u::AbstractArray{<:Any,4},
                              mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                              nonconservative_terms::True, equations,
                              volume_flux_fv, dg::DGSEM, element, cache)
  @unpack contravariant_vectors = cache.elements
  @unpack weights, derivative_matrix = dg.basis

  volume_flux, nonconservative_flux = volume_flux_fv

  # Performance improvement if the metric terms of the subcell FV method are only computed
  # once at the beginning of the simulation, instead of at every Runge-Kutta stage
  fstar1_L[:, 1,            :] .= zero(eltype(fstar1_L))
  fstar1_L[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_L))
  fstar1_R[:, 1,            :] .= zero(eltype(fstar1_R))
  fstar1_R[:, nnodes(dg)+1, :] .= zero(eltype(fstar1_R))

  for j in eachnode(dg)
    normal_direction = get_contravariant_vector(1, contravariant_vectors, 1, j, element)
    for i in 2:nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i-1, j, element)
      u_rr = get_node_vars(u, equations, dg, i,   j, element)

      for m in eachnode(dg)
        normal_direction += weights[i-1] * derivative_matrix[i-1, m] * get_contravariant_vector(1, contravariant_vectors, m, j, element)
      end

      # Compute the conservative part of the contravariant flux
      ftilde1 = volume_flux(u_ll, u_rr, normal_direction, equations)

      # Compute and add in the nonconservative part
      # Note the factor 0.5 necessary for the nonconservative fluxes based on
      # the interpretation of global SBP operators coupled discontinuously via
      # central fluxes/SATs
      ftilde1_L = ftilde1 + 0.5 * nonconservative_flux(u_ll, u_rr, normal_direction, normal_direction, equations)
      ftilde1_R = ftilde1 + 0.5 * nonconservative_flux(u_rr, u_ll, normal_direction, normal_direction, equations)

      set_node_vars!(fstar1_L, ftilde1_L, equations, dg, i, j)
      set_node_vars!(fstar1_R, ftilde1_R, equations, dg, i, j)
    end
  end

  # Fluxes in y
  fstar2_L[:, :, 1           ] .= zero(eltype(fstar2_L))
  fstar2_L[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_L))
  fstar2_R[:, :, 1           ] .= zero(eltype(fstar2_R))
  fstar2_R[:, :, nnodes(dg)+1] .= zero(eltype(fstar2_R))

  # Compute inner fluxes
  for i in eachnode(dg)
    normal_direction = get_contravariant_vector(2, contravariant_vectors, i, 1, element)

    for j in 2:nnodes(dg)
      u_ll = get_node_vars(u, equations, dg, i, j-1, element)
      u_rr = get_node_vars(u, equations, dg, i, j,   element)

      for m in eachnode(dg)
        normal_direction += weights[j-1] * derivative_matrix[j-1, m] * get_contravariant_vector(2, contravariant_vectors, i, m, element)
      end

      # Compute the conservative part of the contravariant flux
      ftilde2 = volume_flux(u_ll, u_rr, normal_direction, equations)

      # Compute and add in the nonconservative part
      # Note the factor 0.5 necessary for the nonconservative fluxes based on
      # the interpretation of global SBP operators coupled discontinuously via
      # central fluxes/SATs
      ftilde2_L = ftilde2 + 0.5 * nonconservative_flux(u_ll, u_rr, normal_direction, normal_direction, equations)
      ftilde2_R = ftilde2 + 0.5 * nonconservative_flux(u_rr, u_ll, normal_direction, normal_direction, equations)

      set_node_vars!(fstar2_L, ftilde2_L, equations, dg, i, j)
      set_node_vars!(fstar2_R, ftilde2_R, equations, dg, i, j)
    end
  end

  return nothing
end


function calc_interface_flux!(cache, u,
                              mesh::StructuredMesh{2},
                              nonconservative_terms, # can be True/False
                              equations, surface_integral, dg::DG)
  @unpack elements = cache

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2)"

    # Interfaces in x-direction (`orientation` = 1)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, mesh,
                         nonconservative_terms, equations,
                         surface_integral, dg, cache)

    # Interfaces in y-direction (`orientation` = 2)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, mesh,
                         nonconservative_terms, equations,
                         surface_integral, dg, cache)
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::StructuredMesh{2},
                                      nonconservative_terms::False, equations,
                                      surface_integral, dg::DG, cache)
  # This is slow for LSA, but for some reason faster for Euler (see #519)
  if left_element <= 0 # left_element = 0 at boundaries
    return nothing
  end

  @unpack surface_flux = surface_integral
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

      # If the mapping is orientation-reversing, the contravariant vectors' orientation
      # is reversed as well. The normal vector must be oriented in the direction
      # from `left_element` to `right_element`, or the numerical flux will be computed
      # incorrectly (downwind direction).
      sign_jacobian = sign(inverse_jacobian[1, i, right_element])

      # First contravariant vector Ja^1 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(1, contravariant_vectors,
                                                                  1, i, right_element)
    else # orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

      # See above
      sign_jacobian = sign(inverse_jacobian[i, 1, right_element])

      # Second contravariant vector Ja^2 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(2, contravariant_vectors,
                                                                  i, 1, right_element)
    end

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, left_direction, right_element] = flux[v]
    end
  end

  return nothing
end

@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::StructuredMesh{2},
                                      nonconservative_terms::True, equations,
                                      surface_integral, dg::DG, cache)
  # See comment on `calc_interface_flux!` with `nonconservative_terms::False`
  if left_element <= 0 # left_element = 0 at boundaries
    return nothing
  end

  surface_flux, nonconservative_flux = surface_integral.surface_flux
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  right_direction = 2 * orientation
  left_direction  = right_direction - 1

  for i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

      # If the mapping is orientation-reversing, the contravariant vectors' orientation
      # is reversed as well. The normal vector must be oriented in the direction
      # from `left_element` to `right_element`, or the numerical flux will be computed
      # incorrectly (downwind direction).
      sign_jacobian = sign(inverse_jacobian[1, i, right_element])

      # First contravariant vector Ja^1 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(1, contravariant_vectors,
                                                                  1, i, right_element)
    else # orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

      # See above
      sign_jacobian = sign(inverse_jacobian[i, 1, right_element])

      # Second contravariant vector Ja^2 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(2, contravariant_vectors,
                                                                  i, 1, right_element)
    end

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)

    # Compute both nonconservative fluxes
    # In general, nonconservative fluxes can depend on both the contravariant
    # vectors (normal direction) at the current node and the averaged ones.
    # However, both are the same at watertight interfaces, so we pass the
    # `normal_direction` twice.
    # Scale with sign_jacobian to ensure that the normal_direction matches that
    # from the flux above
    noncons_left  = sign_jacobian * nonconservative_flux(u_ll, u_rr, normal_direction, normal_direction, equations)
    noncons_right = sign_jacobian * nonconservative_flux(u_rr, u_ll, normal_direction, normal_direction, equations)

    for v in eachvariable(equations)
      # Note the factor 0.5 necessary for the nonconservative fluxes based on
      # the interpretation of global SBP operators coupled discontinuously via
      # central fluxes/SATs
      surface_flux_values[v, i, right_direction, left_element] = flux[v] + 0.5 * noncons_left[v]
      surface_flux_values[v, i, left_direction, right_element] = flux[v] + 0.5 * noncons_right[v]
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::StructuredMesh{2}, equations, surface_integral, dg::DG)
  @assert isperiodic(mesh)
end

function calc_boundary_flux!(cache, u, t, boundary_conditions::NamedTuple,
                             mesh::StructuredMesh{2}, equations, surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  linear_indices = LinearIndices(size(mesh))

  for cell_y in axes(mesh, 2)
    # Negative x-direction
    direction = 1
    element = linear_indices[begin, cell_y]

    for j in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (1, j), (j,), element)
    end

    # Positive x-direction
    direction = 2
    element = linear_indices[end, cell_y]

    for j in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (nnodes(dg), j), (j,), element)
    end
  end

  for cell_x in axes(mesh, 1)
    # Negative y-direction
    direction = 3
    element = linear_indices[cell_x, begin]

    for i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (i, 1), (i,), element)
    end

    # Positive y-direction
    direction = 4
    element = linear_indices[cell_x, end]

    for i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (i, nnodes(dg)), (i,), element)
    end
  end
end


function apply_jacobian!(du,
                         mesh::Union{StructuredMesh{2}, UnstructuredMesh2D, P4estMesh{2}},
                         equations, dg::DG, cache)
  @unpack inverse_jacobian = cache.elements

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      factor = -inverse_jacobian[i, j, element]

      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end


end # @muladd
