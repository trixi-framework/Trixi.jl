function rhs!(du, u, t,
              mesh::CurvedMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" du .= zero(eltype(du))

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


function calc_volume_integral!(du, u,
                               mesh::Union{CurvedMesh{2}, UnstructuredQuadMesh, P4estMesh{2}},
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  @unpack contravariant_vectors = cache.elements

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      flux1 = flux(u_node, 1, equations)
      flux2 = flux(u_node, 2, equations)

      # Compute the contravariant flux by taking the scalar product of the
      # first contravariant vector Ja^1 and the flux vector
      Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2

      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * contravariant_flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # second contravariant vector Ja^2 and the flux vector
      Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2

      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * contravariant_flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


function calc_interface_flux!(cache, u,
                              mesh::CurvedMesh{2},
                              nonconservative_terms, # can be Val{true}/Val{false}
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
                                      mesh::CurvedMesh{2},
                                      nonconservative_terms::Val{false}, equations,
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
                                      mesh::CurvedMesh{2},
                                      nonconservative_terms::Val{true}, equations,
                                      surface_integral, dg::DG, cache)
  # See comment on `calc_interface_flux!` with `nonconservative_terms::Val{false}`
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

    # Call pointwise nonconservative term; Done twice because left/right orientation matters
    # See Bohm et al. 2018 for details on the nonconservative diamond "flux"
    # Scale with sign_jacobian to ensure that the normal_direction matches that from the flux above
    noncons_primary   = sign_jacobian * noncons_interface_flux(u_ll, u_rr, normal_direction, :weak, equations)
    noncons_secondary = sign_jacobian * noncons_interface_flux(u_rr, u_ll, normal_direction, :weak, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, right_direction, left_element] = flux[v] + noncons_primary[v]
      surface_flux_values[v, i, left_direction, right_element] = flux[v] + noncons_secondary[v]
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::CurvedMesh{2}, equations, surface_integral, dg::DG)
  @assert isperiodic(mesh)
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             mesh::CurvedMesh{2}, equations, surface_integral, dg::DG)
  calc_boundary_flux!(cache, u, t,
                      (boundary_condition, boundary_condition,
                       boundary_condition, boundary_condition),
                      mesh, equations, surface_integral, dg)
end


function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             mesh::CurvedMesh{2}, equations, surface_integral, dg::DG)
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
                         mesh::Union{CurvedMesh{2}, UnstructuredQuadMesh, P4estMesh{2}},
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
