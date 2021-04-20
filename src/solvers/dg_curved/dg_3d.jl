function rhs!(du, u, t,
              mesh::CurvedMesh{3}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @_timeit timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Calculate interface fluxes
  @_timeit timer() "interface flux" calc_interface_flux!(
    cache, u, mesh, equations, dg)

  # Calculate boundary fluxes
  @_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, u, t, boundary_conditions, mesh, equations, dg)

  # Calculate surface integrals
  @_timeit timer() "surface integral" calc_surface_integral!(
    du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::CurvedMesh{3},
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  @unpack contravariant_vectors = cache.elements

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)

      flux1 = flux(u_node, 1, equations)
      flux2 = flux(u_node, 2, equations)
      flux3 = flux(u_node, 3, equations)

      # Compute the contravariant flux by taking the scalar product of the
      # first contravariant vector Ja^1 and the flux vector
      Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
      contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2 + Ja13 * flux3

      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * contravariant_flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, k, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # second contravariant vector Ja^2 and the flux vector
      Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
      contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2 + Ja23 * flux3

      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * contravariant_flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, k, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # third contravariant vector Ja^3 and the flux vector
      Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
      contravariant_flux3 = Ja31 * flux1 + Ja32 * flux2 + Ja33 * flux3

      for kk in eachnode(dg)
        integral_contribution = derivative_dhat[kk, k] * contravariant_flux3
        add_to_node_vars!(du, integral_contribution, equations, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


function calc_interface_flux!(cache, u, mesh::CurvedMesh{3},
                              equations, dg::DG)
  @unpack elements = cache
  @unpack surface_flux = dg

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2, 3)"

    # Interfaces in x-direction (`orientation` = 1)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, mesh, equations, dg, cache)

    # Interfaces in y-direction (`orientation` = 2)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, mesh, equations, dg, cache)

    # Interfaces in z-direction (`orientation` = 3)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[3, element],
                         element, 3, u, mesh, equations, dg, cache)
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::CurvedMesh{3}, equations,
                                      dg::DG, cache)
  # This is slow for LSA, but for some reason faster for Euler (see #519)
  if left_element <= 0 # left_element = 0 at boundaries
    return surface_flux_values
  end

  @unpack surface_flux = dg
  @unpack contravariant_vectors = cache.elements

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for j in eachnode(dg), i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, j, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, j, right_element)

      # First contravariant vector Ja^1 as SVector
      normal = get_contravariant_vector(1, contravariant_vectors, 1, i, j, right_element)
    elseif orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), j, left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          j, right_element)

      # Second contravariant vector Ja^2 as SVector
      normal = get_contravariant_vector(2, contravariant_vectors, i, 1, j, right_element)
    else # orientation == 3
      u_ll = get_node_vars(u, equations, dg, i, j, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, j, 1,          right_element)

      # Third contravariant vector Ja^3 as SVector
      normal = get_contravariant_vector(3, contravariant_vectors, i, j, 1, right_element)
    end

    flux = surface_flux(u_ll, u_rr, normal, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, j, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, j, left_direction, right_element] = flux[v]
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::CurvedMesh{3}, equations, dg::DG)
  @assert isperiodic(mesh)
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             mesh::CurvedMesh{3}, equations, dg::DG)
  calc_boundary_flux!(cache, u, t,
                      (boundary_condition, boundary_condition, boundary_condition,
                       boundary_condition, boundary_condition, boundary_condition),
                      mesh, equations, dg)
end


function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             mesh::CurvedMesh{3}, equations, dg::DG)
  @unpack surface_flux = dg
  @unpack surface_flux_values = cache.elements
  linear_indices = LinearIndices(size(mesh))

  for cell_z in axes(mesh, 3), cell_y in axes(mesh, 2)
    # Negative x-direction
    direction = 1
    element = linear_indices[begin, cell_y, cell_z]

    for k in eachnode(dg), j in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations,  dg, cache,
                                       direction, (1, j, k), (j, k), element)
    end

    # Positive x-direction
    direction = 2
    element = linear_indices[end, cell_y, cell_z]

    for k in eachnode(dg), j in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (nnodes(dg), j, k), (j, k), element)
    end
  end

  for cell_z in axes(mesh, 3), cell_x in axes(mesh, 1)
    # Negative y-direction
    direction = 3
    element = linear_indices[cell_x, begin, cell_z]

    for k in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (i, 1, k), (i, k), element)
    end

    # Positive y-direction
    direction = 4
    element = linear_indices[cell_x, end, cell_z]

    for k in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (i, nnodes(dg), k), (i, k), element)
    end
  end

  for cell_y in axes(mesh, 2), cell_x in axes(mesh, 1)
    # Negative z-direction
    direction = 5
    element = linear_indices[cell_x, cell_y, begin]

    for j in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 3,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (i, j, 1), (i, j), element)
    end

    # Positive z-direction
    direction = 6
    element = linear_indices[cell_x, cell_y, end]

    for j in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 3,
                                       boundary_conditions[direction],
                                       mesh, equations, dg, cache,
                                       direction, (i, j, nnodes(dg)), (i, j), element)
    end
  end
end


function apply_jacobian!(du,
                         mesh::CurvedMesh{3},
                         equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      factor = -cache.elements.inverse_jacobian[i, j, k, element]

      for v in eachvariable(equations)
        du[v, i, j, k, element] *= factor
      end
    end
  end

  return nothing
end
