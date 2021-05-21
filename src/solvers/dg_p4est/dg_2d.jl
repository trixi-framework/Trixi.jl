function rhs!(du, u, t,
              mesh::P4estMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    equations, dg, cache)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(
    du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function prolong2interfaces!(cache, u,
                             mesh::P4estMesh{2},
                             equations, dg::DG)
  @unpack interfaces = cache

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for interface in eachinterface(dg, cache)
    primary_element   = interfaces.element_ids[1, interface]
    secondary_element = interfaces.element_ids[2, interface]

    primary_indices   = interfaces.node_indices[1, interface]
    secondary_indices = interfaces.node_indices[2, interface]

    # Use Tuple `node_indices` and `evaluate_index` to copy values
    # from the correct face and in the correct orientation
    for i in eachnode(dg), v in eachvariable(equations)
      interfaces.u[1, v, i, interface] = u[v, evaluate_index(primary_indices, size_, 1, i),
                                              evaluate_index(primary_indices, size_, 2, i),
                                              primary_element]

      interfaces.u[2, v, i, interface] = u[v, evaluate_index(secondary_indices, size_, 1, i),
                                              evaluate_index(secondary_indices, size_, 2, i),
                                              secondary_element]
    end
  end

  return nothing
end


function calc_interface_flux!(surface_flux_values,
                              mesh::P4estMesh{2},
                              equations, dg::DG, cache)
  @unpack surface_flux = dg
  @unpack u, element_ids, node_indices = cache.interfaces

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    primary_element   = element_ids[1, interface]
    secondary_element = element_ids[2, interface]

    primary_indices   = node_indices[1, interface]
    secondary_indices = node_indices[2, interface]

    primary_direction   = indices2direction(primary_indices)
    secondary_direction = indices2direction(secondary_indices)

    # Use Tuple `node_indices` and `evaluate_index` to access node indices
    # at the correct face and in the correct orientation to get normal vectors
    for i in eachnode(dg)
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)

      normal_vector = get_normal_vector(primary_direction, cache,
                                        evaluate_index(primary_indices, size_, 1, i),
                                        evaluate_index(primary_indices, size_, 2, i),
                                        primary_element)

      flux_ = surface_flux(u_ll, u_rr, normal_vector, equations)

      # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
      # to left and right element storage in the correct orientation
      for v in eachvariable(equations)
        surface_index = evaluate_index_surface(primary_indices, size_, 1, i)
        surface_flux_values[v, surface_index, primary_direction, primary_element] = flux_[v]

        surface_index = evaluate_index_surface(secondary_indices, size_, 1, i)
        surface_flux_values[v, surface_index, secondary_direction, secondary_element] = -flux_[v]
      end
    end
  end

  return nothing
end


function calc_surface_integral!(du, mesh::P4estMesh,
                                equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  # Note that all fluxes have been computed with outward-pointing normal vectors
  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, element] += (surface_flux_values[v, l, 1, element]
                                          * boundary_interpolation[1, 1])
        # surface at +x
        du[v, nnodes(dg), l, element] += (surface_flux_values[v, l, 2, element]
                                          * boundary_interpolation[nnodes(dg), 2])
        # surface at -y
        du[v, l, 1,          element] += (surface_flux_values[v, l, 3, element]
                                          * boundary_interpolation[1, 1])
        # surface at +y
        du[v, l, nnodes(dg), element] += (surface_flux_values[v, l, 4, element]
                                          * boundary_interpolation[nnodes(dg), 2])
      end
    end
  end

  return nothing
end
