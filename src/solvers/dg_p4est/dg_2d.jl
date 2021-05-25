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
                              nonconservative_terms::Val{false},
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


function prolong2boundaries!(cache, u,
                             mesh::P4estMesh{2}, equations, dg::DG)
  @unpack boundaries = cache

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for boundary in eachboundary(dg, cache)
    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]

    # Use Tuple `node_indices` and `evaluate_index` to copy values
    # from the correct face and in the correct orientation
    for i in eachnode(dg), v in eachvariable(equations)
      boundaries.u[v, i, boundary] = u[v, evaluate_index(node_indices, size_, 1, i),
                                          evaluate_index(node_indices, size_, 2, i),
                                          element]
    end
  end

  return nothing
end


function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
                             mesh::P4estMesh, equations, dg::DG)
  @unpack boundaries = cache
  @unpack surface_flux_values, node_coordinates = cache.elements
  @unpack surface_flux = dg

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for local_index in eachindex(boundary_indexing)
    # Use the local index to get the global boundary index from the pre-sorted list
    boundary = boundary_indexing[local_index]

    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]
    direction     = indices2direction(node_indices)

    # Use Tuple `node_indices` and `evaluate_index` to access node indices
    # at the correct face and in the correct orientation to get normal vectors
    for i in eachnode(dg)
      node_i = evaluate_index(node_indices, size_, 1, i)
      node_j = evaluate_index(node_indices, size_, 2, i)

      # Extract solution data from boundary container
      u_inner = get_node_vars(boundaries.u, equations, dg, i, boundary)

      # Outward-pointing normal vector
      normal_vector = get_normal_vector(direction, cache, node_i, node_j, element)

      # Coordinates at boundary node
      x = get_node_coords(node_coordinates, equations, dg, node_i, node_j, element)

      flux_ = boundary_condition(u_inner, normal_vector, x, t, surface_flux, equations)

      # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
      # to left and right element storage in the correct orientation
      for v in eachvariable(equations)
        surface_index = evaluate_index_surface(node_indices, size_, 1, i)
        surface_flux_values[v, surface_index, direction, element] = flux_[v]
      end
    end
  end
end


function prolong2mortars!(cache, u,
                          mesh::P4estMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)
  @unpack element_ids, node_indices = cache.mortars

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for mortar in eachmortar(dg, cache)
    small_node_indices  = node_indices[1, mortar]
    large_node_indices  = node_indices[2, mortar]

    # Copy solution small to small
    for pos in 1:2, i in eachnode(dg), v in eachvariable(equations)
      # Use Tuple `node_indices` and `evaluate_index` to copy values
      # from the correct face and in the correct orientation
      mortars.u[1, pos, v, i, mortar] = u[v, evaluate_index(small_node_indices, size_, 1, i),
                                             evaluate_index(small_node_indices, size_, 2, i),
                                             element_ids[pos, mortar]]
    end

    # Copy solution large to large
    for i in eachnode(dg), v in eachvariable(equations)
      # Use Tuple `node_indices` and `evaluate_index` to copy values
      # from the correct face and in the correct orientation
      mortars.u_large[v, i, mortar] = u[v, evaluate_index(large_node_indices, size_, 1, i),
                                           evaluate_index(large_node_indices, size_, 2, i),
                                           element_ids[3, mortar]]
    end

    # Interpolate large element face data to small face locations
    u_large = view(cache.mortars.u_large, :, :, mortar)

    multiply_dimensionwise!(view(cache.mortars.u, 2, 1, :, :, mortar),
                            mortar_l2.forward_lower,
                            u_large)
    multiply_dimensionwise!(view(cache.mortars.u, 2, 2, :, :, mortar),
                            mortar_l2.forward_upper,
                            u_large)
  end

  return nothing
end


function calc_mortar_flux!(surface_flux_values,
                           mesh::P4estMesh{2},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DG, cache)
  # @unpack neighbor_ids, u_lower, u_upper, orientations = cache.mortars
  # @unpack fstar_upper_threaded, fstar_lower_threaded = cache

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, dg, u_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, dg, u_lower, mortar, orientation)

    mortar_fluxes_to_elements!(surface_flux_values, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
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
