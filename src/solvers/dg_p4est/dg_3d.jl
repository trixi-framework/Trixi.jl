# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::P4estMesh{3}, equations, mortar_l2::LobattoLegendreMortarL2, uEltype)
  return (;)
end


function prolong2interfaces!(cache, u,
                             mesh::P4estMesh{3},
                             equations, dg::DG)
  @unpack interfaces = cache

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  @threaded for interface in eachinterface(dg, cache)
    primary_element   = interfaces.element_ids[1, interface]
    secondary_element = interfaces.element_ids[2, interface]

    primary_indices   = interfaces.node_indices[1, interface]
    secondary_indices = interfaces.node_indices[2, interface]

    # Use Tuple `node_indices` and `evaluate_index` to copy values
    # from the correct face and in the correct orientation
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      interfaces.u[1, v, i, j, interface] = u[v, evaluate_index(primary_indices, size_, 1, i, j),
                                                 evaluate_index(primary_indices, size_, 2, i, j),
                                                 evaluate_index(primary_indices, size_, 3, i, j),
                                                 primary_element]

      interfaces.u[2, v, i, j, interface] = u[v, evaluate_index(secondary_indices, size_, 1, i, j),
                                                 evaluate_index(secondary_indices, size_, 2, i, j),
                                                 evaluate_index(secondary_indices, size_, 3, i, j),
                                                 secondary_element]
    end
  end

  return nothing
end


function calc_interface_flux!(surface_flux_values,
                              mesh::P4estMesh{3},
                              nonconservative_terms::Val{false},
                              equations, dg::DG, cache)
  @unpack surface_flux = dg
  @unpack u, element_ids, node_indices = cache.interfaces

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

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
    for j in eachnode(dg), i in eachnode(dg)
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, j, interface)

      normal_vector = get_normal_vector(primary_direction, cache,
                                        evaluate_index(primary_indices, size_, 1, i, j),
                                        evaluate_index(primary_indices, size_, 2, i, j),
                                        evaluate_index(primary_indices, size_, 3, i, j),
                                        primary_element)

      flux_ = surface_flux(u_ll, u_rr, normal_vector, equations)

      # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
      # to left and right element storage in the correct orientation
      for v in eachvariable(equations)
        surf_i = evaluate_index_surface(primary_indices, size_, 1, i, j)
        surf_j = evaluate_index_surface(primary_indices, size_, 2, i, j)
        surface_flux_values[v, surf_i, surf_j, primary_direction, primary_element] = flux_[v]

        surf_i = evaluate_index_surface(secondary_indices, size_, 1, i, j)
        surf_j = evaluate_index_surface(secondary_indices, size_, 2, i, j)
        surface_flux_values[v, surf_i, surf_j, secondary_direction, secondary_element] = -flux_[v]
      end
    end
  end

  return nothing
end


function prolong2boundaries!(cache, u,
                             mesh::P4estMesh{3},
                             equations, dg::DG)
  @unpack boundaries = cache

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  @threaded for boundary in eachboundary(dg, cache)
    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]

    # Use Tuple `node_indices` and `evaluate_index` to copy values
    # from the correct face and in the correct orientation
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      boundaries.u[v, i, j, boundary] = u[v, evaluate_index(node_indices, size_, 1, i, j),
                                             evaluate_index(node_indices, size_, 2, i, j),
                                             evaluate_index(node_indices, size_, 3, i, j),
                                             element]
    end
  end

  return nothing
end


function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
                             mesh::P4estMesh{3}, equations, dg::DG)
  @unpack boundaries = cache
  @unpack surface_flux_values, node_coordinates = cache.elements
  @unpack surface_flux = dg

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  @threaded for local_index in eachindex(boundary_indexing)
    # Use the local index to get the global boundary index from the pre-sorted list
    boundary = boundary_indexing[local_index]

    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]
    direction     = indices2direction(node_indices)

    # Use Tuple `node_indices` and `evaluate_index` to access node indices
    # at the correct face and in the correct orientation to get normal vectors
    for j in eachnode(dg), i in eachnode(dg)
      node_i = evaluate_index(node_indices, size_, 1, i, j)
      node_j = evaluate_index(node_indices, size_, 2, i, j)
      node_k = evaluate_index(node_indices, size_, 3, i, j)

      # Extract solution data from boundary container
      u_inner = get_node_vars(boundaries.u, equations, dg, i, j, boundary)

      # Outward-pointing normal vector
      normal_vector = get_normal_vector(direction, cache, node_i, node_j, node_k, element)

      # Coordinates at boundary node
      x = get_node_coords(node_coordinates, equations, dg, node_i, node_j, node_k, element)

      flux_ = boundary_condition(u_inner, normal_vector, x, t, surface_flux, equations)

      # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
      # to left and right element storage in the correct orientation
      for v in eachvariable(equations)
        surf_i = evaluate_index_surface(node_indices, size_, 1, i, j)
        surf_j = evaluate_index_surface(node_indices, size_, 2, i, j)
        surface_flux_values[v, surf_i, surf_j, direction, element] = flux_[v]
      end
    end
  end
end


function prolong2mortars!(cache, u,
                          mesh::P4estMesh{3}, equations,
                          mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)

  return nothing
end


function calc_mortar_flux!(surface_flux_values,
                           mesh::P4estMesh{3},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DG, cache)

  return nothing
end


function calc_surface_integral!(du, mesh::P4estMesh{3},
                                equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  # Note that all fluxes have been computed with outward-pointing normal vectors
  @threaded for element in eachelement(dg, cache)
    for m in eachnode(dg), l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, m, element] += (surface_flux_values[v, l, m, 1, element]
                                             * boundary_interpolation[1, 1])
        # surface at +x
        du[v, nnodes(dg), l, m, element] += (surface_flux_values[v, l, m, 2, element]
                                             * boundary_interpolation[nnodes(dg), 2])
        # surface at -y
        du[v, l, 1,          m, element] += (surface_flux_values[v, l, m, 3, element]
                                             * boundary_interpolation[1, 1])
        # surface at +y
        du[v, l, nnodes(dg), m, element] += (surface_flux_values[v, l, m, 4, element]
                                             * boundary_interpolation[nnodes(dg), 2])
        # surface at -z
        du[v, l, m, 1,          element] += (surface_flux_values[v, l, m, 5, element]
                                             * boundary_interpolation[1,          1])
        # surface at +z
        du[v, l, m, nnodes(dg), element] += (surface_flux_values[v, l, m, 6, element]
                                             * boundary_interpolation[nnodes(dg), 2])
      end
    end
  end

  return nothing
end
