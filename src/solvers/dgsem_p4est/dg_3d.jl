# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::P4estMesh{3}, equations, mortar_l2::LobattoLegendreMortarL2, uEltype)
  # TODO: Taal compare performance of different types
  fstar_threaded = [Array{uEltype, 4}(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2), 4)
                    for _ in 1:Threads.nthreads()]

  fstar_tmp_threaded = [Array{uEltype, 3}(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                         for _ in 1:Threads.nthreads()]
  u_threaded          = [Array{uEltype, 3}(undef, nvariables(equations), nnodes(mortar_l2), nnodes(mortar_l2))
                         for _ in 1:Threads.nthreads()]

  (; fstar_threaded, fstar_tmp_threaded, u_threaded)
end


# TODO: p4est interface performance, generalize (2D, 3D) and document
# !!! warning
#     This assumes that loops using the return values are ordered as
#
#     my_i, my_j, my_k = # begin values
#     for j in ...
#       for i in ...
#         # do stuff with (i, j) and (my_i, my_j, my_k)
#         my_[ijk] += my_[ijk]_step_i
#       end
#       my_[ijk] += my_[ijk]_step_j
#     end
@inline function index_to_start_step_3d(index::Symbol, index_range)
  index_begin = first(index_range)
  index_end   = last(index_range)

  if index === :one
    return index_begin, 0, 0
  elseif index === :end
    return index_end, 0, 0
  elseif index === :i
    return index_begin, 1, index_begin - index_end - 1
  elseif index === :i_backwards
    return index_end, -1, index_end + 1 - index_begin
  elseif index === :j
    return index_begin, 0, 1
  else # if index === :j_backwards
    return index_end, 0, -1
  end
end

function prolong2interfaces!(cache, u,
                             mesh::P4estMesh{3},
                             equations, surface_integral, dg::DG)
  @unpack interfaces = cache
  index_range = eachnode(dg)

  @threaded for interface in eachinterface(dg, cache)
    # Copy solution data from the primary element on a case-by-case basis
    # to get the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the indices of the primary side
    # will always run forwards.
    primary_element = interfaces.element_ids[1, interface]
    primary_indices = interfaces.node_indices[1, interface]

    i_primary_start, i_primary_step_i, i_primary_step_j = index_to_start_step_3d(primary_indices[1], index_range)
    j_primary_start, j_primary_step_i, j_primary_step_j = index_to_start_step_3d(primary_indices[2], index_range)
    k_primary_start, k_primary_step_i, k_primary_step_j = index_to_start_step_3d(primary_indices[3], index_range)

    i_primary = i_primary_start
    j_primary = j_primary_start
    k_primary = k_primary_start
    for j in eachnode(dg)
      for i in eachnode(dg)
        for v in eachvariable(equations)
          interfaces.u[1, v, i, j, interface] = u[v, i_primary, j_primary, k_primary, primary_element]
        end
        i_primary += i_primary_step_i
        j_primary += j_primary_step_i
        k_primary += k_primary_step_i
      end
      i_primary += i_primary_step_j
      j_primary += j_primary_step_j
      k_primary += k_primary_step_j
    end

    # Copy solution data from the secondary element on a case-by-case basis
    # to get the correct face and orientation.
    secondary_element = interfaces.element_ids[2, interface]
    secondary_indices = interfaces.node_indices[2, interface]

    i_secondary_start, i_secondary_step_i, i_secondary_step_j = index_to_start_step_3d(secondary_indices[1], index_range)
    j_secondary_start, j_secondary_step_i, j_secondary_step_j = index_to_start_step_3d(secondary_indices[2], index_range)
    k_secondary_start, k_secondary_step_i, k_secondary_step_j = index_to_start_step_3d(secondary_indices[3], index_range)

    i_secondary = i_secondary_start
    j_secondary = j_secondary_start
    k_secondary = k_secondary_start
    for j in eachnode(dg)
      for i in eachnode(dg)
        for v in eachvariable(equations)
          interfaces.u[2, v, i, j, interface] = u[v, i_secondary, j_secondary, k_secondary, secondary_element]
        end
        i_secondary += i_secondary_step_i
        j_secondary += j_secondary_step_i
        k_secondary += k_secondary_step_i
      end
      i_secondary += i_secondary_step_j
      j_secondary += j_secondary_step_j
      k_secondary += k_secondary_step_j
    end
  end

  return nothing
end


function calc_interface_flux!(surface_flux_values,
                              mesh::P4estMesh{3},
                              nonconservative_terms::Val{false},
                              equations, surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
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
                             equations, surface_integral, dg::DG)
  @unpack boundaries = cache

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  @threaded for boundary in eachboundary(dg, cache)
    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]

    # Use Tuple `node_indices` and `evaluate_index` to copy values
    # from the correct face and in the correct orientation
    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        boundaries.u[v, i, j, boundary] = u[v, evaluate_index(node_indices, size_, 1, i, j),
                                               evaluate_index(node_indices, size_, 2, i, j),
                                               evaluate_index(node_indices, size_, 3, i, j),
                                               element]
      end
    end
  end

  return nothing
end


function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
                             mesh::P4estMesh{3},
                             equations, surface_integral, dg::DG)
  @unpack boundaries = cache
  @unpack surface_flux_values, node_coordinates = cache.elements
  @unpack surface_flux = surface_integral

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
                          mortar_l2::LobattoLegendreMortarL2,
                          surface_integral, dg::DGSEM)
  # temporary buffer for projections
  @unpack fstar_tmp_threaded = cache
  @unpack element_ids, node_indices = cache.mortars

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  @threaded for mortar in eachmortar(dg, cache)
    fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

    small_indices = node_indices[1, mortar]
    large_indices = node_indices[2, mortar]

    # Copy solution small to small
    for pos in 1:4
      for j in eachnode(dg), i in eachnode(dg)
        for v in eachvariable(equations)
          # Use Tuple `node_indices` and `evaluate_index` to copy values
          # from the correct face and in the correct orientation
          cache.mortars.u[1, v, pos, i, j, mortar] = u[v, evaluate_index(small_indices, size_, 1, i, j),
                                                          evaluate_index(small_indices, size_, 2, i, j),
                                                          evaluate_index(small_indices, size_, 3, i, j),
                                                          element_ids[pos, mortar]]
        end
      end
    end

    # Buffer to copy solution values of the large element in the correct orientation
    # before interpolating
    u_buffer = cache.u_threaded[Threads.threadid()]

    # Copy solution of large element face to buffer in the correct orientation
    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        # Use Tuple `node_indices` and `evaluate_index` to copy values
        # from the correct face and in the correct orientation
        u_buffer[v, i, j] = u[v, evaluate_index(large_indices, size_, 1, i, j),
                                 evaluate_index(large_indices, size_, 2, i, j),
                                 evaluate_index(large_indices, size_, 3, i, j),
                                 element_ids[5, mortar]]
      end
    end

    # Interpolate large element face data from buffer to small face locations
    multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, :, mortar),
                            mortar_l2.forward_lower,
                            mortar_l2.forward_lower,
                            u_buffer,
                            fstar_tmp)
    multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, :, mortar),
                            mortar_l2.forward_upper,
                            mortar_l2.forward_lower,
                            u_buffer,
                            fstar_tmp)
    multiply_dimensionwise!(view(cache.mortars.u, 2, :, 3, :, :, mortar),
                            mortar_l2.forward_lower,
                            mortar_l2.forward_upper,
                            u_buffer,
                            fstar_tmp)
    multiply_dimensionwise!(view(cache.mortars.u, 2, :, 4, :, :, mortar),
                            mortar_l2.forward_upper,
                            mortar_l2.forward_upper,
                            u_buffer,
                            fstar_tmp)
  end

  return nothing
end


function calc_mortar_flux!(surface_flux_values,
                           mesh::P4estMesh{3},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
  @unpack u, element_ids, node_indices = cache.mortars
  @unpack fstar_threaded, fstar_tmp_threaded = cache
  @unpack surface_flux = surface_integral

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar = fstar_threaded[Threads.threadid()]
    fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    # Use Tuple `node_indices` and `evaluate_index` to access node indices
    # at the correct face and in the correct orientation to get normal vectors
    for pos in 1:4
      for j in eachnode(dg), i in eachnode(dg)
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, pos, i, j, mortar)

        normal_vector = get_normal_vector(small_direction, cache,
                                          evaluate_index(small_indices, size_, 1, i, j),
                                          evaluate_index(small_indices, size_, 2, i, j),
                                          evaluate_index(small_indices, size_, 3, i, j),
                                          element_ids[pos, mortar])

        flux_ = surface_flux(u_ll, u_rr, normal_vector, equations)

        # Copy flux to buffer
        set_node_vars!(fstar, flux_, equations, dg, i, j, pos)
      end
    end

    # Buffer to interpolate flux values of the large element to before copying
    # in the correct orientation
    u_buffer = cache.u_threaded[Threads.threadid()]

    mortar_fluxes_to_elements!(surface_flux_values,
                               mesh, equations, mortar_l2, dg, cache,
                               mortar, fstar, u_buffer, fstar_tmp)
  end

  return nothing
end


@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::P4estMesh{3}, equations,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache, mortar, fstar, u_buffer, fstar_tmp)
  @unpack element_ids, node_indices = cache.mortars

  small_indices  = node_indices[1, mortar]
  large_indices  = node_indices[2, mortar]

  small_direction = indices2direction(small_indices)
  large_direction = indices2direction(large_indices)

  size_ = (nnodes(dg), nnodes(dg), nnodes(dg))

  # Copy solution small to small
  for pos in 1:4
    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
        # to left and right element storage in the correct orientation
        surface_index1 = evaluate_index_surface(small_indices, size_, 1, i, j)
        surface_index2 = evaluate_index_surface(small_indices, size_, 2, i, j)
        surface_flux_values[v, surface_index1, surface_index2, small_direction,
                            element_ids[pos, mortar]] = fstar[v, i, j, pos]
      end
    end
  end

  large_element = element_ids[5, mortar]

  # Project small fluxes to large element.
  multiply_dimensionwise!(
    u_buffer,
    mortar_l2.reverse_lower, mortar_l2.reverse_lower,
    view(fstar, .., 1),
    fstar_tmp)
  add_multiply_dimensionwise!(
    u_buffer,
    mortar_l2.reverse_upper, mortar_l2.reverse_lower,
    view(fstar, .., 2),
    fstar_tmp)
  add_multiply_dimensionwise!(
    u_buffer,
    mortar_l2.reverse_lower, mortar_l2.reverse_upper,
    view(fstar, .., 3),
    fstar_tmp)
  add_multiply_dimensionwise!(
    u_buffer,
    mortar_l2.reverse_upper, mortar_l2.reverse_upper,
    view(fstar, .., 4),
    fstar_tmp)

  # The flux is calculated in the outward direction of the small elements,
  # so the sign must be switched to get the flux in outward direction
  # of the large element.
  # The contravariant vectors of the large element (and therefore the normal vectors
  # of the large element as well) are four times as large as the contravariant vectors
  # of the small elements. Therefore, the flux need to be scaled by a factor of 4
  # to obtain the flux of the large element.
  u_buffer .*= -4

  # Copy interpolated flux values from buffer to large element face in the correct orientation
  for j in eachnode(dg), i in eachnode(dg)
    for v in eachvariable(equations)
      # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
      # to surface flux storage in the correct orientation
      surface_index1 = evaluate_index_surface(large_indices, size_, 1, i, j)
      surface_index2 = evaluate_index_surface(large_indices, size_, 2, i, j)
      surface_flux_values[v, surface_index1, surface_index2,
                          large_direction, large_element] = u_buffer[v, i, j]
    end
  end

  return nothing
end


function calc_surface_integral!(du, u,
                                mesh::P4estMesh{3},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM, cache)
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


end # @muladd
