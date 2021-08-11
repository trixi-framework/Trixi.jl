# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# The methods below are specialized on the mortar type
# and called from the basic `create_cache` method at the top.
function create_cache(mesh::P4estMesh{2}, equations, mortar_l2::LobattoLegendreMortarL2, uEltype)
  # TODO: Taal performance using different types
  MA2d = MArray{Tuple{nvariables(equations), nnodes(mortar_l2)},
                uEltype, 2,
                nvariables(equations) * nnodes(mortar_l2)}
  fstar_upper_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
  fstar_lower_threaded = MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]
  u_threaded =           MA2d[MA2d(undef) for _ in 1:Threads.nthreads()]

  (; fstar_upper_threaded, fstar_lower_threaded, u_threaded)
end


# TODO: p4est interface performance, move and generalzie this function for 3D
@inline function index_to_start_step(index::Symbol, idx_one::Int, idx_end::Int)
  if index === :one
    return idx_one, 0
  elseif index === :end
    return idx_end, 0
  elseif index === :i
    return idx_one, 1
  else # if index === :i_backwards
    return idx_end, -1
  end
end

function prolong2interfaces!(cache, u,
                             mesh::P4estMesh{2},
                             equations, surface_integral, dg::DG)
  @unpack interfaces = cache
  idx_one = 1
  idx_end = nnodes(dg)

  @threaded for interface in eachinterface(dg, cache)
    # Copy solution data from the primary element on a case-by-case basis to get
    # the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the index of the primary side
    # will always run forwards.
    primary_element = interfaces.element_ids[1, interface]
    primary_indices = interfaces.node_indices[1, interface]

    i_primary_start, i_primary_step = index_to_start_step(
      primary_indices[1], idx_one, idx_end)
    j_primary_start, j_primary_step = index_to_start_step(
      primary_indices[2], idx_one, idx_end)

    i_primary = i_primary_start
    j_primary = j_primary_start
    for i in eachnode(dg)
      for v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i_primary, j_primary, primary_element]
      end
      i_primary += i_primary_step
      j_primary += j_primary_step
    end

    # Copy solution data from the secondary element on a case-by-case basis to get
    # the correct face and orientation.
    secondary_element = interfaces.element_ids[2, interface]
    secondary_indices = interfaces.node_indices[2, interface]

    i_secondary_start, i_secondary_step = index_to_start_step(
      secondary_indices[1], idx_one, idx_end)
    j_secondary_start, j_secondary_step = index_to_start_step(
      secondary_indices[2], idx_one, idx_end)

    i_secondary = i_secondary_start
    j_secondary = j_secondary_start
    for i in eachnode(dg)
      for v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, i_secondary, j_secondary, secondary_element]
      end
      i_secondary += i_secondary_step
      j_secondary += j_secondary_step
    end
  end

  return nothing
end


function calc_interface_flux!(surface_flux_values,
                              mesh::P4estMesh{2},
                              nonconservative_terms::Val{false},
                              equations, surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
  @unpack u, element_ids, node_indices = cache.interfaces
  @unpack contravariant_vectors = cache.elements
  idx_one = 1
  idx_end = nnodes(dg)

  @threaded for interface in eachinterface(dg, cache)
    # Get information on the primary element, compute the surface fluxes,
    # and store them for the primary element
    primary_element  = element_ids[1, interface]
    primary_indices  = node_indices[1, interface]
    primary_direction = indices2direction(primary_indices)

    i_primary_start, i_primary_step = index_to_start_step(
      primary_indices[1], idx_one, idx_end)
    j_primary_start, j_primary_step = index_to_start_step(
      primary_indices[2], idx_one, idx_end)

    i_primary = i_primary_start
    j_primary = j_primary_start
    for i in eachnode(dg)
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)

      # Contravariant vectors at interfaces in negative coordinate direction
      # are pointing inwards. This is handled by `get_normal_direction`.
      normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                                              i_primary, j_primary, primary_element)
      flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)

      for v in eachvariable(equations)
        surface_flux_values[v, i, primary_direction, primary_element] = flux_[v]
      end

      i_primary += i_primary_step
      j_primary += j_primary_step
    end

    # Get information on the secondary element and copy the numerical fluxes
    # from the primary element to the secondary one
    secondary_element = element_ids[2, interface]
    secondary_indices = node_indices[2, interface]
    secondary_direction = indices2direction(secondary_indices)

    # Note that the index of the primary side will always run forward but
    # the secondary index might need to run backwards for flipped sides.
    if :i_backwards in secondary_indices
      for i in eachnode(dg), v in eachvariable(equations)
        surface_flux_values[v, idx_end + 1 - i, secondary_direction, secondary_element] =
          -surface_flux_values[v, i, primary_direction, primary_element]
      end
    else
      for i in eachnode(dg), v in eachvariable(equations)
        surface_flux_values[v, i, secondary_direction, secondary_element] =
          -surface_flux_values[v, i, primary_direction, primary_element]
      end
    end
  end

  return nothing
end


function prolong2boundaries!(cache, u,
                             mesh::P4estMesh{2},
                             equations, surface_integral, dg::DG)
  @unpack boundaries = cache
  idx_one = 1
  idx_end = nnodes(dg)

  @threaded for boundary in eachboundary(dg, cache)
    # Copy solution data from the element on a case-by-case basis to get
    # the correct face and orientation.
    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]

    i_node_start, i_node_step = index_to_start_step(
      node_indices[1], idx_one, idx_end)
    j_node_start, j_node_step = index_to_start_step(
      node_indices[2], idx_one, idx_end)

    i_node = i_node_start
    j_node = j_node_start
    for i in eachnode(dg)
      for v in eachvariable(equations)
        boundaries.u[v, i, boundary] = u[v, i_node, j_node, element]
      end
      i_node += i_node_step
      j_node += j_node_step
    end
  end

  return nothing
end


function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
                             mesh::P4estMesh{2},
                             equations, surface_integral, dg::DG)
  @unpack boundaries = cache
  @unpack surface_flux_values, node_coordinates, contravariant_vectors = cache.elements
  @unpack surface_flux = surface_integral
  idx_one = 1
  idx_end = nnodes(dg)

  @threaded for local_index in eachindex(boundary_indexing)
    # Use the local index to get the global boundary index from the pre-sorted list
    boundary = boundary_indexing[local_index]

    # Get information on the adjacent element, compute the surface fluxes,
    # and store them
    element       = boundaries.element_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]
    direction     = indices2direction(node_indices)

    i_node_start, i_node_step = index_to_start_step(
      node_indices[1], idx_one, idx_end)
    j_node_start, j_node_step = index_to_start_step(
      node_indices[2], idx_one, idx_end)

    i_node = i_node_start
    j_node = j_node_start
    for i in eachnode(dg)
      # Extract solution data from boundary container
      u_inner = get_node_vars(boundaries.u, equations, dg, i, boundary)

      # Outward-pointing normal vector
      normal_direction = get_normal_direction(direction, contravariant_vectors, i_node, j_node, element)

      # Coordinates at boundary node
      x = get_node_coords(node_coordinates, equations, dg, i_node, j_node, element)

      flux_ = boundary_condition(u_inner, normal_direction, x, t, surface_flux, equations)

      # Copy flux to element storage in the correct orientation
      for v in eachvariable(equations)
        surface_flux_values[v, i, direction, element] = flux_[v]
      end

      i_node += i_node_step
      j_node += j_node_step
    end
  end
end


function prolong2mortars!(cache, u,
                          mesh::P4estMesh{2}, equations,
                          mortar_l2::LobattoLegendreMortarL2,
                          surface_integral, dg::DGSEM)
  @unpack element_ids, node_indices = cache.mortars

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for mortar in eachmortar(dg, cache)
    small_indices = node_indices[1, mortar]
    large_indices = node_indices[2, mortar]

    # Copy solution small to small
    for pos in 1:2
      for i in eachnode(dg)
        for v in eachvariable(equations)
          # Use Tuple `node_indices` and `evaluate_index` to copy values
          # from the correct face and in the correct orientation
          cache.mortars.u[1, v, pos, i, mortar] = u[v, evaluate_index(small_indices, size_, 1, i),
                                                       evaluate_index(small_indices, size_, 2, i),
                                                       element_ids[pos, mortar]]
        end
      end
    end

    # Buffer to copy solution values of the large element in the correct orientation
    # before interpolating
    u_buffer = cache.u_threaded[Threads.threadid()]

    # Copy solution of large element face to buffer in the correct orientation
    for i in eachnode(dg)
      for v in eachvariable(equations)
        # Use Tuple `node_indices` and `evaluate_index` to copy values
        # from the correct face and in the correct orientation
        u_buffer[v, i] = u[v, evaluate_index(large_indices, size_, 1, i),
                              evaluate_index(large_indices, size_, 2, i),
                              element_ids[3, mortar]]
      end
    end

    # Interpolate large element face data from buffer to small face locations
    multiply_dimensionwise!(view(cache.mortars.u, 2, :, 1, :, mortar),
                            mortar_l2.forward_lower,
                            u_buffer)
    multiply_dimensionwise!(view(cache.mortars.u, 2, :, 2, :, mortar),
                            mortar_l2.forward_upper,
                            u_buffer)
  end

  return nothing
end


function calc_mortar_flux!(surface_flux_values,
                           mesh::P4estMesh{2},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2,
                           surface_integral, dg::DG, cache)
  @unpack u, element_ids, node_indices = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache
  @unpack surface_flux = surface_integral

  size_ = (nnodes(dg), nnodes(dg))

  @threaded for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar = (fstar_lower_threaded[Threads.threadid()],
             fstar_upper_threaded[Threads.threadid()])

    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)

    # Use Tuple `node_indices` and `evaluate_index` to access node indices
    # at the correct face and in the correct orientation to get normal vectors
    for pos in 1:2
      for i in eachnode(dg)
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, pos, i, mortar)

        normal_vector = get_normal_vector(small_direction, cache,
                                          evaluate_index(small_indices, size_, 1, i),
                                          evaluate_index(small_indices, size_, 2, i),
                                          element_ids[pos, mortar])

        flux_ = surface_flux(u_ll, u_rr, normal_vector, equations)

        # Copy flux to buffer
        set_node_vars!(fstar[pos], flux_, equations, dg, i)
      end
    end

    # Buffer to interpolate flux values of the large element to before copying
    # in the correct orientation
    u_buffer = cache.u_threaded[Threads.threadid()]

    mortar_fluxes_to_elements!(surface_flux_values,
                               mesh, equations, mortar_l2, dg, cache,
                               mortar, fstar, u_buffer)
  end

  return nothing
end


@inline function mortar_fluxes_to_elements!(surface_flux_values,
                                            mesh::P4estMesh{2}, equations,
                                            mortar_l2::LobattoLegendreMortarL2,
                                            dg::DGSEM, cache, mortar, fstar, u_buffer)
  @unpack element_ids, node_indices = cache.mortars

  small_indices  = node_indices[1, mortar]
  large_indices  = node_indices[2, mortar]

  small_direction = indices2direction(small_indices)
  large_direction = indices2direction(large_indices)

  size_ = (nnodes(dg), nnodes(dg))

  # Copy solution small to small
  for pos in 1:2
    for i in eachnode(dg)
      for v in eachvariable(equations)
        # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
        # to left and right element storage in the correct orientation
        surface_index = evaluate_index_surface(small_indices, size_, 1, i)
        surface_flux_values[v, surface_index,
                            small_direction,
                            element_ids[pos, mortar]] = fstar[pos][v, i]
      end
    end
  end

  large_element = element_ids[3, mortar]

  # Project small fluxes to large element.
  # TODO: Taal performance, see comment in dg_tree/dg_2d.jl
  multiply_dimensionwise!(u_buffer,
                          mortar_l2.reverse_upper, fstar[2],
                          mortar_l2.reverse_lower, fstar[1])

  # The flux is calculated in the outward direction of the small elements,
  # so the sign must be switched to get the flux in outward direction
  # of the large element.
  # The contravariant vectors of the large element (and therefore the normal vectors
  # of the large element as well) are twice as large as the contravariant vectors
  # of the small elements. Therefore, the flux need to be scaled by a factor of 2
  # to obtain the flux of the large element.
  u_buffer .*= -2

  # Copy interpolated flux values from buffer to large element face in the correct orientation
  for i in eachnode(dg)
    for v in eachvariable(equations)
      # Use Tuple `node_indices` and `evaluate_index_surface` to copy flux
      # to surface flux storage in the correct orientation
      surface_index = evaluate_index_surface(large_indices, size_, 1, i)
      surface_flux_values[v, surface_index, large_direction, large_element] = u_buffer[v, i]
    end
  end

  return nothing
end


function calc_surface_integral!(du, u,
                                mesh::P4estMesh{2},
                                equations,
                                surface_integral::SurfaceIntegralWeakForm,
                                dg::DGSEM, cache)
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


end # @muladd
