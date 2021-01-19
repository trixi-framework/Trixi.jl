# This file collects all methods that have been updated to work with parabolic systems of equations

function rhs!(du::AbstractArray{<:Any,4}, u, gradients, t,
              mesh::TreeMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache, cache_gradients)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, gradients, have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache)

  # Prolong solution and gradients to interfaces
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, cache_gradients, u, gradients, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(cache.elements.surface_flux_values,
                                                              have_nonconservative_terms(equations), equations,
                                                              dg, cache, cache_gradients)

  # Prolong solution and gradients to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, equations, dg)

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, t, boundary_conditions, equations, dg)

  # Prolong solution and gradients to mortars
  @timeit_debug timer() "prolong2mortars" prolong2mortars!(cache, cache_gradients, u, gradients,
                                                           equations, dg.mortar, dg)

  # Calculate mortar fluxes
  @timeit_debug timer() "mortar flux" calc_mortar_flux!(cache.elements.surface_flux_values,
                                                        have_nonconservative_terms(equations), equations,
                                                        dg.mortar, dg, cache, cache_gradients)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, gradients,
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis

  Threads.@threads for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)
      gradients_x_node = get_node_vars(gradients[1], equations, dg, i, j, element)
      gradients_y_node = get_node_vars(gradients[2], equations, dg, i, j, element)
      gradients_node = (gradients_x_node, gradients_y_node)

      flux1 = calcflux(u_node, gradients_node, 1, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      flux2 = calcflux(u_node, gradients_node, 2, equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


function prolong2interfaces!(cache, cache_gradients, u::AbstractArray{<:Any,4}, gradients,
                             equations, dg::DG)
  prolong2interfaces!(cache, u, equations, dg)
  prolong2interfaces!(cache_gradients[1], gradients[1], equations, dg)
  prolong2interfaces!(cache_gradients[2], gradients[2], equations, dg)

  return nothing
end


function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,4},
                              nonconservative_terms::Val{false}, equations,
                              dg::DG, cache, cache_gradients)
  @unpack surface_flux = dg
  @unpack u, neighbor_ids, orientations = cache.interfaces
  gradients_x = cache_gradients[1].interfaces.u
  gradients_y = cache_gradients[2].interfaces.u

  Threads.@threads for interface in eachinterface(dg, cache)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for i in eachnode(dg)
      # Call pointwise Riemann solver
      u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
      gradients_x_ll, gradients_x_rr = get_surface_node_vars(gradients_x, equations, dg, i, interface)
      gradients_y_ll, gradients_y_rr = get_surface_node_vars(gradients_y, equations, dg, i, interface)
      gradients_ll = (gradients_x_ll, gradients_y_ll)
      gradients_rr = (gradients_x_rr, gradients_y_rr)
      flux = surface_flux(u_ll, u_rr, gradients_ll, gradients_rr, orientations[interface], equations)

      # Copy flux to left and right element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, left_direction,  left_id]  = flux[v]
        surface_flux_values[v, i, right_direction, right_id] = flux[v]
      end
    end
  end

  return nothing
end


function prolong2mortars!(cache, cache_gradients, u::AbstractArray{<:Any,4}, gradients,
                          equations, mortar_l2::LobattoLegendreMortarL2, dg::DGSEM)
  prolong2mortars!(cache, u, equations, mortar_l2, dg)
  prolong2mortars!(cache_gradients[1], gradients[1], equations, mortar_l2, dg)
  prolong2mortars!(cache_gradients[2], gradients[2], equations, mortar_l2, dg)

  return nothing
end


function calc_mortar_flux!(surface_flux_values::AbstractArray{<:Any,4},
                           nonconservative_terms::Val{false}, equations,
                           mortar_l2::LobattoLegendreMortarL2, dg::DG, cache, cache_gradients)
  @unpack neighbor_ids, u_lower, u_upper, orientations = cache.mortars
  @unpack fstar_upper_threaded, fstar_lower_threaded = cache
  gradients_x_upper = cache_gradients[1].mortars.u_upper
  gradients_x_lower = cache_gradients[1].mortars.u_lower
  gradients_y_upper = cache_gradients[2].mortars.u_upper
  gradients_y_lower = cache_gradients[2].mortars.u_lower
  gradients_upper = (gradients_x_upper, gradients_y_upper)
  gradients_lower = (gradients_x_lower, gradients_y_lower)

  Threads.@threads for mortar in eachmortar(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_upper = fstar_upper_threaded[Threads.threadid()]
    fstar_lower = fstar_lower_threaded[Threads.threadid()]

    # Calculate fluxes
    orientation = orientations[mortar]
    calc_fstar!(fstar_upper, equations, dg, u_upper, gradients_upper, mortar, orientation)
    calc_fstar!(fstar_lower, equations, dg, u_lower, gradients_lower, mortar, orientation)

    mortar_fluxes_to_elements!(surface_flux_values, equations, mortar_l2, dg, cache,
                               mortar, fstar_upper, fstar_lower)
  end

  return nothing
end


@inline function calc_fstar!(destination::AbstractArray{<:Any,2}, equations, dg::DGSEM,
                             u_interfaces, gradients_interfaces, interface, orientation)
  @unpack surface_flux = dg

  for i in eachnode(dg)
    # Call pointwise two-point numerical flux function
    u_ll, u_rr = get_surface_node_vars(u_interfaces, equations, dg, i, interface)
    gradients_x_ll, gradients_x_rr = get_surface_node_vars(gradients_interfaces[1], equations, dg, i, interface)
    gradients_y_ll, gradients_y_rr = get_surface_node_vars(gradients_interfaces[2], equations, dg, i, interface)
    gradients_ll = (gradients_x_ll, gradients_y_ll)
    gradients_rr = (gradients_x_rr, gradients_y_rr)
    flux = surface_flux(u_ll, u_rr, gradients_ll, gradients_rr, orientation, equations)

    # Copy flux to left and right element storage
    set_node_vars!(destination, flux, equations, dg, i)
  end

  return nothing
end
