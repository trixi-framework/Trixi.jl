function compute_coefficients!(u, func, t, mesh::StructuredMesh{<:Real, 3}, equations, dg::DG, cache)
  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for i in eachnode(dg), j in eachnode(dg), k in eachnode(dg)
      coords_node = element.node_coordinates[i, j, k]
      u_node = func(coords_node, t, equations)

      # Allocation-free version of u[:, i, j, k, element] = u_node
      set_node_vars!(u, u_node, equations, dg, i, j, k, element_ind)
    end
  end
end


function rhs!(du::AbstractArray{<:Any,5}, u, t,
    mesh::StructuredMesh, equations,
    initial_condition, boundary_conditions, source_terms,
    dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, have_nonconservative_terms(equations), mesh,
                                                                equations, dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, u, mesh, equations, dg)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, boundary_conditions, mesh, equations, dg)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(have_nonconservative_terms(equations), mesh,
                                                              equations, dg, cache)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,5}, u,
                               nonconservative_terms::Val{false}, mesh::StructuredMesh, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)

      flux1 = transformed_calcflux(u_node, 1, mesh, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, k, element)
      end

      flux2 = transformed_calcflux(u_node, 2, mesh, equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, k, element)
      end

      flux3 = transformed_calcflux(u_node, 3, mesh, equations)
      for kk in eachnode(dg)
        integral_contribution = derivative_dhat[kk, k] * flux3
        add_to_node_vars!(du, integral_contribution, equations, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


function prolong2interfaces!(cache, u::AbstractArray{<:Any,5}, mesh::StructuredMesh, equations, dg::DG)
  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    # Interfaces in x-direction
    for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations)
      element.interfaces[1].u_right[v, j, k] = u[v, 1, j, k, element_ind]
      element.interfaces[2].u_left[v, j, k] = u[v, end, j, k, element_ind]
    end

    # Interfaces in y-direction
    for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      element.interfaces[3].u_right[v, i, k] = u[v, i, 1, k, element_ind]
      element.interfaces[4].u_left[v, i, k] = u[v, i, end, k, element_ind]
    end

    # Interfaces in z-direction
    for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations)
      element.interfaces[5].u_right[v, i, j] = u[v, i, j, 1, element_ind]
      element.interfaces[6].u_left[v, i, j] = u[v, i, j, end, element_ind]
    end
  end

  return nothing
end


function prolong2boundaries!(cache, u::AbstractArray{<:Any,5}, 
    boundary_condition::BoundaryConditionPeriodic, mesh::StructuredMesh, equations, dg::DG)
  @unpack size, linear_indices = mesh

  # Boundaries in x-direction
  for element_y in 1:size[2], element_z in 1:size[3]
    # TODO A loop would be more efficient
    cache.elements[1, element_y, element_z].interfaces[1].u_left .= u[:, end, :, :, linear_indices[end, element_y, element_z]]
    cache.elements[end, element_y, element_z].interfaces[2].u_right .= u[:, 1, :, :, linear_indices[1, element_y, element_z]]
  end

  # Boundaries in y-direction
  for element_x in 1:size[1], element_z in 1:size[3]
    cache.elements[element_x, 1, element_z].interfaces[3].u_left .= u[:, :, end, :, linear_indices[element_x, end, element_z]]
    cache.elements[element_x, end, element_z].interfaces[4].u_right .= u[:, :, 1, :, linear_indices[element_x, 1, element_z]]
  end

  # Boundaries in z-direction
  for element_x in 1:size[1], element_y in 1:size[2]
    cache.elements[element_x, element_y, 1].interfaces[5].u_left .= u[:, :, :, end, linear_indices[element_x, element_y, end]]
    cache.elements[element_x, element_y, end].interfaces[6].u_right .= u[:, :, :, 1, linear_indices[element_x, element_y, 1]]
  end

  return nothing
end


function calc_interface_flux!(nonconservative_terms::Val{false}, mesh::StructuredMesh{<:Real, 3}, equations,
                              dg::DG, cache)
  @unpack surface_flux = dg
  @unpack size = mesh

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    for orientation in (1, 3, 5)
      interface = cache.elements[element].interfaces[orientation]
      calc_interface_flux!(interface, mesh, equations, dg)
    end
  end

  # Boundary in positive x-direction
  for element_y in 1:size[2], element_z in 1:size[3]
    interface = cache.elements[end, element_y, element_z].interfaces[2]
    calc_interface_flux!(interface, mesh, equations, dg)
  end

  # Boundary in positive y-direction
  for element_x in 1:size[1], element_z in 1:size[3]
    interface = cache.elements[element_x, end, element_z].interfaces[4]
    calc_interface_flux!(interface, mesh, equations, dg)
  end

  # Boundary in positive z-direction
  for element_x in 1:size[1], element_y in 1:size[2]
    interface = cache.elements[element_x, element_y, end].interfaces[6]
    calc_interface_flux!(interface, mesh, equations, dg)
  end

  return nothing
end


function calc_interface_flux!(interface::Interface, mesh::StructuredMesh{<:Real, 3}, equations, dg::DG)
  @unpack surface_flux = dg

  for j in eachnode(dg), i in eachnode(dg)
    u_ll = get_node_vars(interface.u_left, equations, dg, i, j)
    u_rr = get_node_vars(interface.u_right, equations, dg, i, j)

    flux = transformed_surface_flux(u_ll, u_rr, interface.orientation, surface_flux, mesh, equations)

    for v in eachvariable(equations)
      interface.surface_flux_values[v, i, j] = flux[v]
    end
  end
end


function calc_surface_integral!(du::AbstractArray{<:Any,5}, mesh::StructuredMesh, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis

  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for m in eachnode(dg), l in eachnode(dg)
      for v in eachvariable(equations)
        # surface at -x
        du[v, 1,          l, m, element_ind] -= element.interfaces[1].surface_flux_values[v, l, m] * boundary_interpolation[1,          1]
        # surface at +x
        du[v, nnodes(dg), l, m, element_ind] += element.interfaces[2].surface_flux_values[v, l, m] * boundary_interpolation[nnodes(dg), 2]
        # surface at -y
        du[v, l, 1,          m, element_ind] -= element.interfaces[3].surface_flux_values[v, l, m] * boundary_interpolation[1,          1]
        # surface at +y
        du[v, l, nnodes(dg), m, element_ind] += element.interfaces[4].surface_flux_values[v, l, m] * boundary_interpolation[nnodes(dg), 2]
        # surface at -z
        du[v, l, m, 1,          element_ind] -= element.interfaces[5].surface_flux_values[v, l, m] * boundary_interpolation[1,          1]
        # surface at +z
        du[v, l, m, nnodes(dg), element_ind] += element.interfaces[6].surface_flux_values[v, l, m] * boundary_interpolation[nnodes(dg), 2]
      end
    end
  end

  return nothing
end


function apply_jacobian!(du::AbstractArray{<:Any,5}, mesh::StructuredMesh, equations, dg::DG, cache)

  for element in eachelement(dg, cache)
    factor = -cache.elements.inverse_jacobian[element]

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, k, element] *= factor
      end
    end
  end

  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,5}, u, t, source_terms::Nothing, mesh::StructuredMesh, equations, dg::DG, cache)
  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,5}, u, t, source_terms, mesh::StructuredMesh, equations, dg::DG, cache)

  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, j, k, element_ind)
      x_local = element.node_coordinates[i, j, k]
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, j, k, element_ind)
    end
  end

  return nothing
end


@inline function transformed_calcflux(u, orientation, mesh::StructuredMesh{<:Real, 3}, equations)
  @unpack size, coordinates_min, coordinates_max = mesh

  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]
  dy = (coordinates_max[2] - coordinates_min[2]) / size[2]
  dz = (coordinates_max[3] - coordinates_min[3]) / size[3]

  if orientation == 1
    factor = 0.25 * dy * dz
  elseif orientation == 2
    factor = 0.25 * dx * dz
  else # orientation == 3
    factor = 0.25 * dx * dy
  end

  return factor * calcflux(u, orientation, equations)
end


function transformed_surface_flux(u_ll, u_rr, orientation, surface_flux, 
    mesh::StructuredMesh{<:Real, 3}, equations::AbstractEquations)

  @unpack size, coordinates_min, coordinates_max = mesh

  dx = (coordinates_max[1] - coordinates_min[1]) / size[1]
  dy = (coordinates_max[2] - coordinates_min[2]) / size[2]
  dz = (coordinates_max[3] - coordinates_min[3]) / size[3]

  if orientation == 1
    factor = 0.25 * dy * dz
  elseif orientation == 2
    factor = 0.25 * dx * dz
  else # orientation == 3
    factor = 0.25 * dx * dy
  end

  return factor * surface_flux(u_ll, u_rr, orientation, equations)
end