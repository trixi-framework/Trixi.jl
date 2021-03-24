function compute_coefficients!(u, func, t, mesh::StructuredMesh{3}, equations, dg::DG, cache)
  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for i in eachnode(dg), j in eachnode(dg), k in eachnode(dg)
      x_node = get_node_coords(element.node_coordinates, equations, dg, i, j, k)
      u_node = func(x_node, t, equations)

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
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, mesh,
                                                                equations, dg.volume_integral, dg, cache)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(u, mesh,
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
                               mesh::StructuredMesh, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)

      flux1 = transformed_flux(u_node, 1, mesh, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, k, element)
      end

      flux2 = transformed_flux(u_node, 2, mesh, equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, k, element)
      end

      flux3 = transformed_flux(u_node, 3, mesh, equations)
      for kk in eachnode(dg)
        integral_contribution = derivative_dhat[kk, k] * flux3
        add_to_node_vars!(du, integral_contribution, equations, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,5}, mesh::StructuredMesh{3}, equations,
                              dg::DG, cache)
  @unpack surface_flux = dg
  @unpack size = mesh

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    for orientation in (1, 3, 5)
      interface = cache.elements[element].interfaces[orientation]
      calc_interface_flux!(interface, u, mesh, equations, dg)
    end
  end

  return nothing
end


function calc_interface_flux!(interface, u, mesh::StructuredMesh{3}, equations, dg::DG)
  @unpack surface_flux = dg

  for j in eachnode(dg), i in eachnode(dg)
    if interface.orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, j, interface.left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, j, interface.right_element)
    elseif interface.orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), j, interface.left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          j, interface.right_element)
    else # interface.orientation == 3
      u_ll = get_node_vars(u, equations, dg, i, j, nnodes(dg), interface.left_element)
      u_rr = get_node_vars(u, equations, dg, i, j, 1,          interface.right_element)
    end

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

  @threaded for element in eachelement(dg, cache)
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
      x_local = get_node_coords(element.node_coordinates, equations, dg, i, j, k)
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, j, k, element_ind)
    end
  end

  return nothing
end


@inline function transformed_flux(u, orientation, mesh::StructuredMesh{3}, equations)
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

  return factor * flux(u, orientation, equations)
end


function transformed_surface_flux(u_ll, u_rr, orientation, surface_flux, 
    mesh::StructuredMesh{3}, equations::AbstractEquations)

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
