function rhs!(du::AbstractArray{<:Any,5}, u, t,
    mesh::CurvedMesh, equations,
    initial_condition, boundary_conditions, source_terms,
    dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, mesh, equations, 
                                                                dg.volume_integral, dg, cache)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(u, mesh, equations, dg, cache)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du::AbstractArray{<:Any,5}, u, mesh::CurvedMesh, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  @unpack metric_terms = cache.elements

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)

      flux1 = metric_terms[2, 2, i, j, k, element] * metric_terms[3, 3, i, j, k, element] * flux(u_node, 1, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, k, element)
      end

      flux2 = metric_terms[1, 1, i, j, k, element] * metric_terms[3, 3, i, j, k, element] * flux(u_node, 2,equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, k, element)
      end

      flux3 = metric_terms[1, 1, i, j, k, element] * metric_terms[2, 2, i, j, k, element] * flux(u_node, 3,equations)
      for kk in eachnode(dg)
        integral_contribution = derivative_dhat[kk, k] * flux3
        add_to_node_vars!(du, integral_contribution, equations, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,5}, mesh::CurvedMesh{3},
                              equations, dg::DG, cache)
  @unpack elements = cache

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2, 3)"

    # Interfaces in x-direction (`orientation` = 1)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, mesh, equations, dg, cache)
    
    # Interfaces in x-direction (`orientation` = 2)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, mesh, equations, dg, cache)

    # Interfaces in x-direction (`orientation` = 3)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[3, element],
                         element, 3, u, mesh, equations, dg, cache)
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element, orientation, u, 
                              mesh::CurvedMesh{3}, equations, dg::DG, cache)
  @unpack surface_flux = dg

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for j in eachnode(dg), i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, j, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, j, right_element)
    elseif orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), j, left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          j, right_element)
    else # orientation == 3
      u_ll = get_node_vars(u, equations, dg, i, j, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, j, 1,          right_element)
    end

    flux = transformed_surface_flux(u_ll, u_rr, orientation, surface_flux, mesh, equations, cache)

    for v in eachvariable(equations)
      surface_flux_values[v, i, j, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, j, left_direction, right_element] = flux[v]
    end
  end

  return nothing
end


function apply_jacobian!(du::AbstractArray{<:Any,5}, mesh::CurvedMesh, equations, dg::DG, cache)

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


function transformed_surface_flux(u_ll, u_rr, orientation, surface_flux, 
    mesh::CurvedMesh{3}, equations::AbstractEquations, cache)
  @unpack metric_terms = cache.elements
  if orientation == 1
    factor = metric_terms[2, 2, 1, 1, 1, 1] * metric_terms[3, 3, 1, 1, 1, 1]
  elseif orientation == 2
    factor = metric_terms[1, 1, 1, 1, 1, 1] * metric_terms[3, 3, 1, 1, 1, 1]
  else # orientation == 3
    factor = metric_terms[1, 1, 1, 1, 1, 1] * metric_terms[2, 2, 1, 1, 1, 1]
  end

  return factor * surface_flux(u_ll, u_rr, orientation, equations)
end
