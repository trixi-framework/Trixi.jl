function rhs!(du::AbstractArray{<:Any,4}, u, t,
    mesh::StructuredMesh, equations,
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


function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, mesh::StructuredMesh, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  @unpack metric_terms = cache.elements

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      flux1 = metric_terms[2, 2, i, j, element] * flux(u_node, 1, equations) - 
              metric_terms[1, 2, i, j, element] * flux(u_node, 2, equations)
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      flux2 = -metric_terms[2, 1, i, j, element] * flux(u_node, 1, equations) + 
              metric_terms[1, 1, i, j, element] * flux(u_node, 2, equations)
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,4}, mesh::StructuredMesh{2},
                              equations, dg::DG, cache)
  @unpack elements = cache

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2)"

    # Interfaces in x-direction (`orientation` = 1)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, equations, dg, cache)
    
    # Interfaces in y-direction (`orientation` = 2)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, equations, dg, cache)
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element, 
                                      orientation, u, equations, dg::DG, cache)
  @unpack surface_flux = dg
  @unpack metric_terms = cache.elements

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

      normal_vector = SVector(metric_terms[2, 2, 1, i, right_element], -metric_terms[1, 2, 1, i, right_element])
    else # orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

      normal_vector = SVector(-metric_terms[2, 1, i, 1, right_element], metric_terms[1, 1, i, 1, right_element])
    end

    flux = surface_flux(u_ll, u_rr, normal_vector, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, left_direction, right_element] = flux[v]
    end
  end
end


function apply_jacobian!(du::AbstractArray{<:Any,4}, mesh::StructuredMesh, equations, dg::DG, cache)
  @unpack inverse_jacobian = cache.elements

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, element] *= -inverse_jacobian[i, j, element]
      end
    end
  end

  return nothing
end
