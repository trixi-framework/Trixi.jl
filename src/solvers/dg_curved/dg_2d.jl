function rhs!(du::AbstractArray{<:Any,4}, u, t,
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

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, u, t, boundary_conditions, equations, mesh, dg)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


# Return the contravariant vector in the specified `orientation` at the specified nodes and element,
# multiplied by the Jacobian of the transformation mapping.
function scaled_contravariant_vector(orientation, i, j, element, cache)
  @unpack metric_terms = cache.elements

  if orientation == 1
    return SVector(metric_terms[2, 2, i, j, element], -metric_terms[1, 2, i, j, element])
  end
  
  # orientation == 2
  return SVector(-metric_terms[2, 1, i, j, element], metric_terms[1, 1, i, j, element])
end


function calc_volume_integral!(du::AbstractArray{<:Any,4}, u, mesh::CurvedMesh, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      # Scalar product of the flux vector with the first contravariant vector, 
      # multiplied with the Jacobian
      v1 = scaled_contravariant_vector(1, i, j, element, cache)
      flux1 = v1[1] * flux(u_node, 1, equations) + v1[2] * flux(u_node, 2, equations)

      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      # Scalar product of the flux vector with the second contravariant vector, 
      # multiplied with the Jacobian
      v2 = scaled_contravariant_vector(2, i, j, element, cache)
      flux2 = v2[1] * flux(u_node, 1, equations) + v2[2] * flux(u_node, 2, equations)

      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,4}, mesh::CurvedMesh{2},
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
  if left_element <= 0 # left_element = -1 at boundaries
    return surface_flux_values
  end

  @unpack surface_flux = dg

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, right_element)

      normal_vector = scaled_contravariant_vector(1, 1, i, right_element, cache)
    else # orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          right_element)

      normal_vector = scaled_contravariant_vector(2, i, 1, right_element, cache)
    end

    flux = surface_flux(u_ll, u_rr, normal_vector, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, left_direction, right_element] = flux[v]
    end
  end

  return surface_flux_values
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             equations::AbstractEquations{2}, mesh::CurvedMesh{2}, dg::DG)
  @assert mesh.periodicity
end


function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             equations::AbstractEquations{2}, mesh::CurvedMesh{2}, dg::DG)
  @unpack surface_flux = dg
  @unpack surface_flux_values, node_coordinates = cache.elements
  linear_indices = LinearIndices(size(mesh))
  
  for cell_y in axes(mesh, 2)
    # Negative x-direction
    direction = 1

    element = linear_indices[begin, cell_y]

    for j in eachnode(dg)
      u_rr = get_node_vars(u, equations, dg, 1, j, element)
      x = get_node_coords(node_coordinates, equations, dg, 1, j, element)

      normal_vector = scaled_contravariant_vector(1, 1, j, element, cache)
      flux = boundary_conditions[direction](u_rr, normal_vector, direction, x, t, surface_flux, equations)

      for v in eachvariable(equations)
        surface_flux_values[v, j, direction, element] = flux[v]
      end
    end

    # Positive x-direction
    direction = 2

    element = linear_indices[end, cell_y]

    for j in eachnode(dg)
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), j, element)
      x = get_node_coords(node_coordinates, equations, dg, nnodes(dg), j, element)

      normal_vector = scaled_contravariant_vector(1, nnodes(dg), j, element, cache)
      flux = boundary_conditions[direction](u_ll, normal_vector, direction, x, t, surface_flux, equations)

      for v in eachvariable(equations)
        surface_flux_values[v, j, direction, element] = flux[v]
      end
    end
  end

  for cell_x in axes(mesh, 1)
    # Negative y-direction
    direction = 3

    element = linear_indices[cell_x, begin]

    for i in eachnode(dg)
      u_rr = get_node_vars(u, equations, dg, i, 1, element)
      x = get_node_coords(node_coordinates, equations, dg, i, 1, element)

      normal_vector = scaled_contravariant_vector(2, i, 1, element, cache)
      flux = boundary_conditions[direction](u_rr, normal_vector, direction, x, t, surface_flux, equations)
                                            
      for v in eachvariable(equations)
        surface_flux_values[v, i, direction, element] = flux[v]
      end
    end

    # Positive y-direction
    direction = 4

    element = linear_indices[cell_x, end]

    for i in eachnode(dg)
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), element)
      x = get_node_coords(node_coordinates, equations, dg, i, nnodes(dg), element)

      normal_vector = scaled_contravariant_vector(2, i, nnodes(dg), element, cache)
      flux = boundary_conditions[direction](u_ll, normal_vector, direction, x, t, surface_flux, equations)

      for v in eachvariable(equations)
        surface_flux_values[v, i, direction, element] = flux[v]
      end
    end
  end
end


function apply_jacobian!(du::AbstractArray{<:Any,4}, mesh::CurvedMesh, equations, dg::DG, cache)
  @unpack inverse_jacobian = cache.elements

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      factor = -inverse_jacobian[i, j, element]

      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end
