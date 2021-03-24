function rhs!(du::AbstractArray{<:Any,5}, u, t,
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


function calc_volume_integral!(du::AbstractArray{<:Any,5}, u, mesh::StructuredMesh, equations,
                               volume_integral::VolumeIntegralWeakForm, dg::DGSEM, cache)
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


function calc_interface_flux!(u::AbstractArray{<:Any,5}, mesh::StructuredMesh{3},
                              equations, dg::DG, cache)
  @unpack elements = cache

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2, 3)"
    Base.Cartesian.@nexprs 3 orientation->calc_interface_flux!(elements.surface_flux_values, 
                                                               elements.left_neighbors[orientation, element], 
                                                               element, orientation, u, mesh, equations, dg)
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element, orientation, u, 
                              mesh::StructuredMesh{3}, equations, dg::DG)
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

    flux = transformed_surface_flux(u_ll, u_rr, orientation, surface_flux, mesh, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, j, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, j, left_direction, right_element] = flux[v]
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


@inline function transformed_flux(u, orientation, mesh::StructuredMesh{3}, equations)
  @unpack coordinates_min, coordinates_max = mesh

  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)
  dy = (coordinates_max[2] - coordinates_min[2]) / size(mesh, 2)
  dz = (coordinates_max[3] - coordinates_min[3]) / size(mesh, 3)

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

  @unpack coordinates_min, coordinates_max = mesh

  dx = (coordinates_max[1] - coordinates_min[1]) / size(mesh, 1)
  dy = (coordinates_max[2] - coordinates_min[2]) / size(mesh, 2)
  dz = (coordinates_max[3] - coordinates_min[3]) / size(mesh, 3)

  if orientation == 1
    factor = 0.25 * dy * dz
  elseif orientation == 2
    factor = 0.25 * dx * dz
  else # orientation == 3
    factor = 0.25 * dx * dy
  end

  return factor * surface_flux(u_ll, u_rr, orientation, equations)
end
