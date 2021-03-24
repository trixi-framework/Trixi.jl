# everything related to a DG semidiscretization in 1D,
# currently limited to Lobatto-Legendre nodes

function compute_coefficients!(u, func, t, mesh::StructuredMesh{1}, equations, dg::DG, cache)
  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for i in eachnode(dg)
      x_node = get_node_coords(element.node_coordinates, equations, dg, i)
      u_node = func(x_node, t, equations)

      # Allocation-free version of u[:, i, element] = u_node
      set_node_vars!(u, u_node, equations, dg, i, element_ind)
    end
  end
end


function rhs!(du::AbstractArray{<:Any,3}, u, t,
              mesh::StructuredMesh{1}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache)

  # Calculate interface and boundary fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(u, mesh, equations, dg, cache)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

  return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,3}, mesh::StructuredMesh{1}, 
                              equations, dg::DG, cache)
  @unpack surface_flux = dg

  @threaded for element in eachelement(dg, cache)
    # Left interface
    interface = cache.elements[element].interfaces[1]

    u_ll = get_node_vars(u, equations, dg, nnodes(dg), interface.left_element)
    u_rr = get_node_vars(u, equations, dg, 1,          interface.right_element)

    interface.surface_flux_values .= surface_flux(u_ll, u_rr, interface.orientation, equations)
  end

  return nothing
end


function calc_surface_integral!(du::AbstractArray{<:Any,3}, mesh::StructuredMesh, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis

  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for v in eachvariable(equations)
      # surface at -x
      du[v, 1,          element_ind] -= element.interfaces[1].surface_flux_values[v] * boundary_interpolation[1,          1]
      # surface at +x
      du[v, nnodes(dg), element_ind] += element.interfaces[2].surface_flux_values[v] * boundary_interpolation[nnodes(dg), 2]
    end
  end

  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,3}, u, t, source_terms::Nothing, mesh::StructuredMesh, equations, dg::DG, cache)
  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,3}, u, t, source_terms, mesh::StructuredMesh, equations, dg::DG, cache)

  @threaded for element_ind in eachelement(dg, cache)
    element = cache.elements[element_ind]

    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element_ind)
      x_local = get_node_coords(element.node_coordinates, equations, dg, i)
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, element_ind)
    end
  end

  return nothing
end
