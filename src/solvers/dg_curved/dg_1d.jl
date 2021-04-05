function rhs!(du::AbstractArray{<:Any,3}, u, t,
              mesh::CurvedMesh{1}, equations,
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
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_interface_flux!(u::AbstractArray{<:Any,3}, mesh::CurvedMesh{1},
                              equations, dg::DG, cache)
  @unpack surface_flux = dg

  @threaded for element in eachelement(dg, cache)
    left_element = cache.elements.left_neighbors[1, element]

    u_ll = get_node_vars(u, equations, dg, nnodes(dg), left_element)
    u_rr = get_node_vars(u, equations, dg, 1,          element)

    flux = surface_flux(u_ll, u_rr, 1, equations)

    for v in eachvariable(equations)
      cache.elements.surface_flux_values[v, 2, left_element] = flux[v]
      cache.elements.surface_flux_values[v, 1, element] = flux[v]
    end
  end

  return nothing
end
