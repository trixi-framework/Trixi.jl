function rhs!(du, u, t,
              mesh::CurvedMesh{1}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Calculate interface and boundary fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(
    cache, u, mesh, equations, dg)

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(
    cache, u, t, boundary_conditions, mesh, equations, dg)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(
    du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_interface_flux!(cache, u, mesh::CurvedMesh{1},
                              equations, dg::DG)
  @unpack surface_flux = dg

  @threaded for element in eachelement(dg, cache)
    left_element = cache.elements.left_neighbors[1, element]

    if left_element > 0 # left_element = 0 at bounaries
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          element)

      flux = surface_flux(u_ll, u_rr, 1, equations)

      for v in eachvariable(equations)
        cache.elements.surface_flux_values[v, 2, left_element] = flux[v]
        cache.elements.surface_flux_values[v, 1, element] = flux[v]
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::CurvedMesh{1}, equations, dg::DG)
  @assert isperiodic(mesh)
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             mesh::CurvedMesh{1}, equations, dg::DG)
  calc_boundary_flux!(cache, u, t, (boundary_condition, boundary_condition),
                      mesh, equations, dg)
end


function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             mesh::CurvedMesh{1}, equations, dg::DG)
  @unpack surface_flux = dg
  @unpack surface_flux_values, node_coordinates = cache.elements

  orientation = 1

  # Negative x-direction
  direction = 1

  u_rr = get_node_vars(u, equations, dg, 1, 1)
  x = get_node_coords(node_coordinates, equations, dg, 1, 1)

  flux = boundary_conditions[direction](u_rr, orientation, direction, x, t, surface_flux, equations)

  for v in eachvariable(equations)
    surface_flux_values[v, direction, 1] = flux[v]
  end

  # Positive x-direction
  direction = 2

  u_rr = get_node_vars(u, equations, dg, nnodes(dg), nelements(dg, cache))
  x = get_node_coords(node_coordinates, equations, dg, nnodes(dg), nelements(dg, cache))

  flux = boundary_conditions[direction](u_rr, orientation, direction, x, t, surface_flux, equations)

  # Copy flux to left and right element storage
  for v in eachvariable(equations)
    surface_flux_values[v, direction, nelements(dg, cache)] = flux[v]
  end
end
