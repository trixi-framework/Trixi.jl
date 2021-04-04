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

  # Calculate boundary fluxes
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, u, t, boundary_conditions, equations, mesh, dg)

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

    if left_element > 0 # left_element = -1 at bounaries
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
                             equations::AbstractEquations{1}, mesh::CurvedMesh{1}, dg::DG)
  @assert mesh.periodicity
end


# calc_boundary_flux! with boundary_condition::BoundaryConditionPeriodic is in dg.jl
function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             equations::AbstractEquations{1}, mesh::CurvedMesh{1}, dg::DG)
  @unpack surface_flux_values, node_coordinates = cache.elements

  orientation = 1

  # Negative x-direction
  direction = 1

  for j in eachnode(dg)
    calc_boundary_flux_at_node!(surface_flux_values, node_coordinates, u, t, boundary_conditions, 
                                equations, dg, orientation, direction, 1, 1)
  end

  # Positive x-direction
  direction = 2

  for j in eachnode(dg)
    calc_boundary_flux_at_node!(surface_flux_values, node_coordinates, u, t, boundary_conditions, 
                                equations, dg, orientation, direction, nelements(dg, cache), nnodes(dg))
  end
end
