# everything related to a DG semidiscretization in 1D,
# currently limited to Lobatto-Legendre nodes

function compute_coefficients!(u, func, t, mesh::StructuredMesh{RealT, 1}, equations, dg::DG, cache) where {RealT}
  @threaded for element_x in 1:mesh.size[1]
    element = cache.elements[element_x]

    for i in eachnode(dg)
      x_node = element.node_coordinates[i]
      u_node = func(x_node, t, equations)

      # Allocation-free version of u[:, i, element] = u_node
      set_node_vars!(u, u_node, equations, dg, i, element_x)
    end
  end
end


function rhs!(du::AbstractArray{<:Any,3}, u, t,
              mesh::StructuredMesh{RealT, 1}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache) where {RealT}
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache)

  # Prolong solution to interfaces and boundaries
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, u, mesh, equations, dg)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, boundary_conditions, equations, dg)

  # Calculate interface and boundary fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(have_nonconservative_terms(equations), mesh, equations,
                                                              dg, cache)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, mesh, equations, dg, cache)

  return nothing
end


function prolong2interfaces!(cache, u::AbstractArray{<:Any,3}, mesh::StructuredMesh, equations, dg::DG)
  for element_x in 1:mesh.size[1]
    element = cache.elements[element_x]

    element.interfaces[1].u_right .= u[:, 1, element_x]
    element.interfaces[2].u_left .= u[:, end, element_x]
  end

  return nothing
end


function prolong2boundaries!(cache, u::AbstractArray{<:Any,3}, 
    boundary_condition::BoundaryConditionPeriodic, equations, dg::DG)
  cache.elements[1].interfaces[1].u_left .= u[:, end, end]
  cache.elements[end].interfaces[2].u_right .= u[:, 1, 1]

  return nothing
end


function calc_interface_flux!(nonconservative_terms::Val{false}, mesh::StructuredMesh{<:Real, 1}, equations,
                              dg::DG, cache)
  @unpack surface_flux = dg

  for element_x in 1:mesh.size[1]
    # Left interface
    interface = cache.elements[element_x].interfaces[1]

    interface.surface_flux_values .= surface_flux(interface.u_left, interface.u_right, interface.orientation, equations)
  end

  interface = cache.elements[end].interfaces[2]
  interface.surface_flux_values .= surface_flux(interface.u_left, interface.u_right, interface.orientation, equations)

  return nothing
end


function calc_surface_integral!(du::AbstractArray{<:Any,3}, mesh::StructuredMesh, equations, dg::DGSEM, cache)
  @unpack boundary_interpolation = dg.basis

  @threaded for element_x in 1:mesh.size[1]
    element = cache.elements[element_x]

    for v in eachvariable(equations)
      # surface at -x
      du[v, 1,          element_x] -= element.interfaces[1].surface_flux_values[v] * boundary_interpolation[1,          1]
      # surface at +x
      du[v, nnodes(dg), element_x] += element.interfaces[2].surface_flux_values[v] * boundary_interpolation[nnodes(dg), 2]
    end
  end

  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,3}, u, t, source_terms::Nothing, mesh::StructuredMesh, equations, dg::DG, cache)
  return nothing
end


function calc_sources!(du::AbstractArray{<:Any,3}, u, t, source_terms, mesh::StructuredMesh, equations, dg::DG, cache)

  @threaded for element_x in 1:mesh.size[1]
    element = cache.elements[element_x]

    for i in eachnode(dg)
      u_local = get_node_vars(u, equations, dg, i, element_x)
      x_local = element.node_coordinates[i]
      du_local = source_terms(u_local, x_local, t, equations)
      add_to_node_vars!(du, du_local, equations, dg, i, element_x)
    end
  end

  return nothing
end