# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::UnstructuredQuadMesh, equations::AbstractEquations,
                      dg::DG, RealT, uEltype)

  polydeg_ = polydeg(dg.basis)
  nvars = nvariables(equations)

  if polydeg_ > mesh.poly_deg
    error("polynomial degree of DG must be less than or equal to mesh polynomial degree")
  end

  elements = init_elements(RealT, uEltype, mesh, dg.basis.nodes, nvars, polydeg_)

  interfaces = init_interfaces(uEltype, mesh, nvars, polydeg_)

  # TODO: if the mesh is periodic we probabyl can avoid creating this "empty" boundary container
  if isperiodic(mesh)
    boundaries = UnstructuredBoundaryContainer2D{RealT, uEltype, nvars, polydeg_}(0)
  else
    boundaries = init_boundaries(RealT, uEltype, mesh, elements, nvars, polydeg_)
  end

  cache = (; elements, interfaces, boundaries)

  return cache
end


# Note! The mesh is appended to the end of some functions here such that we dispatch on the correct
#       version and use the ::UnstructuredQuadMesh variable type below to keep track on it
function rhs!(du::AbstractArray{<:Any,4}, u, t,
              mesh::UnstructuredQuadMesh, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timeit_debug timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timeit_debug timer() "volume integral" calc_volume_integral!(du, u, have_nonconservative_terms(equations), equations,
                                                                dg.volume_integral, dg, cache, mesh)

  # Prolong solution to interfaces
  @timeit_debug timer() "prolong2interfaces" prolong2interfaces!(cache, u, equations, dg, mesh)

  # Calculate interface fluxes
  @timeit_debug timer() "interface flux" calc_interface_flux!(cache.elements.surface_flux_values,
                                                              have_nonconservative_terms(equations), equations,
                                                              dg, cache, mesh)

  # Prolong solution to boundaries
  @timeit_debug timer() "prolong2boundaries" prolong2boundaries!(cache, u, equations, dg, mesh)

  # Calculate boundary fluxes
  #  TODO: remove initial condition as an input argument here, only needed for hacky BCs
  @timeit_debug timer() "boundary flux" calc_boundary_flux!(cache, t, boundary_conditions, equations, dg, initial_condition, mesh)

  # Calculate surface integrals
  @timeit_debug timer() "surface integral" calc_surface_integral!(du, equations, dg, cache, mesh)

  # Apply Jacobian from mapping to reference element
  #  Note! this routine is reused from dg_curved/dg_2d.jl
  @timeit_debug timer() "Jacobian" apply_jacobian!(du, mesh, equations, dg, cache)

  # Calculate source terms
  @timeit_debug timer() "source terms" calc_sources!(du, u, t, source_terms, equations, dg, cache)

  return nothing
end


# compute volume contribution of the DG approximation with the divergence of the contravariant fluxes
function calc_volume_integral!(du::AbstractArray{<:Any,4}, u,
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache, ::UnstructuredQuadMesh)
  @unpack X_xi, X_eta, Y_xi, Y_eta = cache.elements
  @unpack derivative_dhat = dg.basis

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      # compute the physical fluxes in each Cartesian direction
      x_flux = flux(u_node, 1, equations)
      y_flux = flux(u_node, 2, equations)

      # compute the contravariant flux in the x-direction
      flux1  = Y_eta[i, j, element] * x_flux - X_eta[i, j, element] * y_flux
      for ii in eachnode(dg)
        integral_contribution = derivative_dhat[ii, i] * flux1
        add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
      end

      # compute the contravariant flux in the y-direction
      flux2  = -Y_xi[i, j, element] * x_flux + X_xi[i, j, element] * y_flux
      for jj in eachnode(dg)
        integral_contribution = derivative_dhat[jj, j] * flux2
        add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


# prolong the solution into the convenience array in the interior interface container
# Note! this routine is for quadrilateral elements with "right-handed" orientation
function prolong2interfaces!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG, ::UnstructuredQuadMesh)
  @unpack interfaces = cache

  @threaded for interface in eachinterface(dg, cache)
    primary_element   = interfaces.element_ids[1, interface]
    secondary_element = interfaces.element_ids[2, interface]

    primary_side   = interfaces.element_side_ids[1, interface]
    secondary_side = interfaces.element_side_ids[2, interface]

    if primary_side == 1
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i, 1, primary_element]
      end
    elseif primary_side == 2
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, nnodes(dg), i, primary_element]
      end
    elseif primary_side == 3
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), primary_element]
      end
    else # primary_side == 4
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[1, v, i, interface] = u[v, 1, i, primary_element]
      end
    end

    if secondary_side == 1
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, i, 1, secondary_element]
      end
    elseif secondary_side == 2
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, nnodes(dg), i, secondary_element]
      end
    elseif secondary_side == 3
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, i, nnodes(dg), secondary_element]
      end
    else # secondary_side == 4
      for i in eachnode(dg), v in eachvariable(equations)
        interfaces.u[2, v, i, interface] = u[v, 1, i, secondary_element]
      end
    end
  end

  return nothing
end


# compute the numerical flux interface coupling between two elements on an unstructured quadrilateral mesh
function calc_interface_flux!(surface_flux_values::AbstractArray{<:Any,4},
                              nonconservative_terms::Val{false}, equations, dg::DG, cache, ::UnstructuredQuadMesh)
  @unpack surface_flux = dg
  @unpack u, start_index, inc_index, element_ids, element_side_ids = cache.interfaces
  @unpack normals, tangents, scaling = cache.elements


  @threaded for interface in eachinterface(dg, cache)
    # Get neighboring elements
    primary_element   = element_ids[1, interface]
    secondary_element = element_ids[2, interface]

    # Get the local side id on which to compute the flux
    primary_side   = element_side_ids[1, interface]
    secondary_side = element_side_ids[2, interface]

    # initial index for the coordinate system on the secondary element
    secondary_index = start_index[interface]

    # loop through the primary element coordinate system and compute the interface coupling
    for primary_index in eachnode(dg)
      # pull the primary and secondary states from the boundary u values
      u_ll = u[1, :, primary_index, interface]
      u_rr = u[2, :, secondary_index, interface]

      # pull the directional vectors and scaling factors
      #   Note! this assumes a conforming approximation, more must be done in terms of the normals
      #         and tangents for hanging nodes and other non-conforming approximation spaces
      # TODO: remove the allocation that occurs for normal/tangent_vector either with a different
      #       storage layout or adding accessors to extract an SVector
      normal_vector  = normals[:, primary_index, primary_side, primary_element]
      tangent_vector = tangents[:, primary_index, primary_side, primary_element]
      scaling_ll     = scaling[primary_index, primary_side, primary_element]
      scaling_rr     = scaling[secondary_index, secondary_side, secondary_element]

      # rotate states
      u_tilde_ll = rotate_solution(u_ll, normal_vector, tangent_vector, equations)
      u_tilde_rr = rotate_solution(u_rr, normal_vector, tangent_vector, equations)

      # Call pointwise Riemann solver in the rotated direction
      flux_tilde = surface_flux(u_tilde_ll, u_tilde_rr, 1, equations)

      # backrotate the flux into the original direction
      flux = backrotate_flux(flux_tilde, normal_vector, tangent_vector, equations)

      # Scale the flux appropriately and copy back to primary/secondary element storage
      # Note the sign change for the normal flux in the secondary element!
      for v in eachvariable(equations)
        surface_flux_values[v, primary_index  , primary_side  , primary_element  ] =  flux[v] * scaling_ll
        surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -flux[v] * scaling_rr
      end

      # increment the index of the coordinate system in the secondary element
      secondary_index += inc_index[interface]
    end
  end

  return nothing
end


# move the approximate solution onto physical boundaries within a "right-handed" element
function prolong2boundaries!(cache, u::AbstractArray{<:Any,4}, equations, dg::DG, ::UnstructuredQuadMesh)
  @unpack boundaries = cache

  @threaded for boundary in eachboundary(boundaries)
    element = boundaries.element_id[boundary]
    side    = boundaries.element_side_id[boundary]

    if side == 1
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, l, 1, element]
      end
    elseif side == 2
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
      end
    elseif side == 3
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
      end
    else # side == 4
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[1, v, l, boundary] = u[v, 1, l, element]
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             equations::AbstractEquations{2}, dg::DG, initial_condition, mesh::UnstructuredQuadMesh)
  @assert isempty(eachboundary(cache.boundaries))
end


# TODO: intial_condition argument for convenience, cleanup later with better boundar condition handling
function calc_boundary_flux!(cache, t, boundary_condition, equations, dg::DG, initial_condition, mesh::UnstructuredQuadMesh)

  @unpack surface_flux = dg
  @unpack normals, tangents, scaling, surface_flux_values = cache.elements
  @unpack u, element_id, element_side_id, node_coordinates, name  = cache.boundaries

  @threaded for boundary in eachboundary(cache.boundaries)
    # Get the element and side IDs on the primary element
    primary_element = element_id[boundary]
    primary_side    = element_side_id[boundary]

    for i in eachnode(dg)
      # hacky way to set "exact solution" boundary conditions. Only used to test the orientation
      # for a mesh with flipped elements
      u_ext = initial_condition((node_coordinates[1, i, boundary],
                                 node_coordinates[2, i, boundary]),
                                 t, equations)

      # pull the left state from the boundary u values on the primary element as well as the
      # directional vectors and scaling
      #   Note! this assumes a conforming approximation, more must be done in terms of the normals
      #         and tangents for hanging nodes and other non-conforming approximation spaces
      u_ll  = u[1, :, i, boundary]
      n_vec = normals[ :, i, primary_side, primary_element]
      t_vec = tangents[:, i, primary_side, primary_element]
      scal  = scaling[i, primary_side, primary_element]

      # rotate states
      u_tilde_ll  = rotate_solution(u_ll , n_vec, t_vec, equations)
      u_tilde_ext = rotate_solution(u_ext, n_vec, t_vec, equations)

      # Call pointwise Riemann solver in the rotated direction
      flux_tilde = surface_flux(u_tilde_ll, u_tilde_ext, 1, equations)

      # backrotate the flux into the original direction
      flux = backrotate_flux(flux_tilde, n_vec, t_vec, equations)

      # Scale the flux appropriately and copy back to primary element storage
      for v in eachvariable(equations)
        surface_flux_values[v, i, primary_side, primary_element] = flux[v] * scal
      end
    end
  end

  return nothing
end

# Note! The local side numbering for the unstructured quadrilateral element implementation differs
#       from the structured TreeMesh or CurvedMesh local side numbering:
#
#      TreeMesh/CurvedMesh sides   versus   UnstructuredMesh sides
#                  4                                  3
#          -----------------                  -----------------
#          |               |                  |               |
#          | ^ eta         |                  | ^ eta         |
#        1 | |             | 2              4 | |             | 2
#          | |             |                  | |             |
#          | ---> xi       |                  | ---> xi       |
#          -----------------                  -----------------
#                  3                                  1
# Therefore, we require a different surface integral routine here despite their similar structure.
function calc_surface_integral!(du::AbstractArray{<:Any,4}, equations, dg::DGSEM, cache, ::UnstructuredQuadMesh)
  @unpack boundary_interpolation = dg.basis
  @unpack surface_flux_values = cache.elements

  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg), v in eachvariable(equations)
      # surface contribution along local sides 2 and 4 (fixed x and y varies)
      du[v, 1,          l, element] += ( surface_flux_values[v, l, 4, element]
                                          * boundary_interpolation[1, 1] )
      du[v, nnodes(dg), l, element] += ( surface_flux_values[v, l, 2, element]
                                          * boundary_interpolation[nnodes(dg), 2] )
      # surface contribution along local sides 1 and 3 (fixed y and x varies)
      du[v, l, 1,          element] += ( surface_flux_values[v, l, 1, element]
                                          * boundary_interpolation[1, 1] )
      du[v, l, nnodes(dg), element] += ( surface_flux_values[v, l, 3, element]
                                          * boundary_interpolation[nnodes(dg), 2] )
    end
  end

  return nothing
end
