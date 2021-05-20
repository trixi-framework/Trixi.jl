# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::UnstructuredQuadMesh, equations,
                      dg::DG, RealT, uEltype)

  polydeg_ = polydeg(dg.basis)
  nvars = nvariables(equations)

  elements = init_elements(RealT, uEltype, mesh, dg.basis.nodes, nvars, polydeg_)

  interfaces = init_interfaces(uEltype, mesh, nvars, polydeg_)

  boundaries = init_boundaries(RealT, uEltype, mesh, elements, nvars, polydeg_)

  cache = (; elements, interfaces, boundaries)

  # perform a check on the sufficient metric identities condition for free-stream preservation
  # and halt computation if it fails
  if !isapprox(max_discrete_metric_identities(dg, cache), 0, atol=1e-12)
    error("metric terms fail free-stream preservation check with maximum error $(max_discrete_metric_identities(dg, cache))")
  end

  return cache
end


function rhs!(du, u, t,
              mesh::UnstructuredQuadMesh, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @timed timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @timed timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @timed timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg)

  # Calculate interface fluxes
  @timed timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg, cache)

  # Prolong solution to boundaries
  @timed timer() "prolong2boundaries" prolong2boundaries!(
    cache, u, mesh, equations, dg)

  # Calculate boundary fluxes
  @timed timer() "boundary flux" calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations, dg)

  # Calculate surface integrals
  @timed timer() "surface integral" calc_surface_integral!(
    du, mesh, equations, dg, cache)

  # Apply Jacobian from mapping to reference element
  #  Note! this routine is reused from dg_curved/dg_2d.jl
  @timed timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @timed timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


# prolong the solution into the convenience array in the interior interface container
# Note! this routine is for quadrilateral elements with "right-handed" orientation
function prolong2interfaces!(cache, u,
                             mesh::UnstructuredQuadMesh,
                             equations, dg::DG)
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


# compute the numerical flux interface coupling between two elements on an unstructured
# quadrilateral mesh
function calc_interface_flux!(surface_flux_values,
                              mesh::UnstructuredQuadMesh,
                              nonconservative_terms::Val{false}, equations, dg::DG, cache)
  @unpack surface_flux = dg
  @unpack u, start_index, index_increment, element_ids, element_side_ids = cache.interfaces
  @unpack normal_directions = cache.elements


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
      u_ll = get_one_sided_surface_node_vars(u, equations, dg, 1, primary_index, interface)
      u_rr = get_one_sided_surface_node_vars(u, equations, dg, 2, secondary_index, interface)

      # pull the outward pointing (normal) directional vector
      #   Note! this assumes a conforming approximation, more must be done in terms of the normals
      #         for hanging nodes and other non-conforming approximation spaces
      outward_direction = get_surface_normal(normal_directions, primary_index, primary_side,
                                             primary_element)

      # Call pointwise numerical flux with rotation. Direction is normalized inside this function
      flux = surface_flux(u_ll, u_rr, outward_direction, equations)

      # Copy flux back to primary/secondary element storage
      # Note the sign change for the normal flux in the secondary element!
      for v in eachvariable(equations)
        surface_flux_values[v, primary_index  , primary_side  , primary_element  ] =  flux[v]
        surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -flux[v]
      end

      # increment the index of the coordinate system in the secondary element
      secondary_index += index_increment[interface]
    end
  end

  return nothing
end


# move the approximate solution onto physical boundaries within a "right-handed" element
function prolong2boundaries!(cache, u,
                             mesh::UnstructuredQuadMesh,
                             equations, dg::DG)
  @unpack boundaries = cache

  @threaded for boundary in eachboundary(dg, cache)
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
                             mesh::UnstructuredQuadMesh, equations, dg::DG)
  @assert isempty(eachboundary(dg, cache))
end


function calc_boundary_flux!(cache, t, boundary_conditions::Tuple,
                             mesh::UnstructuredQuadMesh, equations, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack element_id, element_side_id = cache.boundaries

  # Loop through each boundary type and associated index vector from the tuples
  # made in `digest_boundary_conditions`
  @threaded for (boundary_condition, boundary_condition_indices) in boundary_conditions[1]
    for boundary in boundary_condition_indices
        #Get the element and side IDs on the boundary element
        element = element_id[boundary]
        side    = element_side_id[boundary]

        #calc boundary flux on the current boundary interface
        calc_boundary_flux!(surface_flux_values, t, boundary_condition, mesh, equations, dg, cache,
                            side, element, boundary)
    end
  end

  return nothing
end

# use a function barrier for now to improve type stability
@noinline function calc_boundary_flux!(surface_flux_values, t, boundary_condition::BC,
                                       mesh::UnstructuredQuadMesh, equations, dg::DG, cache,
                                       side, element, boundary) where {BC}
  for node in eachnode(dg)
    calc_boundary_flux!(surface_flux_values, t, boundary_condition, mesh, equations, dg, cache,
                        node, side, element, boundary)
  end
end

# inlined version of the boundary flux calculation along a physical interface where the
# boundary flux values are set according to a particular `boundary_condition` function
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::UnstructuredQuadMesh, equations, dg::DG, cache,
                                     node_index, side_index, element_index, boundary_index)
  @unpack normal_directions = cache.elements
  @unpack u, node_coordinates = cache.boundaries
  @unpack surface_flux = dg

  # pull the inner solution state from the boundary u values on the boundary element
  u_inner = get_one_sided_surface_node_vars(u, equations, dg, 1, node_index, boundary_index)

  # pull the outward pointing (normal) directional vector
  outward_direction = get_surface_normal(normal_directions, node_index, side_index, element_index)

  # get the external solution values from the prescribed external state
  x = get_node_coords(node_coordinates, equations, dg, node_index, boundary_index)

  # Call pointwise numerical flux function in the rotated direction on the boundary
  #    Note! the direction is normalized inside this function
  flux = boundary_condition(u_inner, outward_direction, x, t, surface_flux, equations)

  for v in eachvariable(equations)
    surface_flux_values[v, node_index, side_index, element_index] = flux[v]
  end
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
function calc_surface_integral!(du, mesh::UnstructuredQuadMesh,
                                equations, dg::DGSEM, cache)
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


# This routine computes the maximum value of the discrete metric identities necessary to ensure
# that the approxmiation will be free-stream preserving (i.e. a constant solution remains constant)
# on a curvilinear mesh.
#   Note! Independent of the equation system and is only a check on the discrete mapping terms.
#         Can be used for a metric identities check on CurvedMesh{2} or UnstructuredQuadMesh
function max_discrete_metric_identities(dg::DGSEM, cache)
  @unpack derivative_matrix = dg.basis
  @unpack contravariant_vectors = cache.elements

  ndims_ = size(contravariant_vectors, 1)

  metric_id_dx = zeros(eltype(contravariant_vectors), nnodes(dg), nnodes(dg))
  metric_id_dy = zeros(eltype(contravariant_vectors), nnodes(dg), nnodes(dg))

  max_metric_ids = zero(dg.basis.nodes[1])

  for i in 1:ndims_, element in eachelement(dg, cache)
    # compute D*Ja_1^i + Ja_2^i*D^T
    @views mul!(metric_id_dx, derivative_matrix, contravariant_vectors[i, 1, :, :, element])
    @views mul!(metric_id_dy, contravariant_vectors[i, 2, :, :, element], derivative_matrix')
    local_max_metric_ids = maximum( abs.(metric_id_dx + metric_id_dy) )

    max_metric_ids = max( max_metric_ids, local_max_metric_ids )
  end

  return max_metric_ids
end
