# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache(mesh::UnstructuredQuadMesh, equations,
                      dg::DG, RealT, uEltype)

  polydeg_ = polydeg(dg.basis)
  nvars = nvariables(equations)

  elements = init_elements(RealT, uEltype, mesh, get_nodes(dg.basis), nvars, polydeg_)

  interfaces = init_interfaces(uEltype, mesh, nvars, polydeg_)

  boundaries = init_boundaries(RealT, uEltype, mesh, elements, nvars, polydeg_)

  cache = (; elements, interfaces, boundaries)

  # perform a check on the sufficient metric identities condition for free-stream preservation
  # and halt computation if it fails
  if !isapprox(max_discrete_metric_identities(dg, cache), 0, atol=1e-12)
    error("metric terms fail free-stream preservation check with maximum error $(max_discrete_metric_identities(dg, cache))")
  end

  # Add specialized parts of the cache required to compute the flux differencing volume integral
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

  return cache
end


function rhs!(du, u, t,
              mesh::UnstructuredQuadMesh, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg, cache)

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, t, boundary_conditions, mesh, equations, dg.surface_integral, dg)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  #  Note! this routine is reused from dg_curved/dg_2d.jl
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


# Calculate 2D twopoint contravariant flux (element version)
@inline function calcflux_twopoint!(ftilde1, ftilde2, u::AbstractArray{<:Any,4}, element,
                                    mesh::Union{CurvedMesh{2}, UnstructuredQuadMesh},
                                    equations, volume_flux, dg::DGSEM, cache)
  @unpack contravariant_vectors = cache.elements

  for j in eachnode(dg), i in eachnode(dg)
    # pull the solution value and two contravariant vectors at node i,j
    u_node = get_node_vars(u, equations, dg, i, j, element)
    Ja11_node, Ja12_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    Ja21_node, Ja22_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    # diagonal (consistent) part not needed since diagonal of
    # dg.basis.derivative_split_transpose is zero!
    set_node_vars!(ftilde1, zero(u_node), equations, dg, i, i, j)
    set_node_vars!(ftilde2, zero(u_node), equations, dg, j, i, j)

    # contravariant fluxes in the first direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      flux2 = volume_flux(u_node, u_node_ii, 2, equations)
      # pull the contravariant vectors and compute their average
      Ja11_node_ii, Ja12_node_ii = get_contravariant_vector(1, contravariant_vectors, ii, j, element)
      Ja11_avg = 0.5 * (Ja11_node + Ja11_node_ii)
      Ja12_avg = 0.5 * (Ja12_node + Ja12_node_ii)
      # compute the contravariant sharp flux
      fluxtilde1 = Ja11_avg * flux1 + Ja12_avg * flux2
      # save and exploit symmetry
      set_node_vars!(ftilde1, fluxtilde1, equations, dg, i, ii, j)
      set_node_vars!(ftilde1, fluxtilde1, equations, dg, ii, i, j)
    end

    # contravariant fluxes in the second direction
    for jj in (j+1):nnodes(dg)
      u_node_jj  = get_node_vars(u, equations, dg, i, jj, element)
      flux1 = volume_flux(u_node, u_node_jj, 1, equations)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      # pull the contravariant vectors and compute their average
      Ja21_node_jj, Ja22_node_jj = get_contravariant_vector(2, contravariant_vectors, i, jj, element)
      Ja21_avg = 0.5 * (Ja21_node + Ja21_node_jj)
      Ja22_avg = 0.5 * (Ja22_node + Ja22_node_jj)
      # compute the contravariant sharp flux
      fluxtilde2 = Ja21_avg * flux1 + Ja22_avg * flux2
      # save and exploit symmetry
      set_node_vars!(ftilde2, fluxtilde2, equations, dg, j,  i, jj)
      set_node_vars!(ftilde2, fluxtilde2, equations, dg, jj, i, j)
    end
  end

  calcflux_twopoint_nonconservative!(ftilde1, ftilde2, u, element,
                                     have_nonconservative_terms(equations),
                                     mesh, equations, dg, cache)
end


function calcflux_twopoint_nonconservative!(f1, f2, u::AbstractArray{<:Any,4}, element,
                                            nonconservative_terms::Val{true},
                                            mesh::Union{CurvedMesh{2}, UnstructuredQuadMesh},
                                            equations, dg::DG, cache)
  #TODO: Create a unified interface, e.g. using non-symmetric two-point (extended) volume fluxes
  #      For now, just dispatch to an existing function for the IdealMhdEquations
  @unpack contravariant_vectors = cache.elements
  calcflux_twopoint_nonconservative!(f1, f2, u, element, contravariant_vectors, equations, dg, cache)
end


@inline function split_form_kernel!(du::AbstractArray{<:Any,4}, u,
                                    nonconservative_terms::Val{false}, element,
                                    mesh::Union{CurvedMesh{2}, UnstructuredQuadMesh}, equations,
                                    volume_flux, dg::DGSEM, cache, alpha=true)
  @unpack derivative_split = dg.basis
  @unpack contravariant_vectors = cache.elements

  # Calculate volume integral in one element
  for j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, element)

    # compute the fluxes in the x and y directions
    flux1 = flux(u_node, 1, equations)
    flux2 = flux(u_node, 2, equations)

    # first direction: use consistency of the volume flux to make this evaluation cheaper
    # pull the contravariant vector
    Ja11_node, Ja12_node = get_contravariant_vector(1, contravariant_vectors, i, j, element)
    # compute the two contravariant fluxes
    fluxtilde1 = Ja11_node * flux1 + Ja12_node * flux2
    integral_contribution = alpha * derivative_split[i, i] * fluxtilde1
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)

    # second direction: use consistency of the volume flux to make this evaluation cheaper
    # pull the contravariant vector
    Ja21_node, Ja22_node = get_contravariant_vector(2, contravariant_vectors, i, j, element)
    fluxtilde2 = Ja21_node * flux1 + Ja22_node * flux2
    integral_contribution = alpha * derivative_split[j, j] * fluxtilde2
    add_to_node_vars!(du, integral_contribution, equations, dg, i, j, element)

    # use symmetry of the volume flux for the remaining terms in the first direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      flux2 = volume_flux(u_node, u_node_ii, 2, equations)
      # pull the contravariant vectors and compute the average
      Ja11_node_ii, Ja12_node_ii = get_contravariant_vector(1, contravariant_vectors, ii, j, element)
      Ja11_avg = 0.5 * (Ja11_node + Ja11_node_ii)
      Ja12_avg = 0.5 * (Ja12_node + Ja12_node_ii)
      # compute the contravariant sharp flux
      fluxtilde1 = Ja11_avg * flux1 + Ja12_avg * flux2
      integral_contribution = alpha * derivative_split[i, ii] * fluxtilde1
      add_to_node_vars!(du, integral_contribution, equations, dg, i,  j, element)
      integral_contribution = alpha * derivative_split[ii, i] * fluxtilde1
      add_to_node_vars!(du, integral_contribution, equations, dg, ii, j, element)
    end

    # use symmetry of the volume flux for the remaining terms in the second direction
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, element)
      flux1 = volume_flux(u_node, u_node_jj, 1, equations)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      # pull the contravariant vectors and compute the average
      Ja21_node_jj, Ja22_node_jj = get_contravariant_vector(2, contravariant_vectors, i, jj, element)
      Ja21_avg = 0.5 * (Ja21_node + Ja21_node_jj)
      Ja22_avg = 0.5 * (Ja22_node + Ja22_node_jj)
      # compute the contravariant sharp flux
      fluxtilde2 = Ja21_avg * flux1 + Ja22_avg * flux2
      integral_contribution = alpha * derivative_split[j, jj] * fluxtilde2
      add_to_node_vars!(du, integral_contribution, equations, dg, i, j,  element)
      integral_contribution = alpha * derivative_split[jj, j] * fluxtilde2
      add_to_node_vars!(du, integral_contribution, equations, dg, i, jj, element)
    end
  end
end


# prolong the solution into the convenience array in the interior interface container
# Note! this routine is for quadrilateral elements with "right-handed" orientation
function prolong2interfaces!(cache, u,
                             mesh::UnstructuredQuadMesh,
                             equations, surface_integral, dg::DG)
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
                              nonconservative_terms::Val{false}, equations,
                              surface_integral, dg::DG, cache)
  @unpack surface_flux = surface_integral
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


# compute the numerical flux interface with nonconservative terms coupling between two elements
# on an unstructured quadrilateral mesh
function calc_interface_flux!(surface_flux_values,
                              mesh::UnstructuredQuadMesh,
                              nonconservative_terms::Val{true}, equations,
                              surface_integral, dg::DG, cache)
  @unpack u, start_index, index_increment, element_ids, element_side_ids = cache.interfaces
  @unpack normal_directions = cache.elements

  fstar_primary_threaded             = cache.fstar_upper_threaded
  fstar_secondary_threaded           = cache.fstar_lower_threaded
  noncons_diamond_primary_threaded   = cache.noncons_diamond_upper_threaded
  noncons_diamond_secondary_threaded = cache.noncons_diamond_lower_threaded

  @threaded for interface in eachinterface(dg, cache)
    # Choose thread-specific pre-allocated container
    fstar_primary             = fstar_primary_threaded[Threads.threadid()]
    fstar_secondary           = fstar_secondary_threaded[Threads.threadid()]
    noncons_diamond_primary   = noncons_diamond_primary_threaded[Threads.threadid()]
    noncons_diamond_secondary = noncons_diamond_secondary_threaded[Threads.threadid()]

    # Get the primary element index and local side index
    primary_element = element_ids[1, interface]
    primary_side = element_side_ids[1, interface]

    # Get initial index for the coordinate system on the secondary element and its
    # index increment
    secondary_index = start_index[interface]
    secondary_index_increment = index_increment[interface]

    # Calculate the conservative portion of the numerical flux
    calc_fstar!(fstar_primary, fstar_secondary, equations, surface_integral, dg,
                u, normal_directions, interface,
                primary_element, primary_side, secondary_index, secondary_index_increment)

    for primary_index in eachnode(dg)
      # Pull the primary and secondary states from the boundary u values
      u_ll = get_one_sided_surface_node_vars(u, equations, dg, 1, primary_index, interface)
      u_rr = get_one_sided_surface_node_vars(u, equations, dg, 2, secondary_index, interface)
      # Pull the outward pointing (normal) directional vector
      #   Note! this assumes a conforming approximation, more must be done in terms of the normals
      #         for hanging nodes and other non-conforming approximation spaces
      outward_direction = get_surface_normal(normal_directions, primary_index, primary_side,
                                             primary_element)
      # Call pointwise nonconservative term
      noncons_primary   = noncons_interface_flux(u_ll, u_rr, outward_direction, :weak, equations)
      noncons_secondary = noncons_interface_flux(u_rr, u_ll, outward_direction, :weak, equations)
      # Save to primary and secondary temporay storage
      set_node_vars!(noncons_diamond_primary,   noncons_primary,   equations, dg, primary_index)
      set_node_vars!(noncons_diamond_secondary, noncons_secondary, equations, dg, secondary_index)
      # increment the index of the coordinate system in the secondary element
      secondary_index += secondary_index_increment
    end

    # Get neighboring element and local side index
    secondary_element = element_ids[2, interface]
    secondary_side = element_side_ids[2, interface]

    # Reinitialize index for the coordinate system on the secondary element
    secondary_index = start_index[interface]
    # loop through the primary element coordinate system and compute the interface coupling
    for primary_index in eachnode(dg)
      # Copy flux back to primary/secondary element storage
      # Note the sign change for the components in the secondary element!
      for v in eachvariable(equations)
        surface_flux_values[v, primary_index, primary_side, primary_element] = (fstar_primary[v, primary_index] +
            noncons_diamond_primary[v, primary_index])
        surface_flux_values[v, secondary_index, secondary_side, secondary_element] = -(fstar_secondary[v, secondary_index] +
            noncons_diamond_secondary[v, secondary_index])
      end
      # increment the index of the coordinate system in the secondary element
      secondary_index += secondary_index_increment
    end
  end

  return nothing
end


@inline function calc_fstar!(destination_primary::AbstractArray{<:Any,2},
                             destination_secondary::AbstractArray{<:Any,2},
                             equations, surface_integral, dg::DGSEM,
                             u_interfaces, normal_directions,
                             interface, primary_element, primary_side,
                             secondary_element_start_index, secondary_element_index_increment)
  @unpack surface_flux = surface_integral

  secondary_index = secondary_element_start_index
  for primary_index in eachnode(dg)
    # pull the primary and secondary states from the boundary u values
    u_ll = get_one_sided_surface_node_vars(u_interfaces, equations, dg, 1, primary_index, interface)
    u_rr = get_one_sided_surface_node_vars(u_interfaces, equations, dg, 2, secondary_index, interface)

    # pull the outward pointing (normal) directional vector
    #   Note! this assumes a conforming approximation, more must be done in terms of the normals
    #         for hanging nodes and other non-conforming approximation spaces
    outward_direction = get_surface_normal(normal_directions, primary_index, primary_side,
                                           primary_element)

    # Call pointwise numerical flux with rotation. Direction is normalized inside this function
    flux = surface_flux(u_ll, u_rr, outward_direction, equations)

    # Copy flux to left and right element storage
    set_node_vars!(destination_primary,   flux, equations, dg, primary_index)
    set_node_vars!(destination_secondary, flux, equations, dg, secondary_index)
    # increment the index of the coordinate system in the secondary element
    secondary_index += secondary_element_index_increment
  end

  return nothing
end


# move the approximate solution onto physical boundaries within a "right-handed" element
function prolong2boundaries!(cache, u,
                             mesh::UnstructuredQuadMesh,
                             equations, surface_integral, dg::DG)
  @unpack boundaries = cache

  @threaded for boundary in eachboundary(dg, cache)
    element = boundaries.element_id[boundary]
    side    = boundaries.element_side_id[boundary]

    if side == 1
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[v, l, boundary] = u[v, l, 1, element]
      end
    elseif side == 2
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[v, l, boundary] = u[v, nnodes(dg), l, element]
      end
    elseif side == 3
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[v, l, boundary] = u[v, l, nnodes(dg), element]
      end
    else # side == 4
      for l in eachnode(dg), v in eachvariable(equations)
        boundaries.u[v, l, boundary] = u[v, 1, l, element]
      end
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::Union{UnstructuredQuadMesh, P4estMesh},
                             equations, surface_integral, dg::DG)
  @assert isempty(eachboundary(dg, cache))
end


# Function barrier for type stability
function calc_boundary_flux!(cache, t, boundary_conditions,
                             mesh::Union{UnstructuredQuadMesh, P4estMesh},
                             equations, surface_integral, dg::DG)
  @unpack boundary_condition_types, boundary_indices = boundary_conditions

  calc_boundary_flux_by_type!(cache, t, boundary_condition_types, boundary_indices,
                              mesh, equations, surface_integral, dg)
  return nothing
end


# Iterate over tuples of boundary condition types and associated indices
# in a type-stable way using "lispy tuple programming".
function calc_boundary_flux_by_type!(cache, t, BCs::NTuple{N,Any},
                                     BC_indices::NTuple{N,Vector{Int}},
                                     mesh::Union{UnstructuredQuadMesh, P4estMesh},
                                     equations, surface_integral, dg::DG) where {N}
  # Extract the boundary condition type and index vector
  boundary_condition = first(BCs)
  boundary_condition_indices = first(BC_indices)
  # Extract the remaining types and indices to be processed later
  remaining_boundary_conditions = Base.tail(BCs)
  remaining_boundary_condition_indices = Base.tail(BC_indices)

  # process the first boundary condition type
  calc_boundary_flux!(cache, t, boundary_condition, boundary_condition_indices,
                      mesh, equations, surface_integral, dg)

  # recursively call this method with the unprocessed boundary types
  calc_boundary_flux_by_type!(cache, t, remaining_boundary_conditions,
                              remaining_boundary_condition_indices,
                              mesh, equations, surface_integral, dg)

  return nothing
end

# terminate the type-stable iteration over tuples
function calc_boundary_flux_by_type!(cache, t, BCs::Tuple{}, BC_indices::Tuple{},
                                     mesh::Union{UnstructuredQuadMesh, P4estMesh},
                                     equations, surface_integral, dg::DG)
  nothing
end


function calc_boundary_flux!(cache, t, boundary_condition, boundary_indexing,
                             mesh::UnstructuredQuadMesh, equations,
                             surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack element_id, element_side_id = cache.boundaries

  @threaded for local_index in eachindex(boundary_indexing)
    # use the local index to get the global boundary index from the pre-sorted list
    boundary = boundary_indexing[local_index]

    # get the element and side IDs on the boundary element
    element = element_id[boundary]
    side    = element_side_id[boundary]

    # calc boundary flux on the current boundary interface
    for node in eachnode(dg)
      calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                          mesh, equations, surface_integral, dg, cache,
                          node, side, element, boundary)
    end
  end
end


# inlined version of the boundary flux calculation along a physical interface where the
# boundary flux values are set according to a particular `boundary_condition` function
@inline function calc_boundary_flux!(surface_flux_values, t, boundary_condition,
                                     mesh::UnstructuredQuadMesh, equations,
                                     surface_integral, dg::DG, cache,
                                     node_index, side_index, element_index, boundary_index)
  @unpack normal_directions = cache.elements
  @unpack u, node_coordinates = cache.boundaries
  @unpack surface_flux = surface_integral

  # pull the inner solution state from the boundary u values on the boundary element
  u_inner = get_node_vars(u, equations, dg, node_index, boundary_index)

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
function calc_surface_integral!(du, u, mesh::UnstructuredQuadMesh,
                                equations, surface_integral, dg::DGSEM, cache)
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

  max_metric_ids = zero(eltype(contravariant_vectors))

  for i in 1:ndims_, element in eachelement(dg, cache)
    # compute D*Ja_1^i + Ja_2^i*D^T
    @views mul!(metric_id_dx, derivative_matrix, contravariant_vectors[i, 1, :, :, element])
    @views mul!(metric_id_dy, contravariant_vectors[i, 2, :, :, element], derivative_matrix')
    local_max_metric_ids = maximum( abs.(metric_id_dx + metric_id_dy) )

    max_metric_ids = max( max_metric_ids, local_max_metric_ids )
  end

  return max_metric_ids
end
