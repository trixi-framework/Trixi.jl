# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# TODO: Unify this function with the `ParallelTreeMesh` `rhs!` once mortar support has been added
# to the `ParallelP4estMesh`
function rhs!(du, u, t,
              mesh::ParallelP4estMesh{2}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Start to receive MPI data
  @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

  # Prolong solution to MPI interfaces
  @trixi_timeit timer() "prolong2mpiinterfaces" prolong2mpiinterfaces!(
    cache, u, mesh, equations, dg.surface_integral, dg)

  # Start to send MPI data
  @trixi_timeit timer() "start MPI send" start_mpi_send!(
    cache.mpi_cache, mesh, equations, dg, cache)

  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

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

  # Prolong solution to mortars
  @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
    cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.mortar, dg.surface_integral, dg, cache)

  # Finish to receive MPI data
  @trixi_timeit timer() "finish MPI receive" finish_mpi_receive!(
    cache.mpi_cache, mesh, equations, dg, cache)

  # Calculate MPI interface fluxes
  @trixi_timeit timer() "MPI interface flux" calc_mpi_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg, cache)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  # Finish to send MPI data
  @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

  return nothing
end


function prolong2mpiinterfaces!(cache, u,
                                mesh::ParallelP4estMesh{2},
                                equations, surface_integral, dg::DG)
  @unpack mpi_interfaces = cache
  index_range = eachnode(dg)

  @threaded for interface in eachmpiinterface(dg, cache)
    # Copy solution data from the local element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the index of the primary side
    # will always run forwards.
    local_side = mpi_interfaces.local_sides[interface]
    local_element = mpi_interfaces.local_element_ids[interface]
    local_indices = mpi_interfaces.node_indices[interface]

    i_element_start, i_element_step = index_to_start_step_2d(local_indices[1], index_range)
    j_element_start, j_element_step = index_to_start_step_2d(local_indices[2], index_range)

    i_element = i_element_start
    j_element = j_element_start
    for i in eachnode(dg)
      for v in eachvariable(equations)
        mpi_interfaces.u[local_side, v, i, interface] = u[v, i_element, j_element, local_element]
      end
      i_element += i_element_step
      j_element += j_element_step
    end
  end

  return nothing
end


function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::ParallelP4estMesh{2},
                                  nonconservative_terms,
                                  equations, surface_integral, dg::DG, cache)
  @unpack local_element_ids, node_indices, local_sides = cache.mpi_interfaces
  @unpack contravariant_vectors = cache.elements
  index_range = eachnode(dg)
  index_end = last(index_range)

  @threaded for interface in eachmpiinterface(dg, cache)
    # Get element and side index information on the local element
    local_element = local_element_ids[interface]
    local_indices = node_indices[interface]
    local_direction = indices2direction(local_indices)
    local_side = local_sides[interface]

    # Create the local i,j indexing on the local element used to pull normal direction information
    i_element_start, i_element_step = index_to_start_step_2d(local_indices[1], index_range)
    j_element_start, j_element_step = index_to_start_step_2d(local_indices[2], index_range)

    i_element = i_element_start
    j_element = j_element_start

    # Initiate the node index to be used in the surface for loop,
    # the surface flux storage must be indexed in alignment with the local element indexing
    if :i_backward in local_indices
      surface_node = index_end
      surface_node_step = -1
    else
      surface_node = 1
      surface_node_step = 1
    end

    for node in eachnode(dg)
      # Get the normal direction on the local element
      # Contravariant vectors at interfaces in negative coordinate direction
      # are pointing inwards. This is handled by `get_normal_direction`.
      normal_direction = get_normal_direction(local_direction, contravariant_vectors,
                                              i_element, j_element, local_element)

      calc_mpi_interface_flux!(surface_flux_values, mesh, nonconservative_terms, equations,
                               surface_integral, dg, cache,
                               interface, normal_direction,
                               node, local_side,
                               surface_node, local_direction, local_element)

      # Increment local element indices to pull the normal direction
      i_element += i_element_step
      j_element += j_element_step

      # Increment the surface node index along the local element
      surface_node += surface_node_step
    end
  end

  return nothing
end

# Inlined version of the interface flux computation for conservation laws
@inline function calc_mpi_interface_flux!(surface_flux_values,
                                          mesh::P4estMesh{2},
                                          nonconservative_terms::Val{false}, equations,
                                          surface_integral, dg::DG, cache,
                                          interface_index, normal_direction,
                                          interface_node_index, local_side,
                                          surface_node_index, local_direction_index, local_element_index)
  @unpack u = cache.mpi_interfaces
  @unpack surface_flux = surface_integral

  u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface_node_index, interface_index)

  if local_side == 1
    flux_ = surface_flux(u_ll, u_rr, normal_direction, equations)
  else # local_side == 2
    flux_ = -surface_flux(u_ll, u_rr, -normal_direction, equations)
  end

  for v in eachvariable(equations)
    surface_flux_values[v, surface_node_index, local_direction_index, local_element_index] = flux_[v]
  end
end



end # muladd