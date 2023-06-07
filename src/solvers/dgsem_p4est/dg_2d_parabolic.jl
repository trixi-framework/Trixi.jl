# This method is called when a SemidiscretizationHyperbolicParabolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_parabolic(mesh::P4estMesh, equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, parabolic_scheme, RealT, uEltype)

  balance!(mesh)

  elements   = init_elements(mesh, equations_hyperbolic, dg.basis, uEltype)
  interfaces = init_interfaces(mesh, equations_hyperbolic, dg.basis, elements)
  boundaries = init_boundaries(mesh, equations_hyperbolic, dg.basis, elements)
 
  n_vars = nvariables(equations_hyperbolic)
  n_elements = nelements(elements)
  n_nodes = nnodes(dg.basis) # nodes in one direction
  u_transformed = Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_elements)
  gradients = ntuple(_ -> similar(u_transformed), ndims(mesh))
  flux_viscous = ntuple(_ -> similar(u_transformed), ndims(mesh))

  cache = (; elements, interfaces, boundaries, gradients, flux_viscous, u_transformed)

  return cache
end

function calc_gradient!(gradients, u_transformed, t,
                        mesh::P4estMesh{2}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG, 
                        cache, cache_parabolic)

  gradients_x, gradients_y = gradients

  # Reset du
  @trixi_timeit timer() "reset gradients" begin
    reset_du!(gradients_x, dg, cache)
    reset_du!(gradients_y, dg, cache)
  end

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" begin
    (; derivative_dhat) = dg.basis
    (; contravariant_vectors) = cache.elements
    
    @threaded for element in eachelement(dg, cache)

      # Calculate gradients with respect to reference coordinates in one element
      for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, element)

        for ii in eachnode(dg)
          multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i], u_node, equations_parabolic, dg, ii, j, element)
        end

        for jj in eachnode(dg)
          multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j], u_node, equations_parabolic, dg, i, jj, element)
        end
      end

      # now that the reference coordinate gradients are computed, transform them node-by-node to physical gradients
      # using the contravariant vectors
      for j in eachnode(dg), i in eachnode(dg)
        Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
        Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)

        gradients_reference_1 = get_node_vars(gradients_x, equations_parabolic, dg, i, j, element)
        gradients_reference_2 = get_node_vars(gradients_y, equations_parabolic, dg, i, j, element)

        # note that the contravariant vectors are transposed compared with computations of flux
        # divergences in `calc_volume_integral!`. See 
        # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190 
        # for a more detailed discussion. 
        gradient_x_node = Ja11 * gradients_reference_1 + Ja21 * gradients_reference_2
        gradient_y_node = Ja12 * gradients_reference_1 + Ja22 * gradients_reference_2

        set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg, i, j, element)
        set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg, i, j, element)
      end

    end
  end

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache_parabolic, u_transformed, mesh, equations_parabolic, dg.surface_integral, dg)

  # Calculate interface fluxes for the gradient. This reuses P4est `calc_interface_flux!` along with a 
  # specialization for AbstractEquationsParabolic.
  @trixi_timeit timer() "interface flux" calc_interface_flux!(cache_parabolic.elements.surface_flux_values, 
                                                              mesh, False(), # False() = no nonconservative terms
                                                              equations_parabolic, dg.surface_integral, dg, 
                                                              cache_parabolic)

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache_parabolic, u_transformed, mesh, equations_parabolic, dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux_gradients!(
    cache_parabolic, t, boundary_conditions_parabolic, mesh, equations_parabolic,
    dg.surface_integral, dg)

  # TODO: parabolic; mortars
  @assert nmortars(dg, cache) == 0

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" begin
    (; boundary_interpolation) = dg.basis
    (; surface_flux_values) = cache_parabolic.elements
    (; contravariant_vectors) = cache.elements

    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1,          1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]
    @threaded for element in eachelement(dg, cache)
      for l in eachnode(dg)
        for v in eachvariable(equations_parabolic)

          # Compute x-component of gradients

          # surface at -x
          normal_direction_x, _ = get_normal_direction(1, contravariant_vectors, 1, l, element)
          gradients_x[v, 1,          l, element] = (
            gradients_x[v, 1,          l, element] + surface_flux_values[v, l, 1, element] * factor_1 * normal_direction_x)

          # surface at +x
          normal_direction_x, _ = get_normal_direction(2, contravariant_vectors, nnodes(dg), l, element)
          gradients_x[v, nnodes(dg), l, element] = (
            gradients_x[v, nnodes(dg), l, element] + surface_flux_values[v, l, 2, element] * factor_2 * normal_direction_x)

          # surface at -y
          normal_direction_x, _ = get_normal_direction(3, contravariant_vectors, l, 1, element)
          gradients_x[v, l, 1,          element] = (
            gradients_x[v, l, 1,          element] + surface_flux_values[v, l, 3, element] * factor_1 * normal_direction_x)

          # surface at +y
          normal_direction_x, _ = get_normal_direction(4, contravariant_vectors, l, nnodes(dg), element)
          gradients_x[v, l, nnodes(dg), element] = (
            gradients_x[v, l, nnodes(dg), element] + surface_flux_values[v, l, 4, element] * factor_2 * normal_direction_x)

          # Compute y-component of gradients

          # surface at -x
          _, normal_direction_y = get_normal_direction(1, contravariant_vectors, 1, l, element)
          gradients_y[v, 1,          l, element] = (
            gradients_y[v, 1,          l, element] + surface_flux_values[v, l, 1, element] * factor_1 * normal_direction_y)

          # surface at +x
          _, normal_direction_y = get_normal_direction(2, contravariant_vectors, nnodes(dg), l, element)
          gradients_y[v, nnodes(dg), l, element] = (
            gradients_y[v, nnodes(dg), l, element] + surface_flux_values[v, l, 2, element] * factor_2 * normal_direction_y)

          # surface at -y
          _, normal_direction_y = get_normal_direction(3, contravariant_vectors, l, 1, element)
          gradients_y[v, l, 1,          element] = (
            gradients_y[v, l, 1,          element] + surface_flux_values[v, l, 3, element] * factor_1 * normal_direction_y)

          # surface at +y
          _, normal_direction_y = get_normal_direction(4, contravariant_vectors, l, nnodes(dg), element)
          gradients_y[v, l, nnodes(dg), element] = (
            gradients_y[v, l, nnodes(dg), element] + surface_flux_values[v, l, 4, element] * factor_2 * normal_direction_y)
        end
      end
    end
  end

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" begin
    apply_jacobian_parabolic!(gradients_x, mesh, equations_parabolic, dg, cache_parabolic)
    apply_jacobian_parabolic!(gradients_y, mesh, equations_parabolic, dg, cache_parabolic)
  end

  return nothing
end

# This version is used for parabolic gradient computations
@inline function calc_interface_flux!(surface_flux_values, mesh::P4estMesh{2},
                                      nonconservative_terms::False, 
                                      equations::AbstractEquationsParabolic,
                                      surface_integral, dg::DG, cache,
                                      interface_index, normal_direction,
                                      primary_node_index, primary_direction_index, primary_element_index,
                                      secondary_node_index, secondary_direction_index, secondary_element_index)
  @unpack u = cache.interfaces
  @unpack surface_flux = surface_integral

  u_ll, u_rr = get_surface_node_vars(u, equations, dg, primary_node_index, interface_index)

  flux_ = 0.5 * (u_ll + u_rr) # we assume that the gradient computations utilize a central flux

  # Note that we don't flip the sign on the secondondary flux. This is because for parabolic terms,
  # the normals are not embedded in `flux_` for the parabolic gradient computations. 
  for v in eachvariable(equations)
    surface_flux_values[v, primary_node_index, primary_direction_index, primary_element_index] = flux_[v]
    surface_flux_values[v, secondary_node_index, secondary_direction_index, secondary_element_index] = flux_[v]
  end
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_volume_integral!(du, flux_viscous,
                               mesh::P4estMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
  (; derivative_dhat) = dg.basis
  (; contravariant_vectors) = cache.elements
  flux_viscous_x, flux_viscous_y = flux_viscous

  @threaded for element in eachelement(dg, cache)
    # Calculate volume terms in one element
    for j in eachnode(dg), i in eachnode(dg)
      flux1 = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, element)
      flux2 = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, element)
  
      # Compute the contravariant flux by taking the scalar product of the
      # first contravariant vector Ja^1 and the flux vector
      Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
      for ii in eachnode(dg)
        multiply_add_to_node_vars!(du, derivative_dhat[ii, i], contravariant_flux1, equations_parabolic, dg, ii, j, element)
      end
  
      # Compute the contravariant flux by taking the scalar product of the
      # second contravariant vector Ja^2 and the flux vector
      Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
      for jj in eachnode(dg)
        multiply_add_to_node_vars!(du, derivative_dhat[jj, j], contravariant_flux2, equations_parabolic, dg, i, jj, element)
      end
    end
  end

  return nothing
end


# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache_parabolic, flux_viscous,
                             mesh::P4estMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
  (; interfaces) = cache_parabolic
  (; contravariant_vectors) = cache_parabolic.elements 
  index_range = eachnode(dg)
  flux_viscous_x, flux_viscous_y = flux_viscous

  @threaded for interface in eachinterface(dg, cache)
    # Copy solution data from the primary element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    # Note that in the current implementation, the interface will be
    # "aligned at the primary element", i.e., the index of the primary side
    # will always run forwards.
    primary_element = interfaces.neighbor_ids[1, interface]
    primary_indices = interfaces.node_indices[1, interface]
    primary_direction = indices2direction(primary_indices)

    i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1], index_range)
    j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2], index_range)

    i_primary = i_primary_start
    j_primary = j_primary_start
    for i in eachnode(dg)

      # this is the outward normal direction on the primary element
      normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                                              i_primary, j_primary, primary_element)

      for v in eachvariable(equations_parabolic)
        # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
        flux_viscous = SVector(flux_viscous_x[v, i_primary, j_primary, primary_element], 
                               flux_viscous_y[v, i_primary, j_primary, primary_element])

        interfaces.u[1, v, i, interface] = dot(flux_viscous, normal_direction)
      end
      i_primary += i_primary_step
      j_primary += j_primary_step
    end

    # Copy solution data from the secondary element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    secondary_element = interfaces.neighbor_ids[2, interface]
    secondary_indices = interfaces.node_indices[2, interface]
    secondary_direction = indices2direction(secondary_indices)

    i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1], index_range)
    j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2], index_range)

    i_secondary = i_secondary_start
    j_secondary = j_secondary_start
    for i in eachnode(dg)
      # This is the outward normal direction on the secondary element.
      # Here, we assume that normal_direction on the secondary element is 
      # the negative of normal_direction on the primary element.  
      normal_direction = get_normal_direction(secondary_direction, contravariant_vectors,
                                              i_secondary, j_secondary, secondary_element)

      for v in eachvariable(equations_parabolic)
        # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
        flux_viscous = SVector(flux_viscous_x[v, i_secondary, j_secondary, secondary_element], 
                               flux_viscous_y[v, i_secondary, j_secondary, secondary_element])
        # store the normal flux with respect to the primary normal direction
        interfaces.u[2, v, i, interface] = -dot(flux_viscous, normal_direction)
      end
      i_secondary += i_secondary_step
      j_secondary += j_secondary_step
    end
  end

  return nothing
end

function calc_interface_flux!(surface_flux_values,
                              mesh::P4estMesh{2}, equations_parabolic,
                              dg::DG, cache_parabolic)

  (; neighbor_ids, node_indices) = cache_parabolic.interfaces
  (; contravariant_vectors) = cache_parabolic.elements
  index_range = eachnode(dg)
  index_end = last(index_range)

  @threaded for interface in eachinterface(dg, cache_parabolic)
    # Get element and side index information on the primary element
    primary_element = neighbor_ids[1, interface]
    primary_indices = node_indices[1, interface]
    primary_direction_index = indices2direction(primary_indices)

    # Create the local i,j indexing on the primary element used to pull normal direction information
    i_primary_start, i_primary_step = index_to_start_step_2d(primary_indices[1], index_range)
    j_primary_start, j_primary_step = index_to_start_step_2d(primary_indices[2], index_range)

    i_primary = i_primary_start
    j_primary = j_primary_start

    # Get element and side index information on the secondary element
    secondary_element = neighbor_ids[2, interface]
    secondary_indices = node_indices[2, interface]
    secondary_direction_index = indices2direction(secondary_indices)

    # Initiate the secondary index to be used in the surface for loop.
    # This index on the primary side will always run forward but
    # the secondary index might need to run backwards for flipped sides.
    if :i_backward in secondary_indices
      node_secondary = index_end
      node_secondary_step = -1
    else
      node_secondary = 1
      node_secondary_step = 1
    end

    for node in eachnode(dg)
      # We prolong the viscous flux dotted with respect the outward normal on the 
      # primary element. We assume a BR-1 type of flux.
      viscous_flux_normal_ll, viscous_flux_normal_rr = 
        get_surface_node_vars(cache_parabolic.interfaces.u, equations_parabolic, dg, node, interface)

      flux = 0.5 * (viscous_flux_normal_ll + viscous_flux_normal_rr)

      for v in eachvariable(equations_parabolic)
        surface_flux_values[v, node, primary_direction_index, primary_element] = flux[v]
        surface_flux_values[v, node_secondary, secondary_direction_index, secondary_element] = -flux[v]
      end

      # Increment primary element indices to pull the normal direction
      i_primary += i_primary_step
      j_primary += j_primary_step
      # Increment the surface node index along the secondary element
      node_secondary += node_secondary_step
    end
  end

  return nothing
end

# TODO: parabolic, finish implementing `calc_boundary_flux_gradients!` and `calc_boundary_flux_divergence!`
function prolong2boundaries!(cache_parabolic, flux_viscous,
                             mesh::P4estMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
  (; boundaries) = cache_parabolic
  (; contravariant_vectors) = cache_parabolic.elements
  index_range = eachnode(dg)

  flux_viscous_x, flux_viscous_y = flux_viscous

  @threaded for boundary in eachboundary(dg, cache_parabolic)
    # Copy solution data from the element using "delayed indexing" with
    # a start value and a step size to get the correct face and orientation.
    element       = boundaries.neighbor_ids[boundary]
    node_indices  = boundaries.node_indices[boundary]

    i_node_start, i_node_step = index_to_start_step_2d(node_indices[1], index_range)
    j_node_start, j_node_step = index_to_start_step_2d(node_indices[2], index_range)

    i_node = i_node_start
    j_node = j_node_start
    for i in eachnode(dg)
      # this is the outward normal direction on the primary element
      normal_direction = get_normal_direction(primary_direction, contravariant_vectors,
                        i_node, j_node, primary_element)

      for v in eachvariable(equations_parabolic)
        flux_viscous = SVector(flux_viscous_x[v, i_primary, j_primary, primary_element], 
            flux_viscous_y[v, i_primary, j_primary, primary_element])

        boundaries.u[v, i, boundary] = dot(flux_viscous, normal_direction)
      end
      i_node += i_node_step
      j_node += j_node_step
    end
  end

  return nothing
end
