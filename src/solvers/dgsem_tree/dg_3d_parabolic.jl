# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

# This file collects all methods that have been updated to work with parabolic systems of equations
#
# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(f(u, grad(u))) (i.e., the "regular" rhs! call)
# boundary conditions will be applied to both grad(u) and div(f(u, grad(u))).
function rhs_parabolic!(du, u, t, mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                        initial_condition, boundary_conditions_parabolic, source_terms,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
  @unpack u_transformed, gradients, flux_viscous = cache_parabolic

  # Convert conservative variables to a form more suitable for viscous flux calculations
  @trixi_timeit timer() "transform variables" transform_variables!(
    u_transformed, u, mesh, equations_parabolic, dg, parabolic_scheme, cache, cache_parabolic)

  # Compute the gradients of the transformed variables
  @trixi_timeit timer() "calculate gradient" calc_gradient!(
    gradients, u_transformed, t, mesh, equations_parabolic, boundary_conditions_parabolic, dg,
    cache, cache_parabolic)

  # Compute and store the viscous fluxes
  @trixi_timeit timer() "calculate viscous fluxes" calc_viscous_fluxes!(
    flux_viscous, gradients, u_transformed, mesh, equations_parabolic, dg, cache, cache_parabolic)

  # The remainder of this function is essentially a regular rhs! for parabolic equations (i.e., it
  # computes the divergence of the viscous fluxes)
  #
  # OBS! In `calc_viscous_fluxes!`, the viscous flux values at the volume nodes of each element have
  # been computed and stored in `fluxes_viscous`. In the following, we *reuse* (abuse) the
  # `interfaces` and `boundaries` containers in `cache_parabolic` to interpolate and store the
  # *fluxes* at the element surfaces, as opposed to interpolating and storing the *solution* (as it
  # is done in the hyperbolic operator). That is, `interfaces.u`/`boundaries.u` store *viscous flux values*
  # and *not the solution*.  The advantage is that a) we do not need to allocate more storage, b) we
  # do not need to recreate the existing data structure only with a different name, and c) we do not
  # need to interpolate solutions *and* gradients to the surfaces.

  # TODO: parabolic; reconsider current data structure reuse strategy

  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, flux_viscous, mesh, equations_parabolic, dg, cache)

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache_parabolic, flux_viscous, mesh, equations_parabolic, dg.surface_integral, dg, cache)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache_parabolic.elements.surface_flux_values, mesh, equations_parabolic, dg, cache_parabolic)

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache_parabolic, flux_viscous, mesh, equations_parabolic, dg.surface_integral, dg, cache)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux_divergence!(
    cache_parabolic, t, boundary_conditions_parabolic, mesh, equations_parabolic,
    dg.surface_integral, dg)

  # TODO: parabolic; extend to mortars
  @assert nmortars(dg, cache) == 0

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations_parabolic, dg.surface_integral, dg, cache_parabolic)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations_parabolic, dg, cache_parabolic)

  return nothing
end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, mesh::TreeMesh{3},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, parabolic_scheme, cache, cache_parabolic)
  @threaded for element in eachelement(dg, cache)
    # Calculate volume terms in one element
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations_parabolic, dg, i, j, k, element)
      u_transformed_node = gradient_variable_transformation(equations_parabolic)(u_node, equations_parabolic)
      set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg, i, j, k, element)
    end
  end
end

# This is the version used when calculating the divergence of the viscous fluxes
function calc_volume_integral!(du, flux_viscous,
                               mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  flux_viscous_x, flux_viscous_y, flux_viscous_z = flux_viscous

  @threaded for element in eachelement(dg, cache)
    # Calculate volume terms in one element
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      flux_1_node = get_node_vars(flux_viscous_x, equations_parabolic, dg, i, j, k, element)
      flux_2_node = get_node_vars(flux_viscous_y, equations_parabolic, dg, i, j, k, element)
      flux_3_node = get_node_vars(flux_viscous_z, equations_parabolic, dg, i, j, k, element)

      for ii in eachnode(dg)
        multiply_add_to_node_vars!(du, derivative_dhat[ii, i], flux_1_node, equations_parabolic, dg, ii, j, k, element)
      end

      for jj in eachnode(dg)
        multiply_add_to_node_vars!(du, derivative_dhat[jj, j], flux_2_node, equations_parabolic, dg, i, jj, k, element)
      end

      for kk in eachnode(dg)
        multiply_add_to_node_vars!(du, derivative_dhat[kk, k], flux_3_node, equations_parabolic, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
function prolong2interfaces!(cache_parabolic, flux_viscous,
                             mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
  @unpack interfaces = cache_parabolic
  @unpack orientations = interfaces

  flux_viscous_x, flux_viscous_y, flux_viscous_z = flux_viscous

  @threaded for interface in eachinterface(dg, cache)
    left_element  = interfaces.neighbor_ids[1, interface]
    right_element = interfaces.neighbor_ids[2, interface]

    if orientations[interface] == 1
      # interface in x-direction
      for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations_parabolic)
        # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
        interfaces.u[1, v, j, k, interface] = flux_viscous_x[v, nnodes(dg), j, k, left_element]
        interfaces.u[2, v, j, k, interface] = flux_viscous_x[v,          1, j, k, right_element]
      end
    elseif orientations[interface] == 2
      # interface in y-direction
      for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations_parabolic)
        # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
        interfaces.u[1, v, i, k, interface] = flux_viscous_y[v, i, nnodes(dg), k, left_element]
        interfaces.u[2, v, i, k, interface] = flux_viscous_y[v, i,          1, k, right_element]
      end
    else # if orientations[interface] == 3
      # interface in z-direction
      for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations_parabolic)
        # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
        interfaces.u[1, v, i, j, interface] = flux_viscous_z[v, i, j, nnodes(dg), left_element]
        interfaces.u[2, v, i, j, interface] = flux_viscous_z[v, i, j,          1, right_element]
      end
    end
  end

  return nothing
end


# This is the version used when calculating the divergence of the viscous fluxes
function calc_interface_flux!(surface_flux_values,
                              mesh::TreeMesh{3}, equations_parabolic,
                              dg::DG, cache_parabolic)
  @unpack neighbor_ids, orientations = cache_parabolic.interfaces

  @threaded for interface in eachinterface(dg, cache_parabolic)
    # Get neighboring elements
    left_id  = neighbor_ids[1, interface]
    right_id = neighbor_ids[2, interface]

    # Determine interface direction with respect to elements:
    # orientation = 1: left -> 2, right -> 1
    # orientation = 2: left -> 4, right -> 3
    # orientation = 3: left -> 6, right -> 5
    left_direction  = 2 * orientations[interface]
    right_direction = 2 * orientations[interface] - 1

    for j in eachnode(dg), i in eachnode(dg)
      # Get precomputed fluxes at interfaces
      flux_ll, flux_rr = get_surface_node_vars(cache_parabolic.interfaces.u, equations_parabolic,
                                               dg, i, j, interface)

      # Compute interface flux as mean of left and right viscous fluxes
      # TODO: parabolic; only BR1 at the moment
      flux = 0.5 * (flux_ll + flux_rr)

      # Copy flux to left and right element storage
      for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, j, left_direction,  left_id]  = flux[v]
        surface_flux_values[v, i, j, right_direction, right_id] = flux[v]
      end
    end
  end

  return nothing
end


# This is the version used when calculating the divergence of the viscous fluxes
function prolong2boundaries!(cache_parabolic, flux_viscous,
                             mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
  @unpack boundaries = cache_parabolic
  @unpack orientations, neighbor_sides = boundaries
  flux_viscous_x, flux_viscous_y, flux_viscous_z = flux_viscous

  @threaded for boundary in eachboundary(dg, cache_parabolic)
    element = boundaries.neighbor_ids[boundary]

    if orientations[boundary] == 1
      # boundary in x-direction
      if neighbor_sides[boundary] == 1
        # element in -x direction of boundary
        for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations_parabolic)
          # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
          boundaries.u[1, v, j, k, boundary] = flux_viscous_x[v, nnodes(dg), j, k, element]
        end
      else # Element in +x direction of boundary
        for k in eachnode(dg), j in eachnode(dg), v in eachvariable(equations_parabolic)
          # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
          boundaries.u[2, v, j, k, boundary] = flux_viscous_x[v, 1,          j, k, element]
        end
      end
    elseif orientations[boundary] == 2
      # boundary in y-direction
      if neighbor_sides[boundary] == 1
        # element in -y direction of boundary
        for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations_parabolic)
          # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
          boundaries.u[1, v, i, k, boundary] = flux_viscous_y[v, i, nnodes(dg), k, element]
        end
      else
        # element in +y direction of boundary
        for k in eachnode(dg), i in eachnode(dg), v in eachvariable(equations_parabolic)
          # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
          boundaries.u[2, v, i, k, boundary] = flux_viscous_y[v, i, 1,          k, element]
        end
      end
    else # if orientations[boundary] == 3
      # boundary in z-direction
      if neighbor_sides[boundary] == 1
        # element in -z direction of boundary
        for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations_parabolic)
          # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
          boundaries.u[1, v, i, j, boundary] = flux_viscous_z[v, i, j, nnodes(dg), element]
        end
      else
        # element in +z direction of boundary
        for j in eachnode(dg), i in eachnode(dg), v in eachvariable(equations_parabolic)
          # OBS! `boundaries.u` stores the interpolated *fluxes* and *not the solution*!
          boundaries.u[2, v, i, j, boundary] = flux_viscous_z[v, i, j, 1,          element]
        end
      end
    end
  end

  return nothing
end


function calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh::TreeMesh{3},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, cache, cache_parabolic)
  gradients_x, gradients_y, gradients_z = gradients
  flux_viscous_x, flux_viscous_y, flux_viscous_z = flux_viscous # output arrays

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      # Get solution and gradients
      u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, k, element)
      gradients_1_node = get_node_vars(gradients_x, equations_parabolic, dg, i, j, k, element)
      gradients_2_node = get_node_vars(gradients_y, equations_parabolic, dg, i, j, k, element)
      gradients_3_node = get_node_vars(gradients_z, equations_parabolic, dg, i, j, k, element)

      # Calculate viscous flux and store each component for later use
      flux_viscous_node_x = flux(u_node, (gradients_1_node, gradients_2_node, gradients_3_node), 1, equations_parabolic)
      flux_viscous_node_y = flux(u_node, (gradients_1_node, gradients_2_node, gradients_3_node), 2, equations_parabolic)
      flux_viscous_node_z = flux(u_node, (gradients_1_node, gradients_2_node, gradients_3_node), 3, equations_parabolic)
      set_node_vars!(flux_viscous_x, flux_viscous_node_x, equations_parabolic, dg, i, j, k, element)
      set_node_vars!(flux_viscous_y, flux_viscous_node_y, equations_parabolic, dg, i, j, k, element)
      set_node_vars!(flux_viscous_z, flux_viscous_node_z, equations_parabolic, dg, i, j, k, element)
    end
  end
end


# TODO: parabolic; decide if we should keep this.
function get_unsigned_normal_vector_3d(direction)
  if direction > 6 || direction < 1
    error("Direction = $direction; in 3D, direction should be 1, 2, 3, 4, 5, or 6.")
  end
  if direction == 1 || direction == 2
    return SVector(1.0, 0.0, 0.0)
  elseif direction == 3 || direction == 4
    return SVector(0.0, 1.0, 0.0)
  else
    return SVector(0.0, 0.0, 1.0)
  end
end

function calc_boundary_flux_gradients!(cache, t, boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                      mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                                      surface_integral, dg::DG)
  return nothing
end

function calc_boundary_flux_divergence!(cache, t, boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
  return nothing
end

function calc_boundary_flux_gradients!(cache, t, boundary_conditions_parabolic::NamedTuple,
                                      mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                                      surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction_gradient!(surface_flux_values, t, boundary_conditions_parabolic[1],
                                            equations_parabolic, surface_integral, dg, cache,
                                            1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction_gradient!(surface_flux_values, t, boundary_conditions_parabolic[2],
                                            equations_parabolic, surface_integral, dg, cache,
                                            2, firsts[2], lasts[2])
  calc_boundary_flux_by_direction_gradient!(surface_flux_values, t, boundary_conditions_parabolic[3],
                                            equations_parabolic, surface_integral, dg, cache,
                                            3, firsts[3], lasts[3])
  calc_boundary_flux_by_direction_gradient!(surface_flux_values, t, boundary_conditions_parabolic[4],
                                            equations_parabolic, surface_integral, dg, cache,
                                            4, firsts[4], lasts[4])
  calc_boundary_flux_by_direction_gradient!(surface_flux_values, t, boundary_conditions_parabolic[5],
                                            equations_parabolic, surface_integral, dg, cache,
                                            5, firsts[5], lasts[5])
  calc_boundary_flux_by_direction_gradient!(surface_flux_values, t, boundary_conditions_parabolic[6],
                                            equations_parabolic, surface_integral, dg, cache,
                                            6, firsts[6], lasts[6])
end


function calc_boundary_flux_by_direction_gradient!(surface_flux_values::AbstractArray{<:Any,5}, t,
                                                   boundary_condition,
                                                   equations_parabolic::AbstractEquationsParabolic,
                                                   surface_integral, dg::DG, cache,
                                                   direction, first_boundary, last_boundary)
  @unpack surface_flux = surface_integral
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for j in eachnode(dg), i in eachnode(dg)
      # Get boundary flux
      u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg, i, j, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        u_inner = u_ll
      else # Element is on the right, boundary on the left
        u_inner = u_rr
      end

      # TODO: revisit if we want more general boundary treatments.
      # This assumes the gradient numerical flux at the boundary is the gradient variable,
      # which is consistent with BR1, LDG.
      flux_inner = u_inner

      x = get_node_coords(node_coordinates, equations_parabolic, dg, i, j, boundary)
      flux = boundary_condition(flux_inner, u_inner, get_unsigned_normal_vector_3d(direction),
                                x, t, Gradient(), equations_parabolic)

      # Copy flux to left and right element storage
      for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, j, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end

function calc_boundary_flux_divergence!(cache, t, boundary_conditions_parabolic::NamedTuple,
                                        mesh::TreeMesh{3}, equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_boundary_flux_by_direction_divergence!(surface_flux_values, t, boundary_conditions_parabolic[1],
                                              equations_parabolic, surface_integral, dg, cache,
                                              1, firsts[1], lasts[1])
  calc_boundary_flux_by_direction_divergence!(surface_flux_values, t, boundary_conditions_parabolic[2],
                                              equations_parabolic, surface_integral, dg, cache,
                                              2, firsts[2], lasts[2])
  calc_boundary_flux_by_direction_divergence!(surface_flux_values, t, boundary_conditions_parabolic[3],
                                              equations_parabolic, surface_integral, dg, cache,
                                              3, firsts[3], lasts[3])
  calc_boundary_flux_by_direction_divergence!(surface_flux_values, t, boundary_conditions_parabolic[4],
                                              equations_parabolic, surface_integral, dg, cache,
                                              4, firsts[4], lasts[4])
  calc_boundary_flux_by_direction_divergence!(surface_flux_values, t, boundary_conditions_parabolic[5],
                                              equations_parabolic, surface_integral, dg, cache,
                                              5, firsts[5], lasts[5])
  calc_boundary_flux_by_direction_divergence!(surface_flux_values, t, boundary_conditions_parabolic[6],
                                              equations_parabolic, surface_integral, dg, cache,
                                              6, firsts[6], lasts[6])
end
function calc_boundary_flux_by_direction_divergence!(surface_flux_values::AbstractArray{<:Any,5}, t,
                                                     boundary_condition,
                                                     equations_parabolic::AbstractEquationsParabolic,
                                                     surface_integral, dg::DG, cache,
                                                     direction, first_boundary, last_boundary)
  @unpack surface_flux = surface_integral

  # Note: cache.boundaries.u contains the unsigned normal component (using "orientation", not "direction")
  # of the viscous flux, as computed in `prolong2boundaries!`
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for j in eachnode(dg), i in eachnode(dg)
      # Get viscous boundary fluxes
      flux_ll, flux_rr = get_surface_node_vars(u, equations_parabolic, dg, i, j, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        flux_inner = flux_ll
      else # Element is on the right, boundary on the left
        flux_inner = flux_rr
      end

      x = get_node_coords(node_coordinates, equations_parabolic, dg, i, j, boundary)

      # TODO: add a field in `cache.boundaries` for gradient information. UPDATE THIS COMMENT
      # Here, we pass in `u_inner = nothing` since we overwrite cache.boundaries.u with gradient information.
      # This currently works with Dirichlet/Neuman boundary conditions for LaplaceDiffusion3D and
      # NoSlipWall/Adiabatic boundary conditions for CompressibleNavierStokesDiffusion3D as of 2022-6-27.
      # It will not work with implementations which utilize `u_inner` to impose boundary conditions.
      flux = boundary_condition(flux_inner, nothing, get_unsigned_normal_vector_3d(direction),
                                x, t, Divergence(), equations_parabolic)

      # Copy flux to left and right element storage
      for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, j, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end


# Calculate the gradient of the transformed variables
function calc_gradient!(gradients, u_transformed, t,
                        mesh::TreeMesh{3}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG, cache, cache_parabolic)

  gradients_x, gradients_y, gradients_z = gradients

  # Reset du
  @trixi_timeit timer() "reset gradients" begin
    reset_du!(gradients_x, dg, cache)
    reset_du!(gradients_y, dg, cache)
    reset_du!(gradients_z, dg, cache)
  end

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" begin
    @unpack derivative_dhat = dg.basis
    @threaded for element in eachelement(dg, cache)

      # Calculate volume terms in one element
      for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, k, element)

        for ii in eachnode(dg)
          multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i], u_node, equations_parabolic, dg, ii, j, k, element)
        end

        for jj in eachnode(dg)
          multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j], u_node, equations_parabolic, dg, i, jj, k, element)
        end

        for kk in eachnode(dg)
          multiply_add_to_node_vars!(gradients_z, derivative_dhat[kk, k], u_node, equations_parabolic, dg, i, j, kk, element)
        end
      end
    end
  end

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" prolong2interfaces!(
    cache_parabolic, u_transformed, mesh, equations_parabolic, dg.surface_integral, dg)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" begin
    @unpack surface_flux_values = cache_parabolic.elements
    @unpack neighbor_ids, orientations = cache_parabolic.interfaces

    @threaded for interface in eachinterface(dg, cache_parabolic)
      # Get neighboring elements
      left_id  = neighbor_ids[1, interface]
      right_id = neighbor_ids[2, interface]

      # Determine interface direction with respect to elements:
      # orientation = 1: left -> 2, right -> 1
      # orientation = 2: left -> 4, right -> 3
      # orientation = 3: left -> 6, right -> 5
      left_direction  = 2 * orientations[interface]
      right_direction = 2 * orientations[interface] - 1

      for j in eachnode(dg), i in eachnode(dg)
        # Call pointwise Riemann solver
        u_ll, u_rr = get_surface_node_vars(cache_parabolic.interfaces.u,
                                           equations_parabolic, dg, i, j, interface)
        flux = 0.5 * (u_ll + u_rr)

        # Copy flux to left and right element storage
        for v in eachvariable(equations_parabolic)
          surface_flux_values[v, i, j, left_direction,  left_id]  = flux[v]
          surface_flux_values[v, i, j, right_direction, right_id] = flux[v]
        end
      end
    end
  end

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" prolong2boundaries!(
    cache_parabolic, u_transformed, mesh, equations_parabolic, dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux_gradients!(
    cache_parabolic, t, boundary_conditions_parabolic, mesh, equations_parabolic,
    dg.surface_integral, dg)

  # TODO: parabolic; mortars

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" begin
    @unpack boundary_interpolation = dg.basis
    @unpack surface_flux_values = cache_parabolic.elements

    # Note that all fluxes have been computed with outward-pointing normal vectors.
    # Access the factors only once before beginning the loop to increase performance.
    # We also use explicit assignments instead of `+=` to let `@muladd` turn these
    # into FMAs (see comment at the top of the file).
    factor_1 = boundary_interpolation[1,          1]
    factor_2 = boundary_interpolation[nnodes(dg), 2]
    @threaded for element in eachelement(dg, cache)
      for m in eachnode(dg), l in eachnode(dg)
        for v in eachvariable(equations_parabolic)
          # surface at -x
          gradients_x[v, 1,          l, m, element] = (
            gradients_x[v, 1,          l, m, element] - surface_flux_values[v, l, m, 1, element] * factor_1)

          # surface at +x
          gradients_x[v, nnodes(dg), l, m, element] = (
            gradients_x[v, nnodes(dg), l, m, element] + surface_flux_values[v, l, m, 2, element] * factor_2)

          # surface at -y
          gradients_y[v, l, 1,       m, element] = (
            gradients_y[v, l, 1,       m, element] - surface_flux_values[v, l, m, 3, element] * factor_1)

          # surface at +y
          gradients_y[v, l, nnodes(dg), m, element] = (
            gradients_y[v, l, nnodes(dg), m, element] + surface_flux_values[v, l, m, 4, element] * factor_2)

          # surface at -z
          gradients_z[v, l, m, 1,       element] = (
            gradients_z[v, l, m, 1,       element] - surface_flux_values[v, l, m, 5, element] * factor_1)

          # surface at +z
          gradients_z[v, l, m, nnodes(dg), element] = (
            gradients_z[v, l, m, nnodes(dg), element] + surface_flux_values[v, l, m, 6, element] * factor_2)
        end
      end
    end
  end

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" begin
    apply_jacobian!(gradients_x, mesh, equations_parabolic, dg, cache_parabolic)
    apply_jacobian!(gradients_y, mesh, equations_parabolic, dg, cache_parabolic)
    apply_jacobian!(gradients_z, mesh, equations_parabolic, dg, cache_parabolic)
  end

  return nothing
end


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_parabolic(mesh::TreeMesh{3}, equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, parabolic_scheme, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations_hyperbolic, dg.basis, RealT, uEltype)

  n_vars = nvariables(equations_hyperbolic)
  n_nodes = nnodes(elements)
  n_elements = nelements(elements)
  u_transformed = Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_nodes, n_elements)
  gradients = ntuple(_ -> similar(u_transformed), ndims(mesh))
  flux_viscous = ntuple(_ -> similar(u_transformed), ndims(mesh))

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  # mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  # cache = (; elements, interfaces, boundaries, mortars)
  cache = (; elements, interfaces, boundaries, gradients, flux_viscous, u_transformed)

  # Add specialized parts of the cache required to compute the mortars etc.
  # cache = (;cache..., create_cache(mesh, equations_parabolic, dg.mortar, uEltype)...)

  return cache
end


# Needed to *not* flip the sign of the inverse Jacobian.
# This is because the parabolic fluxes are assumed to be of the form
#   `du/dt + df/dx = dg/dx + source(x,t)`,
# where f(u) is the inviscid flux and g(u) is the viscous flux.
function apply_jacobian!(du, mesh::TreeMesh{3},
                         equations::AbstractEquationsParabolic, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    factor = cache.elements.inverse_jacobian[element]

    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, k, element] *= factor
      end
    end
  end

  return nothing
end

end # @muladd
