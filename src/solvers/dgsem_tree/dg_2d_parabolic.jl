# This file collects all methods that have been updated to work with parabolic systems of equations
#
# assumptions: parabolic terms are of the form div(f(u, grad(u))) and
# will be discretized first order form as follows:
#               1. compute grad(u)
#               2. compute f(u, grad(u))
#               3. compute div(u)
# boundary conditions will be applied to both grad(u) and div(u).
function rhs_parabolic!(du, u, t, mesh::TreeMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                        initial_condition, boundary_conditions, source_terms,
                        dg::DG, dg_parabolic, cache, cache_parabolic)

  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)


  @unpack u_transformed, u_grad, viscous_flux = cache_parabolic
  @trixi_timeit timer() "transform variables" transform_variables!(u_transformed, u,
                                                                   mesh, equations_parabolic,
                                                                   dg, dg_parabolic,
                                                                   cache, cache_parabolic)

  @trixi_timeit timer() "calculate gradient" calc_gradient!(u_grad, u_transformed, t, mesh,
                                                            equations_parabolic,
                                                            boundary_conditions, dg,
                                                            cache, cache_parabolic)

  @trixi_timeit timer() "calculate viscous fluxes" calc_viscous_fluxes!(viscous_flux, u_grad, u_transformed,
                                                                        mesh, equations_parabolic,
                                                                        dg, cache, cache_parabolic)

  @trixi_timeit timer() "calculate divergence" calc_divergence!(du, u_transformed, t, viscous_flux,
                                                                mesh,
                                                                equations_parabolic,
                                                                boundary_conditions, dg,
                                                                dg_parabolic, cache,
                                                                cache_parabolic)

  return nothing

end

# Transform solution variables prior to taking the gradient
# (e.g., conservative to primitive variables). Defaults to doing nothing.
# TODO: can we avoid copying data?
function transform_variables!(u_transformed, u, mesh::TreeMesh{2},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, dg_parabolic, cache, cache_parabolic)
  @threaded for element in eachelement(dg, cache)
    # Calculate volume terms in one element
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)
      u_transformed_node = gradient_variable_transformation(equations_parabolic, dg_parabolic)(u_node, equations_parabolic)
      set_node_vars!(u_transformed, u_transformed_node, equations_parabolic, dg, i, j, element)
    end
  end
end

# note: the argument dg_parabolic is not a DG type; it contains solver-specific
# information such as an LDG penalty parameter.
function calc_divergence!(du, u, t, viscous_flux,
                          mesh::TreeMesh{2}, equations_parabolic,
                          boundary_conditions_parabolic, dg::DG,
                          dg_parabolic, # not a `DG` type
                          cache, cache_parabolic)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" begin
    reset_du!(du, dg, cache)
  end

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" begin
    @unpack derivative_dhat = dg.basis
    @threaded for element in eachelement(dg, cache)

      # Calculate volume terms in one element
      for j in eachnode(dg), i in eachnode(dg)
        flux_1_node = get_node_vars(viscous_flux[1], equations_parabolic, dg, i, j, element)
        flux_2_node = get_node_vars(viscous_flux[2], equations_parabolic, dg, i, j, element)

        for ii in eachnode(dg)
          multiply_add_to_node_vars!(du, derivative_dhat[ii, i], flux_1_node, equations_parabolic, dg, ii, j, element)
        end

        for jj in eachnode(dg)
          multiply_add_to_node_vars!(du, derivative_dhat[jj, j], flux_2_node, equations_parabolic, dg, i, jj, element)
        end
      end
    end
  end

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" begin
    @unpack interfaces = cache_parabolic
    @unpack orientations = interfaces

    @threaded for interface in eachinterface(dg, cache)
      left_element  = interfaces.neighbor_ids[1, interface]
      right_element = interfaces.neighbor_ids[2, interface]

      if orientations[interface] == 1
        # interface in x-direction
        for j in eachnode(dg), v in eachvariable(equations_parabolic)
          interfaces.u[1, v, j, interface] = viscous_flux[1][v, nnodes(dg), j, left_element]
          interfaces.u[2, v, j, interface] = viscous_flux[1][v,          1, j, right_element]
        end
      else # if orientations[interface] == 2
        # interface in y-direction
        for i in eachnode(dg), v in eachvariable(equations_parabolic)
          interfaces.u[1, v, i, interface] = viscous_flux[2][v, i, nnodes(dg), left_element]
          interfaces.u[2, v, i, interface] = viscous_flux[2][v, i,          1, right_element]
        end
      end
    end
  end

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
      left_direction  = 2 * orientations[interface]
      right_direction = 2 * orientations[interface] - 1

      for i in eachnode(dg)
        # Call pointwise Riemann solver
        u_ll, u_rr = get_surface_node_vars(cache_parabolic.interfaces.u, equations_parabolic, dg, i, interface)
        flux = 0.5 * (u_ll + u_rr)

        # Copy flux to left and right element storage
        for v in eachvariable(equations_parabolic)
          surface_flux_values[v, i, left_direction,  left_id]  = flux[v]
          surface_flux_values[v, i, right_direction, right_id] = flux[v]
        end
      end
    end
  end

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" begin
    @unpack boundaries = cache_parabolic
    @unpack orientations, neighbor_sides = boundaries

    @threaded for boundary in eachboundary(dg, cache_parabolic)
      element = boundaries.neighbor_ids[boundary]

      if orientations[boundary] == 1
        # TODO Make this cleaner (remove the let)
        let u = viscous_flux[1]
          # boundary in x-direction
          if neighbor_sides[boundary] == 1
            # element in -x direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
              boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
            end
          else # Element in +x direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
              boundaries.u[2, v, l, boundary] = u[v, 1,          l, element]
            end
          end
        end
      else # if orientations[boundary] == 2
        # TODO Make this cleaner (remove the let)
        let u = viscous_flux[2]
          # boundary in y-direction
          if neighbor_sides[boundary] == 1
            # element in -y direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
              boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
            end
          else
            # element in +y direction of boundary
            for l in eachnode(dg), v in eachvariable(equations_parabolic)
              boundaries.u[2, v, l, boundary] = u[v, l, 1,          element]
            end
          end
        end
      end
    end
  end

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" begin
    calc_divergence_boundary_flux!(cache_parabolic, t, boundary_conditions_parabolic,
                                   mesh, equations_parabolic, dg.surface_integral, dg)
  end

  # Prolong solution to mortars
  # @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
  #   cache, u, mesh, equations_parabolic, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  # @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
  #   cache.elements.surface_flux_values, mesh,
  #   have_nonconservative_terms(equations_parabolic), equations_parabolic,
  #   dg.mortar, dg.surface_integral, dg, cache)

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
      for l in eachnode(dg)
        for v in eachvariable(equations_parabolic)
          # surface at -x
          du[v, 1,          l, element] = (
            du[v, 1,          l, element] - surface_flux_values[v, l, 1, element] * factor_1)

          # surface at +x
          du[v, nnodes(dg), l, element] = (
            du[v, nnodes(dg), l, element] + surface_flux_values[v, l, 2, element] * factor_2)

          # surface at -y
          du[v, l, 1,          element] = (
            du[v, l, 1,          element] - surface_flux_values[v, l, 3, element] * factor_1)

          # surface at +y
          du[v, l, nnodes(dg), element] = (
            du[v, l, nnodes(dg), element] + surface_flux_values[v, l, 4, element] * factor_2)
        end
      end
    end
  end

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" begin
    apply_jacobian!(du, mesh, equations_parabolic, dg, cache_parabolic)
  end

  return nothing
end


function calc_viscous_fluxes!(viscous_flux, u_grad, u, mesh::TreeMesh{2},
                              equations_parabolic::AbstractEquationsParabolic,
                              dg::DG, cache, cache_parabolic)
  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      # Get solution and gradients
      u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)
      u_grad_1_node = get_node_vars(u_grad[1], equations_parabolic, dg, i, j, element)
      u_grad_2_node = get_node_vars(u_grad[2], equations_parabolic, dg, i, j, element)

      # Calculate viscous flux and store each component for later use
      viscous_flux_node = flux(u_node, (u_grad_1_node, u_grad_2_node), equations_parabolic)
      set_node_vars!(viscous_flux[1], viscous_flux_node[1], equations_parabolic, dg, i, j, element)
      set_node_vars!(viscous_flux[2], viscous_flux_node[2], equations_parabolic, dg, i, j, element)
    end
  end
end

# TODO: decide if we should keep this, and if so, extend to 3D.
function get_unsigned_normal_vector_2d(direction)
  if direction > 4 || direction < 1
    @warn "Direction = $direction; in 2D, direction should be 1, 2, 3, or 4."
  end
  if direction==1 || direction==2
    return SVector(1.0, 0.0)
  else
    return SVector(0.0, 1.0)
  end
end

function calc_gradient_boundary_flux!(cache, t, boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                      mesh::TreeMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                                      surface_integral, dg::DG)
  return nothing
end

function calc_divergence_boundary_flux!(cache, t, boundary_conditions_parabolic::BoundaryConditionPeriodic,
                                        mesh::TreeMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
  return nothing
end

function calc_gradient_boundary_flux!(cache, t, boundary_conditions_parabolic::NamedTuple,
                                      mesh::TreeMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                                      surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_gradient_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[1],
                                            equations_parabolic, surface_integral, dg, cache,
                                            1, firsts[1], lasts[1])
  calc_gradient_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[2],
                                            equations_parabolic, surface_integral, dg, cache,
                                            2, firsts[2], lasts[2])
  calc_gradient_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[3],
                                            equations_parabolic, surface_integral, dg, cache,
                                            3, firsts[3], lasts[3])
  calc_gradient_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[4],
                                            equations_parabolic, surface_integral, dg, cache,
                                            4, firsts[4], lasts[4])
end
function calc_gradient_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t,
                                                   boundary_condition,
                                                   equations_parabolic::AbstractEquationsParabolic,
                                                   surface_integral, dg::DG, cache,
                                                   direction, first_boundary, last_boundary)
  @unpack surface_flux = surface_integral
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for i in eachnode(dg)
      # Get boundary flux
      u_ll, u_rr = get_surface_node_vars(u, equations_parabolic, dg, i, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        u_inner = u_ll
      else # Element is on the right, boundary on the left
        u_inner = u_rr
      end

      # TODO: revisit if we want more general boundary treatments.
      # This assumes the gradient numerical flux at the boundary is the gradient variable,
      # which is consistent with BR1, LDG.
      flux_inner = u_inner

      x = get_node_coords(node_coordinates, equations_parabolic, dg, i, boundary)
      flux = boundary_condition(flux_inner, u_inner, get_unsigned_normal_vector_2d(direction),
                                x, t, Gradient(), equations_parabolic)

      # Copy flux to left and right element storage
      for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end

function calc_divergence_boundary_flux!(cache, t, boundary_conditions_parabolic::NamedTuple,
                                        mesh::TreeMesh{2}, equations_parabolic::AbstractEquationsParabolic,
                                        surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  @unpack n_boundaries_per_direction = cache.boundaries

  # Calculate indices
  lasts = accumulate(+, n_boundaries_per_direction)
  firsts = lasts - n_boundaries_per_direction .+ 1

  # Calc boundary fluxes in each direction
  calc_divergence_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[1],
                                              equations_parabolic, surface_integral, dg, cache,
                                              1, firsts[1], lasts[1])
  calc_divergence_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[2],
                                              equations_parabolic, surface_integral, dg, cache,
                                              2, firsts[2], lasts[2])
  calc_divergence_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[3],
                                              equations_parabolic, surface_integral, dg, cache,
                                              3, firsts[3], lasts[3])
  calc_divergence_boundary_flux_by_direction!(surface_flux_values, t, boundary_conditions_parabolic[4],
                                              equations_parabolic, surface_integral, dg, cache,
                                              4, firsts[4], lasts[4])
end
function calc_divergence_boundary_flux_by_direction!(surface_flux_values::AbstractArray{<:Any,4}, t,
                                                     boundary_condition,
                                                     equations_parabolic::AbstractEquationsParabolic,
                                                     surface_integral, dg::DG, cache,
                                                     direction, first_boundary, last_boundary)
  @unpack surface_flux = surface_integral

  # Note: cache.boundaries.u contains the unsigned normal component (using "orientation", not "direction")
  # of the viscous flux, as computed in `prolong2boundaries` of `calc_divergence!``
  @unpack u, neighbor_ids, neighbor_sides, node_coordinates, orientations = cache.boundaries

  @threaded for boundary in first_boundary:last_boundary
    # Get neighboring element
    neighbor = neighbor_ids[boundary]

    for i in eachnode(dg)
      # Get viscous boundary fluxes
      flux_ll, flux_rr = get_surface_node_vars(u, equations_parabolic, dg, i, boundary)
      if neighbor_sides[boundary] == 1 # Element is on the left, boundary on the right
        flux_inner = flux_ll
      else # Element is on the right, boundary on the left
        flux_inner = flux_rr
      end

      x = get_node_coords(node_coordinates, equations_parabolic, dg, i, boundary)

      # TODO: add a field in `cache.boundaries` for gradient information.
      # Here, we pass in `u_inner = nothing` since we overwrite cache.boundaries.u with gradient information.
      # This currently works with Dirichlet/Neuman boundary conditions for LaplaceDiffusion2D and
      # NoSlipWall/Adiabatic boundary conditions for CompressibleNavierStokesEquations2D as of 2022-6-27.
      # It will not work with implementations which utilize `u_inner` to impose boundary conditions.
      flux = boundary_condition(flux_inner, nothing, get_unsigned_normal_vector_2d(direction),
                                x, t, Divergence(), equations_parabolic)

      # Copy flux to left and right element storage
      for v in eachvariable(equations_parabolic)
        surface_flux_values[v, i, direction, neighbor] = flux[v]
      end
    end
  end

  return nothing
end

function calc_gradient!(u_grad, u, t,
                        mesh::Union{TreeMesh{2}, P4estMesh{2}}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG, cache, cache_parabolic)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" begin
    reset_du!(u_grad[1], dg, cache)
    reset_du!(u_grad[2], dg, cache)
  end

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" begin
    @unpack derivative_dhat = dg.basis
    @threaded for element in eachelement(dg, cache)

      # Calculate volume terms in one element
      for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations_parabolic, dg, i, j, element)

        for ii in eachnode(dg)
          multiply_add_to_node_vars!(u_grad[1], derivative_dhat[ii, i], u_node, equations_parabolic, dg, ii, j, element)
        end

        for jj in eachnode(dg)
          multiply_add_to_node_vars!(u_grad[2], derivative_dhat[jj, j], u_node, equations_parabolic, dg, i, jj, element)
        end
      end
    end
  end

  # Prolong solution to interfaces
  @trixi_timeit timer() "prolong2interfaces" begin
    @unpack interfaces = cache_parabolic
    @unpack orientations = interfaces

    @threaded for interface in eachinterface(dg, cache)
      left_element  = interfaces.neighbor_ids[1, interface]
      right_element = interfaces.neighbor_ids[2, interface]

      if orientations[interface] == 1
        # interface in x-direction
        for j in eachnode(dg), v in eachvariable(equations_parabolic)
          interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, left_element]
          interfaces.u[2, v, j, interface] = u[v,          1, j, right_element]
        end
      else # if orientations[interface] == 2
        # interface in y-direction
        for i in eachnode(dg), v in eachvariable(equations_parabolic)
          interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), left_element]
          interfaces.u[2, v, i, interface] = u[v, i,          1, right_element]
        end
      end
    end
  end

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
      left_direction  = 2 * orientations[interface]
      right_direction = 2 * orientations[interface] - 1

      for i in eachnode(dg)
        # Call pointwise Riemann solver
        u_ll, u_rr = get_surface_node_vars(cache_parabolic.interfaces.u,
                                           equations_parabolic, dg, i, interface)
        flux = 0.5 * (u_ll + u_rr)

        # Copy flux to left and right element storage
        for v in eachvariable(equations_parabolic)
          surface_flux_values[v, i, left_direction,  left_id]  = flux[v]
          surface_flux_values[v, i, right_direction, right_id] = flux[v]
        end
      end
    end
  end

  # Prolong solution to boundaries
  @trixi_timeit timer() "prolong2boundaries" begin
    @unpack boundaries = cache_parabolic
    @unpack orientations, neighbor_sides = boundaries

    @threaded for boundary in eachboundary(dg, cache_parabolic)
      element = boundaries.neighbor_ids[boundary]

      if orientations[boundary] == 1
        # boundary in x-direction
        if neighbor_sides[boundary] == 1
          # element in -x direction of boundary
          for l in eachnode(dg), v in eachvariable(equations_parabolic)
            boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
          end
        else # Element in +x direction of boundary
          for l in eachnode(dg), v in eachvariable(equations_parabolic)
            boundaries.u[2, v, l, boundary] = u[v, 1,          l, element]
          end
        end
      else # if orientations[boundary] == 2
        # boundary in y-direction
        if neighbor_sides[boundary] == 1
          # element in -y direction of boundary
          for l in eachnode(dg), v in eachvariable(equations_parabolic)
            boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
          end
        else
          # element in +y direction of boundary
          for l in eachnode(dg), v in eachvariable(equations_parabolic)
            boundaries.u[2, v, l, boundary] = u[v, l, 1,          element]
          end
        end
      end
    end
  end

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" begin
    calc_gradient_boundary_flux!(cache_parabolic, t, boundary_conditions_parabolic,
                                 mesh, equations_parabolic, dg.surface_integral, dg)
  end

  # Prolong solution to mortars
  # @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
  #   cache, u, mesh, equations_parabolic, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  # @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
  #   cache.elements.surface_flux_values, mesh,
  #   have_nonconservative_terms(equations_parabolic), equations_parabolic,
  #   dg.mortar, dg.surface_integral, dg, cache)

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
      for l in eachnode(dg)
        for v in eachvariable(equations_parabolic)
          let du = u_grad[1]
            # surface at -x
            du[v, 1,          l, element] = (
              du[v, 1,          l, element] - surface_flux_values[v, l, 1, element] * factor_1)

            # surface at +x
            du[v, nnodes(dg), l, element] = (
              du[v, nnodes(dg), l, element] + surface_flux_values[v, l, 2, element] * factor_2)
          end

          let du = u_grad[2]
            # surface at -y
            du[v, l, 1,          element] = (
              du[v, l, 1,          element] - surface_flux_values[v, l, 3, element] * factor_1)

            # surface at +y
            du[v, l, nnodes(dg), element] = (
              du[v, l, nnodes(dg), element] + surface_flux_values[v, l, 4, element] * factor_2)
          end
        end
      end
    end
  end

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" begin
    apply_jacobian!(u_grad[1], mesh, equations_parabolic, dg, cache_parabolic)
    apply_jacobian!(u_grad[2], mesh, equations_parabolic, dg, cache_parabolic)
  end

  return nothing
end


# This method is called when a SemidiscretizationHyperbolic is constructed.
# It constructs the basic `cache` used throughout the simulation to compute
# the RHS etc.
function create_cache_parabolic(mesh::TreeMesh{2}, equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic,
                                dg::DG, dg_parabolic, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations_hyperbolic, dg.basis, RealT, uEltype)

  n_vars = nvariables(equations_hyperbolic)
  n_nodes = nnodes(elements)
  n_elements = nelements(elements)
  u_transformed = Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_elements)
  u_grad = ntuple(_ -> similar(u_transformed), ndims(mesh))
  viscous_flux = ntuple(_ -> similar(u_transformed), ndims(mesh))

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  # mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  # cache = (; elements, interfaces, boundaries, mortars)
  cache = (; elements, interfaces, boundaries, u_grad, viscous_flux, u_transformed)

  # Add specialized parts of the cache required to compute the mortars etc.
  # cache = (;cache..., create_cache(mesh, equations_parabolic, dg.mortar, uEltype)...)

  return cache
end


# Needed to *not* flip the sign of the inverse Jacobian
function apply_jacobian!(du, mesh::TreeMesh{2},
                         equations::AbstractEquationsParabolic, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    factor = cache.elements.inverse_jacobian[element]

    for j in eachnode(dg), i in eachnode(dg)
      for v in eachvariable(equations)
        du[v, i, j, element] *= factor
      end
    end
  end

  return nothing
end
