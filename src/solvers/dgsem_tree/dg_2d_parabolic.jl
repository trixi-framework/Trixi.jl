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
                                                                   equations_parabolic)

  calc_gradient!(u_grad, u_transformed, t, mesh, equations_parabolic,
                 boundary_conditions, dg, cache, cache_parabolic)

  calc_viscous_fluxes!(viscous_flux, u_transformed, u_grad,
                       mesh, equations_parabolic, dg, cache, cache_parabolic)

  calc_divergence!(du, u_transformed, t, viscous_flux, mesh, equations_parabolic,
                   boundary_conditions, dg, dg_parabolic, cache, cache_parabolic)

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
        for j in eachnode(dg), v in eachvariable(equations)
          interfaces.u[1, v, j, interface] = u[v, nnodes(dg), j, left_element]
          interfaces.u[2, v, j, interface] = u[v,          1, j, right_element]
        end
      else # if orientations[interface] == 2
        # interface in y-direction
        for i in eachnode(dg), v in eachvariable(equations)
          interfaces.u[1, v, i, interface] = u[v, i, nnodes(dg), left_element]
          interfaces.u[2, v, i, interface] = u[v, i,          1, right_element]
        end
      end
    end
  end

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" begin
    @unpack surface_flux_values = cache_parabolic.elements
    @unpack u, neighbor_ids, orientations = cache_parabolic.interfaces

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
        u_ll, u_rr = get_surface_node_vars(u, equations, dg, i, interface)
        flux = 0.5 * (u_ll + u_rr)

        # Copy flux to left and right element storage
        for v in eachvariable(equations)
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
          for l in eachnode(dg), v in eachvariable(equations)
            boundaries.u[1, v, l, boundary] = u[v, nnodes(dg), l, element]
          end
        else # Element in +x direction of boundary
          for l in eachnode(dg), v in eachvariable(equations)
            boundaries.u[2, v, l, boundary] = u[v, 1,          l, element]
          end
        end
      else # if orientations[boundary] == 2
        # boundary in y-direction
        if neighbor_sides[boundary] == 1
          # element in -y direction of boundary
          for l in eachnode(dg), v in eachvariable(equations)
            boundaries.u[1, v, l, boundary] = u[v, l, nnodes(dg), element]
          end
        else
          # element in +y direction of boundary
          for l in eachnode(dg), v in eachvariable(equations)
            boundaries.u[2, v, l, boundary] = u[v, l, 1,          element]
          end
        end
      end
    end
  end

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" begin
    @assert boundary_conditions_parabolic == boundary_conditions_periodic
  end

  # Prolong solution to mortars
  # @trixi_timeit timer() "prolong2mortars" prolong2mortars!(
  #   cache, u, mesh, equations, dg.mortar, dg.surface_integral, dg)

  # Calculate mortar fluxes
  # @trixi_timeit timer() "mortar flux" calc_mortar_flux!(
  #   cache.elements.surface_flux_values, mesh,
  #   have_nonconservative_terms(equations), equations,
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
        for v in eachvariable(equations)
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
function create_cache_parabolic(mesh::TreeMesh{2}, equations_parabolic,
                                dg::DG, dg_parabolic, RealT, uEltype)
  # Get cells for which an element needs to be created (i.e. all leaf cells)
  leaf_cell_ids = local_leaf_cells(mesh.tree)

  elements = init_elements(leaf_cell_ids, mesh, equations_parabolic, dg.basis, RealT, uEltype)

  n_vars = nvariables(equations_parabolic)
  n_nodes = nnodes(elements)
  n_elements = nelements(elements)
  u_grad = ntuple(_ -> Array{uEltype}(undef, n_vars, n_nodes, n_nodes, n_elements), ndims(mesh))

  interfaces = init_interfaces(leaf_cell_ids, mesh, elements)

  boundaries = init_boundaries(leaf_cell_ids, mesh, elements)

  # mortars = init_mortars(leaf_cell_ids, mesh, elements, dg.mortar)

  # cache = (; elements, interfaces, boundaries, mortars)
  cache = (; elements, interfaces, boundaries)

  # Add specialized parts of the cache required to compute the mortars etc.
  # cache = (;cache..., create_cache(mesh, equations_parabolic, dg.mortar, uEltype)...)

  return cache
end
