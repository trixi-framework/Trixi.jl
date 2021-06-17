# By default, Julia/LLVM does not use FMAs. Hence, we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi/
@muladd begin


function rhs!(du, u, t,
              mesh::CurvedMesh{3}, equations,
              initial_condition, boundary_conditions, source_terms,
              dg::DG, cache)
  # Reset du
  @trixi_timeit timer() "reset ∂u/∂t" du .= zero(eltype(du))

  # Calculate volume integral
  @trixi_timeit timer() "volume integral" calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.volume_integral, dg, cache)

  # Calculate interface fluxes
  @trixi_timeit timer() "interface flux" calc_interface_flux!(
    cache, u, mesh,
    have_nonconservative_terms(equations), equations,
    dg.surface_integral, dg)

  # Calculate boundary fluxes
  @trixi_timeit timer() "boundary flux" calc_boundary_flux!(
    cache, u, t, boundary_conditions, mesh, equations, dg.surface_integral, dg)

  # Calculate surface integrals
  @trixi_timeit timer() "surface integral" calc_surface_integral!(
    du, u, mesh, equations, dg.surface_integral, dg, cache)

  # Apply Jacobian from mapping to reference element
  @trixi_timeit timer() "Jacobian" apply_jacobian!(
    du, mesh, equations, dg, cache)

  # Calculate source terms
  @trixi_timeit timer() "source terms" calc_sources!(
    du, u, t, source_terms, equations, dg, cache)

  return nothing
end


function calc_volume_integral!(du, u,
                               mesh::CurvedMesh{3},
                               nonconservative_terms::Val{false}, equations,
                               volume_integral::VolumeIntegralWeakForm,
                               dg::DGSEM, cache)
  @unpack derivative_dhat = dg.basis
  @unpack contravariant_vectors = cache.elements

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, k, element)

      flux1 = flux(u_node, 1, equations)
      flux2 = flux(u_node, 2, equations)
      flux3 = flux(u_node, 3, equations)

      # Compute the contravariant flux by taking the scalar product of the
      # first contravariant vector Ja^1 and the flux vector
      Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors, i, j, k, element)
      contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2 + Ja13 * flux3
      for ii in eachnode(dg)
        add_to_node_vars!(du, derivative_dhat[ii, i], contravariant_flux1, equations, dg, ii, j, k, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # second contravariant vector Ja^2 and the flux vector
      Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors, i, j, k, element)
      contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2 + Ja23 * flux3
      for jj in eachnode(dg)
        add_to_node_vars!(du, derivative_dhat[jj, j], contravariant_flux2, equations, dg, i, jj, k, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # third contravariant vector Ja^3 and the flux vector
      Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors, i, j, k, element)
      contravariant_flux3 = Ja31 * flux1 + Ja32 * flux2 + Ja33 * flux3
      for kk in eachnode(dg)
        add_to_node_vars!(du, derivative_dhat[kk, k], contravariant_flux3, equations, dg, i, j, kk, element)
      end
    end
  end

  return nothing
end


# Calculate 3D twopoint contravariant flux (element version)
@inline function calcflux_twopoint!(ftilde1, ftilde2, ftilde3, u::AbstractArray{<:Any,5}, element,
                                    mesh::CurvedMesh{3}, equations, volume_flux, dg::DGSEM, cache)
  @unpack contravariant_vectors = cache.elements

  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    # Pull the solution values and contravariant vectors at the node i,j,k
    u_node = get_node_vars(u, equations, dg, i, j, k, element)
    Ja11_node, Ja12_node, Ja13_node = get_contravariant_vector(1, contravariant_vectors,
                                                               i, j, k, element)
    Ja21_node, Ja22_node, Ja23_node = get_contravariant_vector(2, contravariant_vectors,
                                                               i, j, k, element)
    Ja31_node, Ja32_node, Ja33_node = get_contravariant_vector(3, contravariant_vectors,
                                                               i, j, k, element)
    # diagonal (consistent) part not needed since diagonal of
    # dg.basis.derivative_split_transpose is zero!
    set_node_vars!(ftilde1, zero(u_node), equations, dg, i, i, j, k)
    set_node_vars!(ftilde2, zero(u_node), equations, dg, j, i, j, k)
    set_node_vars!(ftilde3, zero(u_node), equations, dg, k, i, j, k)

    # contravariant fluxes in the first direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      flux2 = volume_flux(u_node, u_node_ii, 2, equations)
      flux3 = volume_flux(u_node, u_node_ii, 3, equations)
      # pull the contravariant vectors and compute their average
      Ja11_node_ii, Ja12_node_ii, Ja13_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                                          ii, j, k, element)
      Ja11_avg = 0.5 * (Ja11_node + Ja11_node_ii)
      Ja12_avg = 0.5 * (Ja12_node + Ja12_node_ii)
      Ja13_avg = 0.5 * (Ja13_node + Ja13_node_ii)
      # compute the contravariant sharp flux
      fluxtilde1 = Ja11_avg * flux1 + Ja12_avg * flux2 + Ja13_avg * flux3
      # save and exploit symmetry
      set_node_vars!(ftilde1, fluxtilde1, equations, dg, i, ii, j, k)
      set_node_vars!(ftilde1, fluxtilde1, equations, dg, ii, i, j, k)
    end

    # contravariant fluxes in the second direction
    for jj in (j+1):nnodes(dg)
      u_node_jj  = get_node_vars(u, equations, dg, i, jj, k, element)
      flux1 = volume_flux(u_node, u_node_jj, 1, equations)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      flux3 = volume_flux(u_node, u_node_jj, 3, equations)
      # pull the contravariant vectors and compute their average
      Ja21_node_jj, Ja22_node_jj, Ja23_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                                          i, jj, k, element)
      Ja21_avg = 0.5 * (Ja21_node + Ja21_node_jj)
      Ja22_avg = 0.5 * (Ja22_node + Ja22_node_jj)
      Ja23_avg = 0.5 * (Ja23_node + Ja23_node_jj)
      # compute the contravariant sharp flux
      fluxtilde2 = Ja21_avg * flux1 + Ja22_avg * flux2 + Ja23_avg * flux3
      # save and exploit symmetry
      set_node_vars!(ftilde2, fluxtilde2, equations, dg, j,  i, jj, k)
      set_node_vars!(ftilde2, fluxtilde2, equations, dg, jj, i, j , k)
    end

    # contravariant fluxes in the third direction
    for kk in (k+1):nnodes(dg)
      u_node_kk  = get_node_vars(u, equations, dg, i, j, kk, element)
      flux1 = volume_flux(u_node, u_node_kk, 1, equations)
      flux2 = volume_flux(u_node, u_node_kk, 2, equations)
      flux3 = volume_flux(u_node, u_node_kk, 3, equations)
      # pull the contravariant vectors and compute their average
      Ja31_node_kk, Ja32_node_kk, Ja33_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                                          i, j, kk, element)
      Ja31_avg = 0.5 * (Ja31_node + Ja31_node_kk)
      Ja32_avg = 0.5 * (Ja32_node + Ja32_node_kk)
      Ja33_avg = 0.5 * (Ja33_node + Ja33_node_kk)
      # compute the contravariant sharp flux
      fluxtilde3 = Ja31_avg * flux1 + Ja32_avg * flux2 + Ja33_avg * flux3
      # save and exploit symmetry
      set_node_vars!(ftilde3, fluxtilde3, equations, dg, k,  i, j, kk)
      set_node_vars!(ftilde3, fluxtilde3, equations, dg, kk, i, j , k)
    end
  end

  calcflux_twopoint_nonconservative!(ftilde1, ftilde2, ftilde3, u, element,
                                     have_nonconservative_terms(equations),
                                     mesh, equations, dg, cache)
end


function calcflux_twopoint_nonconservative!(f1, f2, f3, u::AbstractArray{<:Any,5}, element,
                                            nonconservative_terms::Val{true},
                                            mesh::CurvedMesh{3},
                                            equations, dg::DG, cache)
  #TODO: Create a unified interface, e.g. using non-symmetric two-point (extended) volume fluxes
  #      For now, just dispatch to an existing function for the IdealMhdEquations
  @unpack contravariant_vectors = cache.elements
  calcflux_twopoint_nonconservative!(f1, f2, f3, u, element, contravariant_vectors,
                                     equations, dg, cache)
end


# flux differencing volume integral on curvilinear hexahedral elements. Averaging of the
# mapping terms, stored in `contravariant_vectors`, is peeled apart from the evaluation of
# the physical fluxes in each Cartesian direction
@inline function split_form_kernel!(du::AbstractArray{<:Any,5}, u,
                                    nonconservative_terms::Val{false}, element,
                                    mesh::CurvedMesh{3}, equations, volume_flux, dg::DGSEM, cache,
                                    alpha=true)
  # true * [some floating point value] == [exactly the same floating point value]
  # This can (hopefully) be optimized away due to constant propagation.
  @unpack derivative_split = dg.basis
  @unpack contravariant_vectors = cache.elements

  # Calculate volume integral in one element
  for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
    u_node = get_node_vars(u, equations, dg, i, j, k, element)

    # pull the contravariant vectors in each coordinate direction
    Ja11_node, Ja12_node, Ja13_node = get_contravariant_vector(1, contravariant_vectors,
                                                               i, j, k, element)

    Ja21_node, Ja22_node, Ja23_node = get_contravariant_vector(2, contravariant_vectors,
                                                               i, j, k, element)

    Ja31_node, Ja32_node, Ja33_node = get_contravariant_vector(3, contravariant_vectors,
                                                               i, j, k, element)

    # All diagonal entries of `derivative_split` are zero. Thus, we can skip
    # the computation of the diagonal terms. In addition, we use the symmetry
    # of the `volume_flux` to save half of the possible two-poitn flux
    # computations.

    # x direction
    for ii in (i+1):nnodes(dg)
      u_node_ii = get_node_vars(u, equations, dg, ii, j, k, element)
      flux1 = volume_flux(u_node, u_node_ii, 1, equations)
      flux2 = volume_flux(u_node, u_node_ii, 2, equations)
      flux3 = volume_flux(u_node, u_node_ii, 3, equations)
      # pull the contravariant vectors and compute the average
      Ja11_node_ii, Ja12_node_ii, Ja13_node_ii = get_contravariant_vector(1, contravariant_vectors,
                                                                          ii, j, k, element)
      Ja11_avg = 0.5 * (Ja11_node + Ja11_node_ii)
      Ja12_avg = 0.5 * (Ja12_node + Ja12_node_ii)
      Ja13_avg = 0.5 * (Ja13_node + Ja13_node_ii)
      # compute the contravariant sharp flux
      fluxtilde1 = Ja11_avg * flux1 + Ja12_avg * flux2 + Ja13_avg * flux3
      add_to_node_vars!(du, alpha * derivative_split[i, ii], fluxtilde1, equations, dg, i,  j, k, element)
      add_to_node_vars!(du, alpha * derivative_split[ii, i], fluxtilde1, equations, dg, ii, j, k, element)
    end

    # y direction
    for jj in (j+1):nnodes(dg)
      u_node_jj = get_node_vars(u, equations, dg, i, jj, k, element)
      flux1 = volume_flux(u_node, u_node_jj, 1, equations)
      flux2 = volume_flux(u_node, u_node_jj, 2, equations)
      flux3 = volume_flux(u_node, u_node_jj, 3, equations)
      # pull the contravariant vectors and compute the average
      Ja21_node_jj, Ja22_node_jj, Ja23_node_jj = get_contravariant_vector(2, contravariant_vectors,
                                                                          i, jj, k, element)
      Ja21_avg = 0.5 * (Ja21_node + Ja21_node_jj)
      Ja22_avg = 0.5 * (Ja22_node + Ja22_node_jj)
      Ja23_avg = 0.5 * (Ja23_node + Ja23_node_jj)
      # compute the contravariant sharp flux
      fluxtilde2 = Ja21_avg * flux1 + Ja22_avg * flux2 + Ja23_avg * flux3
      add_to_node_vars!(du, alpha * derivative_split[j, jj], fluxtilde2, equations, dg, i, j,  k, element)
      add_to_node_vars!(du, alpha * derivative_split[jj, j], fluxtilde2, equations, dg, i, jj, k, element)
    end

    # z direction
    for kk in (k+1):nnodes(dg)
      u_node_kk = get_node_vars(u, equations, dg, i, j, kk, element)
      flux1 = volume_flux(u_node, u_node_kk, 1, equations)
      flux2 = volume_flux(u_node, u_node_kk, 2, equations)
      flux3 = volume_flux(u_node, u_node_kk, 3, equations)
      # pull the contravariant vectors and compute the average
      Ja31_node_kk, Ja32_node_kk, Ja33_node_kk = get_contravariant_vector(3, contravariant_vectors,
                                                                          i, j, kk, element)
      Ja31_avg = 0.5 * (Ja31_node + Ja31_node_kk)
      Ja32_avg = 0.5 * (Ja32_node + Ja32_node_kk)
      Ja33_avg = 0.5 * (Ja33_node + Ja33_node_kk)
      # compute the contravariant sharp flux
      fluxtilde3 = Ja31_avg * flux1 + Ja32_avg * flux2 + Ja33_avg * flux3
      add_to_node_vars!(du, alpha * derivative_split[k, kk], fluxtilde3, equations, dg, i, j, k,  element)
      add_to_node_vars!(du, alpha * derivative_split[kk, k], fluxtilde3, equations, dg, i, j, kk, element)
    end
  end
end


function calc_interface_flux!(cache, u, mesh::CurvedMesh{3},
                              nonconservative_terms, # can be Val{true}/Val{false}
                              equations, surface_integral, dg::DG)
  @unpack elements = cache

  @threaded for element in eachelement(dg, cache)
    # Interfaces in negative directions
    # Faster version of "for orientation in (1, 2, 3)"

    # Interfaces in x-direction (`orientation` = 1)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[1, element],
                         element, 1, u, mesh,
                         nonconservative_terms, equations,
                         surface_integral, dg, cache)

    # Interfaces in y-direction (`orientation` = 2)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[2, element],
                         element, 2, u, mesh,
                         nonconservative_terms, equations,
                         surface_integral, dg, cache)

    # Interfaces in z-direction (`orientation` = 3)
    calc_interface_flux!(elements.surface_flux_values,
                         elements.left_neighbors[3, element],
                         element, 3, u, mesh,
                         nonconservative_terms, equations,
                         surface_integral, dg, cache)
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::CurvedMesh{3},
                                      nonconservative_terms::Val{false}, equations,
                                      surface_integral, dg::DG, cache)
  # This is slow for LSA, but for some reason faster for Euler (see #519)
  if left_element <= 0 # left_element = 0 at boundaries
    return surface_flux_values
  end

  @unpack surface_flux = surface_integral
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for j in eachnode(dg), i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, j, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, j, right_element)

      # If the mapping is orientation-reversing, the contravariant vectors' orientation
      # is reversed as well. The normal vector must be oriented in the direction
      # from `left_element` to `right_element`, or the numerical flux will be computed
      # incorrectly (downwind direction).
      sign_jacobian = sign(inverse_jacobian[1, i, j, right_element])

      # First contravariant vector Ja^1 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(1, contravariant_vectors,
                                                                  1, i, j, right_element)
    elseif orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), j, left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          j, right_element)

      # See above
      sign_jacobian = sign(inverse_jacobian[i, 1, j, right_element])

      # Second contravariant vector Ja^2 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(2, contravariant_vectors,
                                                                  i, 1, j, right_element)
    else # orientation == 3
      u_ll = get_node_vars(u, equations, dg, i, j, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, j, 1,          right_element)

      # See above
      sign_jacobian = sign(inverse_jacobian[i, j, 1, right_element])

      # Third contravariant vector Ja^3 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(3, contravariant_vectors,
                                                                  i, j, 1, right_element)
    end

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, j, right_direction, left_element] = flux[v]
      surface_flux_values[v, i, j, left_direction, right_element] = flux[v]
    end
  end

  return nothing
end


@inline function calc_interface_flux!(surface_flux_values, left_element, right_element,
                                      orientation, u,
                                      mesh::CurvedMesh{3},
                                      nonconservative_terms::Val{true}, equations,
                                      surface_integral, dg::DG, cache)
  # See comment on `calc_interface_flux!` with `nonconservative_terms::Val{false}`
  if left_element <= 0 # left_element = 0 at boundaries
    return surface_flux_values
  end

  @unpack surface_flux = surface_integral
  @unpack contravariant_vectors, inverse_jacobian = cache.elements

  right_direction = 2 * orientation
  left_direction = right_direction - 1

  for j in eachnode(dg), i in eachnode(dg)
    if orientation == 1
      u_ll = get_node_vars(u, equations, dg, nnodes(dg), i, j, left_element)
      u_rr = get_node_vars(u, equations, dg, 1,          i, j, right_element)

      # If the mapping is orientation-reversing, the contravariant vectors' orientation
      # is reversed as well. The normal vector must be oriented in the direction
      # from `left_element` to `right_element`, or the numerical flux will be computed
      # incorrectly (downwind direction).
      sign_jacobian = sign(inverse_jacobian[1, i, j, right_element])

      # First contravariant vector Ja^1 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(1, contravariant_vectors,
                                                                  1, i, j, right_element)
    elseif orientation == 2
      u_ll = get_node_vars(u, equations, dg, i, nnodes(dg), j, left_element)
      u_rr = get_node_vars(u, equations, dg, i, 1,          j, right_element)

      # See above
      sign_jacobian = sign(inverse_jacobian[i, 1, j, right_element])

      # Second contravariant vector Ja^2 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(2, contravariant_vectors,
                                                                  i, 1, j, right_element)
    else # orientation == 3
      u_ll = get_node_vars(u, equations, dg, i, j, nnodes(dg), left_element)
      u_rr = get_node_vars(u, equations, dg, i, j, 1,          right_element)

      # See above
      sign_jacobian = sign(inverse_jacobian[i, j, 1, right_element])

      # Third contravariant vector Ja^3 as SVector
      normal_direction = sign_jacobian * get_contravariant_vector(3, contravariant_vectors,
                                                                  i, j, 1, right_element)
    end

    # If the mapping is orientation-reversing, the normal vector will be reversed (see above).
    # However, the flux now has the wrong sign, since we need the physical flux in normal direction.
    flux = sign_jacobian * surface_flux(u_ll, u_rr, normal_direction, equations)

    # Call pointwise nonconservative term; Done twice because left/right orientation matters
    # See Bohm et al. 2018 for details on the nonconservative diamond "flux"
    # Scale with sign_jacobian to ensure that the normal_direction matches that from the flux above
    noncons_primary   = sign_jacobian * noncons_interface_flux(u_ll, u_rr, normal_direction, :weak, equations)
    noncons_secondary = sign_jacobian * noncons_interface_flux(u_rr, u_ll, normal_direction, :weak, equations)

    for v in eachvariable(equations)
      surface_flux_values[v, i, j, right_direction, left_element] = flux[v] + noncons_primary[v]
      surface_flux_values[v, i, j, left_direction, right_element] = flux[v] + noncons_secondary[v]
    end
  end

  return nothing
end


# TODO: Taal dimension agnostic
function calc_boundary_flux!(cache, u, t, boundary_condition::BoundaryConditionPeriodic,
                             mesh::CurvedMesh{3}, equations, surface_integral, dg::DG)
  @assert isperiodic(mesh)
end


function calc_boundary_flux!(cache, u, t, boundary_condition,
                             mesh::CurvedMesh{3}, equations, surface_integral, dg::DG)
  calc_boundary_flux!(cache, u, t,
                      (boundary_condition, boundary_condition, boundary_condition,
                       boundary_condition, boundary_condition, boundary_condition),
                      mesh, equations, surface_integral, dg)
end


function calc_boundary_flux!(cache, u, t, boundary_conditions::Union{NamedTuple,Tuple},
                             mesh::CurvedMesh{3}, equations, surface_integral, dg::DG)
  @unpack surface_flux_values = cache.elements
  linear_indices = LinearIndices(size(mesh))

  for cell_z in axes(mesh, 3), cell_y in axes(mesh, 2)
    # Negative x-direction
    direction = 1
    element = linear_indices[begin, cell_y, cell_z]

    for k in eachnode(dg), j in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (1, j, k), (j, k), element)
    end

    # Positive x-direction
    direction = 2
    element = linear_indices[end, cell_y, cell_z]

    for k in eachnode(dg), j in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 1,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (nnodes(dg), j, k), (j, k), element)
    end
  end

  for cell_z in axes(mesh, 3), cell_x in axes(mesh, 1)
    # Negative y-direction
    direction = 3
    element = linear_indices[cell_x, begin, cell_z]

    for k in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (i, 1, k), (i, k), element)
    end

    # Positive y-direction
    direction = 4
    element = linear_indices[cell_x, end, cell_z]

    for k in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 2,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (i, nnodes(dg), k), (i, k), element)
    end
  end

  for cell_y in axes(mesh, 2), cell_x in axes(mesh, 1)
    # Negative z-direction
    direction = 5
    element = linear_indices[cell_x, cell_y, begin]

    for j in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 3,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (i, j, 1), (i, j), element)
    end

    # Positive z-direction
    direction = 6
    element = linear_indices[cell_x, cell_y, end]

    for j in eachnode(dg), i in eachnode(dg)
      calc_boundary_flux_by_direction!(surface_flux_values, u, t, 3,
                                       boundary_conditions[direction],
                                       mesh, equations, surface_integral, dg, cache,
                                       direction, (i, j, nnodes(dg)), (i, j), element)
    end
  end
end


function apply_jacobian!(du,
                         mesh::CurvedMesh{3},
                         equations, dg::DG, cache)

  @threaded for element in eachelement(dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
      factor = -cache.elements.inverse_jacobian[i, j, k, element]

      for v in eachvariable(equations)
        du[v, i, j, k, element] *= factor
      end
    end
  end

  return nothing
end


end # @muladd
