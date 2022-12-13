# !!! warning "Experimental implementation (upwind SBP)"
#     This is an experimental feature and may change in future releases.

# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


# 2D unstructured cache
function create_cache(mesh::UnstructuredMesh2D, equations, dg::FDSBP, RealT, uEltype)

  elements = init_elements(mesh, equations, dg.basis, RealT, uEltype)

  interfaces = init_interfaces(mesh, elements)

  boundaries = init_boundaries(mesh, elements)

  cache = (; elements, interfaces, boundaries)

  # Add specialized parts of the cache required to for efficient flux computations
  cache = (;cache..., create_cache(mesh, equations, dg.volume_integral, dg, uEltype)...)

  return cache
end


# TODO: FD; Upwind versions of surface / volume integral

# 2D volume integral contributions for `VolumeIntegralStrongForm`
@inline function calc_volume_integral!(du, u,
                                       mesh::UnstructuredMesh2D,
                                       nonconservative_terms::Val{false}, equations,
                                       volume_integral::VolumeIntegralStrongForm,
                                       dg::FDSBP, cache)
  # D = dg.basis # SBP derivative operator
  # @unpack f_threaded = cache

  # # SBP operators from SummationByPartsOperators.jl implement the basic interface
  # # of matrix-vector multiplication. Thus, we pass an "array of structures",
  # # packing all variables per node in an `SVector`.
  # if nvariables(equations) == 1
  #   # `reinterpret(reshape, ...)` removes the leading dimension only if more
  #   # than one variable is used.
  #   u_vectors  = reshape(reinterpret(SVector{nvariables(equations), eltype(u)}, u),
  #                        nnodes(dg), nnodes(dg), nelements(dg, cache))
  #   du_vectors = reshape(reinterpret(SVector{nvariables(equations), eltype(du)}, du),
  #                        nnodes(dg), nnodes(dg), nelements(dg, cache))
  # else
  #   u_vectors  = reinterpret(reshape, SVector{nvariables(equations), eltype(u)}, u)
  #   du_vectors = reinterpret(reshape, SVector{nvariables(equations), eltype(du)}, du)
  # end

  # # Use the tensor product structure to compute the discrete derivatives of
  # # the contravariant fluxes line-by-line and add them to `du` for each element.
  # @threaded for element in eachelement(dg, cache)
  #   f_element = f_threaded[Threads.threadid()]
  #   u_element = view(u_vectors,  :, :, element)

  #   # x direction
  #   @. f_element = flux(u_element, 1, equations)
  #   for j in eachnode(dg)
  #     mul!(view(du_vectors, :, j, element), D, view(f_element, :, j),
  #          one(eltype(du)), one(eltype(du)))
  #   end

  #   # y direction
  #   @. f_element = flux(u_element, 2, equations)
  #   for i in eachnode(dg)
  #     mul!(view(du_vectors, i, :, element), D, view(f_element, i, :),
  #          one(eltype(du)), one(eltype(du)))
  #   end
  # end



  # Pull the derivative matrix
  # TODO: FD, improve performance to use `mul!`. Current version is slow and allocates
  D = Matrix(dg.basis) # SBP derivative operator
  @unpack contravariant_vectors = cache.elements

  @threaded for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      u_node = get_node_vars(u, equations, dg, i, j, element)

      flux1 = flux(u_node, 1, equations)
      flux2 = flux(u_node, 2, equations)

      # Compute the contravariant flux by taking the scalar product of the
      # first contravariant vector Ja^1 and the flux vector
      Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j, element)
      contravariant_flux1 = Ja11 * flux1 + Ja12 * flux2
      for ii in eachnode(dg)
        multiply_add_to_node_vars!(du, D[ii, i], contravariant_flux1, equations, dg, ii, j, element)
      end

      # Compute the contravariant flux by taking the scalar product of the
      # second contravariant vector Ja^2 and the flux vector
      Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j, element)
      contravariant_flux2 = Ja21 * flux1 + Ja22 * flux2
      for jj in eachnode(dg)
        multiply_add_to_node_vars!(du, D[jj, j], contravariant_flux2, equations, dg, i, jj, element)
      end
    end
  end

  return nothing
end


# Note! The local side numbering for the unstructured quadrilateral element implementation differs
#       from the structured TreeMesh or StructuredMesh local side numbering:
#
#      TreeMesh/StructuredMesh sides   versus   UnstructuredMesh sides
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
# Also, the normal directions are already outward pointing for `UnstructuredMesh2D` so all the
# surface contributions are added.
function calc_surface_integral!(du, u, mesh::UnstructuredMesh2D,
                                equations, surface_integral::SurfaceIntegralStrongForm,
                                dg::DG, cache)
  inv_weight_left  = inv(left_boundary_weight(dg.basis))
  inv_weight_right = inv(right_boundary_weight(dg.basis))
  @unpack normal_directions, surface_flux_values = cache.elements

  @threaded for element in eachelement(dg, cache)
    for l in eachnode(dg)
      # surface at -x
      u_node = get_node_vars(u, equations, dg, 1, l, element)
      # compute internal flux in normal direction on side 4
      outward_direction = get_node_coords(normal_directions, equations, dg, l, 4, element)
      f_node = flux(u_node, outward_direction, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 4, element)
      multiply_add_to_node_vars!(du, inv_weight_left, (f_num - f_node),
                                 equations, dg, 1, l, element)

      # surface at +x
      u_node = get_node_vars(u, equations, dg, nnodes(dg), l, element)
      # compute internal flux in normal direction on side 2
      outward_direction = get_node_coords(normal_directions, equations, dg, l, 2, element)
      f_node = flux(u_node, outward_direction, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 2, element)
      multiply_add_to_node_vars!(du, inv_weight_right, (f_num - f_node),
                                 equations, dg, nnodes(dg), l, element)

      # surface at -y
      u_node = get_node_vars(u, equations, dg, l, 1, element)
      # compute internal flux in normal direction on side 1
      outward_direction = get_node_coords(normal_directions, equations, dg, l, 1, element)
      f_node = flux(u_node, outward_direction, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 1, element)
      multiply_add_to_node_vars!(du, inv_weight_left, (f_num - f_node),
                                 equations, dg, l, 1, element)

      # surface at +y
      u_node = get_node_vars(u, equations, dg, l, nnodes(dg), element)
      # compute internal flux in normal direction on side 3
      outward_direction = get_node_coords(normal_directions, equations, dg, l, 3, element)
      f_node = flux(u_node, outward_direction, equations)
      f_num  = get_node_vars(surface_flux_values, equations, dg, l, 3, element)
      multiply_add_to_node_vars!(du, inv_weight_right, (f_num - f_node),
                                 equations, dg, l, nnodes(dg), element)
    end
  end

  return nothing
end


# AnalysisCallback
function integrate_via_indices(func::Func, u,
                               mesh::UnstructuredMesh2D, equations,
                               dg::FDSBP, cache, args...; normalize=true) where {Func}
  # TODO: FD. This is rather inefficient right now and allocates...
  weights = diag(SummationByPartsOperators.mass_matrix(dg.basis))

  # Initialize integral with zeros of the right shape
  integral = zero(func(u, 1, 1, 1, equations, dg, args...))
  total_volume = zero(real(mesh))

  # Use quadrature to numerically integrate over entire domain
  for element in eachelement(dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
      volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
      integral += volume_jacobian * weights[i] * weights[j] * func(u, i, j, element, equations, dg, args...)
      total_volume += volume_jacobian * weights[i] * weights[j]
    end
  end

  # Normalize with total volume
  if normalize
    integral = integral / total_volume
  end

  return integral
end


function calc_error_norms(func, u, t, analyzer,
                          mesh::UnstructuredMesh2D, equations, initial_condition,
                          dg::FDSBP, cache, cache_analysis)
  # TODO: FD. This is rather inefficient right now and allocates...
  weights = diag(SummationByPartsOperators.mass_matrix(dg.basis))
  @unpack node_coordinates, inverse_jacobian = cache.elements

  # Set up data structures
  l2_error   = zero(func(get_node_vars(u, equations, dg, 1, 1, 1), equations))
  linf_error = copy(l2_error)
  total_volume = zero(real(mesh))

  # Iterate over all elements for error calculations
  for element in eachelement(dg, cache)
    for j in eachnode(analyzer), i in eachnode(analyzer)
      volume_jacobian = abs(inv(cache.elements.inverse_jacobian[i, j, element]))
      u_exact = initial_condition(
        get_node_coords(node_coordinates, equations, dg, i, j, element), t, equations)
      diff = func(u_exact, equations) - func(
        get_node_vars(u, equations, dg, i, j, element), equations)
      l2_error += diff.^2 * (weights[i] * weights[j] * volume_jacobian)
      linf_error = @. max(linf_error, abs(diff))
      total_volume += weights[i] * weights[j] * volume_jacobian
    end
  end

  # For L2 error, divide by total volume
  l2_error = @. sqrt(l2_error / total_volume)

  return l2_error, linf_error
end

end # @muladd
