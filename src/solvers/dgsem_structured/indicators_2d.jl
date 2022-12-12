# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

function apply_smoothing!(mesh::StructuredMesh{2}, alpha, alpha_tmp, dg, cache)
  # Diffuse alpha values by setting each alpha to at least 50% of neighboring elements' alpha
  # Copy alpha values such that smoothing is indpedenent of the element access order
  alpha_tmp .= alpha

  # So far, alpha smoothing doesn't work for non-periodic initial conditions for structured meshes.
  @assert isperiodic(mesh) "alpha smoothing for structured meshes works only with periodic initial conditions so far"

  # Loop over elements, because there is no interface container
  for element in eachelement(dg,cache)
    # Get neighboring element ids
    left  = cache.elements.left_neighbors[1, element]
    lower = cache.elements.left_neighbors[2, element]

    # Apply smoothing
    alpha[left]     = max(alpha_tmp[left],    0.5 * alpha_tmp[element], alpha[left])
    alpha[element]  = max(alpha_tmp[element], 0.5 * alpha_tmp[left],    alpha[element])

    alpha[lower]    = max(alpha_tmp[lower],   0.5 * alpha_tmp[element], alpha[lower])
    alpha[element]  = max(alpha_tmp[element], 0.5 * alpha_tmp[lower],   alpha[element])
  end
end


function calc_bounds_2sided_interface!(var_min, var_max, variable, u, t, semi, mesh::StructuredMesh{2})
  _, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  # Calc bounds at interfaces and periodic boundaries
  for element in eachelement(dg, cache)
    # Get neighboring element ids
    left  = cache.elements.left_neighbors[1, element]
    lower = cache.elements.left_neighbors[2, element]

    if left != 0
      for j in eachnode(dg)
        var_left    = variable(get_node_vars(u, equations, dg, nnodes(dg), j, left), equations)
        var_element = variable(get_node_vars(u, equations, dg, 1,          j, element), equations)

        var_min[1, j, element] = min(var_min[1, j, element], var_left)
        var_max[1, j, element] = max(var_max[1, j, element], var_left)

        var_min[nnodes(dg), j, left] = min(var_min[nnodes(dg), j, left], var_element)
        var_max[nnodes(dg), j, left] = max(var_max[nnodes(dg), j, left], var_element)
      end
    end
    if lower != 0
      for i in eachnode(dg)
        var_lower   = variable(get_node_vars(u, equations, dg, i, nnodes(dg), lower), equations)
        var_element = variable(get_node_vars(u, equations, dg, i, 1,          element), equations)

        var_min[i, 1, element] = min(var_min[i, 1, element], var_lower)
        var_max[i, 1, element] = max(var_max[i, 1, element], var_lower)

        var_min[i, nnodes(dg), lower] = min(var_min[i, nnodes(dg), lower], var_element)
        var_max[i, nnodes(dg), lower] = max(var_max[i, nnodes(dg), lower], var_element)
      end
    end
  end

  # Calc bounds at physical boundaries
  if boundary_conditions isa BoundaryConditionPeriodic
    return nothing
  end
  linear_indices = LinearIndices(size(mesh))
  # - xi direction
  for cell_y in axes(mesh, 2)
    element = linear_indices[begin, cell_y]
    for j in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, 1, j, element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[1],
                                         equations, dg, 1, j, element)
      var_outer = variable(u_outer, equations)

      var_min[1, j, element] = min(var_min[1, j, element], var_outer)
      var_max[1, j, element] = max(var_max[1, j, element], var_outer)
    end
  end
  # + xi direction
  for cell_y in axes(mesh, 2)
    element = linear_indices[end, cell_y]
    for j in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[2],
                                         equations, dg, nnodes(dg), j, element)
      var_outer = variable(u_outer, equations)

      var_min[nnodes(dg), j, element] = min(var_min[nnodes(dg), j, element], var_outer)
      var_max[nnodes(dg), j, element] = max(var_max[nnodes(dg), j, element], var_outer)
    end
  end
  # - eta direction
  for cell_x in axes(mesh, 1)
    element = linear_indices[cell_x, begin]
    for i in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, i, 1, element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[3],
                                         equations, dg, i, 1, element)
      var_outer = variable(u_outer, equations)

      var_min[i, 1, element] = min(var_min[i, 1, element], var_outer)
      var_max[i, 1, element] = max(var_max[i, 1, element], var_outer)
    end
  end
  # - eta direction
  for cell_x in axes(mesh, 1)
    element = linear_indices[cell_x, end]
    for i in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[4],
                                         equations, dg, i, nnodes(dg), element)
      var_outer = variable(u_outer, equations)

      var_min[i, nnodes(dg), element] = min(var_min[i, nnodes(dg), element], var_outer)
      var_max[i, nnodes(dg), element] = max(var_max[i, nnodes(dg), element], var_outer)
    end
  end

  return nothing
end

function calc_bounds_1sided_interface!(var_minmax, minmax, variable, u, t, semi, mesh::StructuredMesh{2})
  _, equations, dg, cache = mesh_equations_solver_cache(semi)
  @unpack boundary_conditions = semi
  # Calc bounds at interfaces and periodic boundaries
  for element in eachelement(dg, cache)
    # Get neighboring element ids
    left  = cache.elements.left_neighbors[1, element]
    lower = cache.elements.left_neighbors[2, element]

    if left != 0
      for j in eachnode(dg)
        var_left    = variable(get_node_vars(u, equations, dg, nnodes(dg), j, left),    equations)
        var_element = variable(get_node_vars(u, equations, dg, 1,          j, element), equations)

        var_minmax[1,          j, element] = minmax(var_minmax[1,          j, element], var_left)
        var_minmax[nnodes(dg), j, left]    = minmax(var_minmax[nnodes(dg), j, left],    var_element)
      end
    end
    if lower != 0
      for i in eachnode(dg)
        var_lower   = variable(get_node_vars(u, equations, dg, i, nnodes(dg), lower),   equations)
        var_element = variable(get_node_vars(u, equations, dg, i, 1,          element), equations)

        var_minmax[i, 1,          element] = minmax(var_minmax[i, 1,          element], var_lower)
        var_minmax[i, nnodes(dg), lower]   = minmax(var_minmax[i, nnodes(dg), lower],   var_element)
      end
    end
  end

  # Calc bounds at physical boundaries
  if boundary_conditions isa BoundaryConditionPeriodic
    return nothing
  end
  linear_indices = LinearIndices(size(mesh))
  # - xi direction
  for cell_y in axes(mesh, 2)
    element = linear_indices[begin, cell_y]
    for j in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, 1, j, element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[1],
                                         equations, dg, 1, j, element)
      var_outer = variable(u_outer, equations)

      var_minmax[1, j, element] = minmax(var_minmax[1, j, element], var_outer)
    end
  end
  # + xi direction
  for cell_y in axes(mesh, 2)
    element = linear_indices[end, cell_y]
    for j in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, nnodes(dg), j, element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[2],
                                         equations, dg, nnodes(dg), j, element)
      var_outer = variable(u_outer, equations)

      var_minmax[nnodes(dg), j, element] = minmax(var_minmax[nnodes(dg), j, element], var_outer)
    end
  end
  # - eta direction
  for cell_x in axes(mesh, 1)
    element = linear_indices[cell_x, begin]
    for i in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, i, 1, element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[3],
                                         equations, dg, i, 1, element)
      var_outer = variable(u_outer, equations)

      var_minmax[i, 1, element] = minmax(var_minmax[i, 1, element], var_outer)
    end
  end
  # - eta direction
  for cell_x in axes(mesh, 1)
    element = linear_indices[cell_x, end]
    for i in eachnode(dg)
      u_inner = get_node_vars(u, equations, dg, i, nnodes(dg), element)
      u_outer = get_boundary_outer_state(u_inner, cache, t, boundary_conditions[4],
                                         equations, dg, i, nnodes(dg), element)
      var_outer = variable(u_outer, equations)

      var_minmax[i, nnodes(dg), element] = minmax(var_minmax[i, nnodes(dg), element], var_outer)
    end
  end

  return nothing
end


@inline function update_alpha_per_timestep!(indicator::IndicatorIDP, timestep, n_stages, semi, mesh::StructuredMesh)
  _, _, solver, cache = mesh_equations_solver_cache(semi)
  @unpack weights = solver.basis
  @unpack alpha_mean_per_timestep, alpha_max_per_timestep= indicator.cache
  @unpack alpha = indicator.cache.ContainerShockCapturingIndicator

  if indicator.indicator_smooth
    elements = cache.element_ids_dgfv
  else
    elements = eachelement(solver, cache)
  end

  alpha_max_per_timestep[timestep] = max(alpha_max_per_timestep[timestep], maximum(alpha))
  alpha_avg = zero(eltype(alpha))
  total_volume = zero(eltype(alpha))
  for element in elements
    for j in eachnode(solver), i in eachnode(solver)
      jacobian = inv(cache.elements.inverse_jacobian[i, j, element])
      alpha_avg += jacobian * weights[i] * weights[j] * alpha[i, j, element]
      total_volume += jacobian * weights[i] * weights[j]
    end
  end
  if total_volume > 0
    alpha_mean_per_timestep[timestep] += 1/(n_stages * total_volume) * alpha_avg
  end

  return nothing
end


end # @muladd
