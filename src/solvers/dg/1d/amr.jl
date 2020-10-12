# This file contains functions that are related to the AMR capabilities of the DG solver

# Refine elements in the DG solver based on a list of cell_ids that should be refined
function refine!(dg::Dg1D{Eqn, MeshType, NVARS, POLYDEG}, mesh::TreeMesh,
                 cells_to_refine::AbstractArray{Int}) where {Eqn, MeshType, NVARS, POLYDEG}
  # Return early if there is nothing to do
  if isempty(cells_to_refine)
    return
  end

  # Determine for each existing element whether it needs to be refined
  needs_refinement = falses(nelements(dg.elements))
  tree = mesh.tree
  # The "Ref(...)" is such that we can vectorize the search but not the array that is searched
  elements_to_refine = searchsortedfirst.(Ref(dg.elements.cell_ids[1:nelements(dg.elements)]),
                                          cells_to_refine)
  needs_refinement[elements_to_refine] .= true

  # Retain current solution data
  old_n_elements = nelements(dg.elements)
  old_u = dg.elements.u

  # Get new list of leaf cells
  leaf_cell_ids = leaf_cells(tree)

  # Initialize new elements container
  elements = init_elements(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG))
  n_elements = nelements(elements)

  # Loop over all elements in old container and either copy them or refine them
  element_id = 1
  for old_element_id in 1:old_n_elements
    if needs_refinement[old_element_id]
      # Refine element and store solution directly in new data structure
      refine_element!(elements.u, element_id, old_u, old_element_id,
                      dg.amr_refine_right, dg.amr_refine_left, dg)
      element_id += 2^ndims(dg)
    else
      # Copy old element data to new element container
      @views elements.u[:, :, element_id] .= old_u[:, :, old_element_id]
      element_id += 1
    end
  end

  # Initialize new interfaces container
  interfaces = init_interfaces(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements)
  n_interfaces = ninterfaces(interfaces)

  # Initialize boundaries
  boundaries, n_boundaries_per_direction = init_boundaries(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements)
  n_boundaries = nboundaries(boundaries)



  # Sanity check
  if isperiodic(mesh.tree)
    @assert n_interfaces == 1*n_elements ("For 1D and periodic domains, n_surf must be the same as 1*n_elem")
  end

  # Update DG instance with new data
  dg.elements = elements
  dg.n_elements = n_elements
  dg.n_elements_global = n_elements
  dg.interfaces = interfaces
  dg.n_interfaces = n_interfaces
  dg.boundaries = boundaries
  dg.n_boundaries = n_boundaries
  dg.n_boundaries_per_direction = n_boundaries_per_direction
end


# Refine solution data u for an element, using L2 projection (interpolation)
function refine_element!(u, element_id, old_u, old_element_id,
                        refine_right, refine_left, dg::Dg1D)
  # Store new element ids
  left_id  = element_id
  right_id = element_id + 1

  # Interpolate to left element
  for i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, element_id))
    for k in 1:nnodes(dg)
      acc += get_node_vars(old_u, dg, k, old_element_id) * refine_left[i, k]
    end
    set_node_vars!(u, acc, dg, i, left_id)
  end

  # Interpolate to right element
  for i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, element_id))
    for k in 1:nnodes(dg)
      acc += get_node_vars(old_u, dg, k, old_element_id)  * refine_right[i, k]
    end
    set_node_vars!(u, acc, dg, i, right_id)
  end

end


# Coarsen elements in the DG solver based on a list of cell_ids that should be removed
function coarsen!(dg::Dg1D{Eqn, MeshType, NVARS, POLYDEG}, mesh::TreeMesh,
                  child_cells_to_coarsen::AbstractArray{Int}) where {Eqn, MeshType, NVARS, POLYDEG}
  # Return early if there is nothing to do
  if isempty(child_cells_to_coarsen)
    return
  end

  # Determine for each old element whether it needs to be removed
  to_be_removed = falses(nelements(dg.elements))
  # The "Ref(...)" is such that we can vectorize the search but not the array that is searched
  elements_to_remove = searchsortedfirst.(Ref(dg.elements.cell_ids[1:nelements(dg.elements)]),
                                          child_cells_to_coarsen)
  to_be_removed[elements_to_remove] .= true

  # Retain current solution data
  old_n_elements = nelements(dg.elements)
  old_u = dg.elements.u

  # Get new list of leaf cells
  leaf_cell_ids = leaf_cells(mesh.tree)

  # Initialize new elements container
  elements = init_elements(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG))
  n_elements = nelements(elements)

  # Loop over all elements in old container and either copy them or coarsen them
  skip = 0
  element_id = 1
  for old_element_id in 1:old_n_elements
    # If skip is non-zero, we just coarsened 2^ndims elements and need to omit the following elements
    if skip > 0
      skip -= 1
      continue
    end

    if to_be_removed[old_element_id]
      # If an element is to be removed, sanity check if the following elements
      # are also marked - otherwise there would be an error in the way the
      # cells/elements are sorted
      @assert all(to_be_removed[old_element_id:(old_element_id+2^ndims(dg)-1)]) "bad cell/element order"

      # Coarsen elements and store solution directly in new data structure
      coarsen_elements!(elements.u, element_id, old_u, old_element_id,
                        dg.amr_coarsen_right, dg.amr_coarsen_left, dg)
      element_id += 1
      skip = 2^ndims(dg) - 1
    else
      # Copy old element data to new element container
      @views elements.u[:, :, element_id] .= old_u[:, :, old_element_id]
      element_id += 1
    end
  end

  # Initialize new interfaces container
  interfaces = init_interfaces(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements)
  n_interfaces = ninterfaces(interfaces)

  # Initialize boundaries
  boundaries, n_boundaries_per_direction = init_boundaries(leaf_cell_ids, mesh, Val(NVARS), Val(POLYDEG), elements)
  n_boundaries = nboundaries(boundaries)

  # Sanity check
  if isperiodic(mesh.tree)
    @assert n_interfaces == 1*n_elements ("For 1D and periodic domains, n_surf must be the same as 1*n_elem")
  end

  # Update DG instance with new data
  dg.elements = elements
  dg.n_elements = n_elements
  dg.n_elements_global = n_elements
  dg.interfaces = interfaces
  dg.n_interfaces = n_interfaces
  dg.boundaries = boundaries
  dg.n_boundaries = n_boundaries
  dg.n_boundaries_per_direction = n_boundaries_per_direction
end


# Coarsen solution data u for four elements, using L2 projection
function coarsen_elements!(u, element_id, old_u, old_element_id,
                            coarsen_right, coarsen_left, dg::Dg1D)
  # Store old element ids
  left_id  = old_element_id
  right_id = old_element_id + 1

  for i in 1:nnodes(dg)
    acc = zero(get_node_vars(u, dg, i, element_id))

    # Project from left element
    for k in 1:nnodes(dg)
      acc += get_node_vars(old_u, dg, k, left_id) * coarsen_left[i, k]
    end

    # Project from right element
    for k in 1:nnodes(dg)
      acc += get_node_vars(old_u, dg, k, right_id) * coarsen_right[i, k]
    end

    # Update value
    set_node_vars!(u, acc, dg, i, element_id)
  end
end


# Calculate an AMR indicator value for each element/leaf cell
#
# The indicator value λ ∈ [-1,1] is ≈ -1 for cells that should be coarsened, ≈
# 0 for cells that should remain as-is, and ≈ 1 for cells that should be
# refined.
#
# Note: The implementation here implicitly assumes that we have an element for
# each leaf cell and that they are in the same order.
#
# FIXME: This is currently implemented for each test case - we need something
# appropriate that is both equation and test case independent
function calc_amr_indicator(dg::Dg1D, mesh::TreeMesh, time)
  lambda = zeros(dg.n_elements)

  if dg.amr_indicator === :gauss
    base_level = 4
    max_level = 6
    threshold_high = 0.6
    threshold_low = 0.1

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      # Determine target level from peak value
      peak = maximum(dg.elements.u[:, :, element_id])
      if peak > threshold_high
        target_level = max_level
      elseif peak > threshold_low
        target_level = max_level - 1
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  elseif dg.amr_indicator === :blast_wave
    base_level = 4
    max_level = 6
    blending_factor_threshold = 0.01

    # (Re-)initialize element variable storage for blending factor
    if (!haskey(dg.element_variables, :amr_indicator_values) ||
        length(dg.element_variables[:amr_indicator_values]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values] = Vector{Float64}(undef, dg.n_elements)
    end
    if (!haskey(dg.element_variables, :amr_indicator_values_tmp) ||
        length(dg.element_variables[:amr_indicator_values_tmp]) != dg.n_elements)
      dg.element_variables[:amr_indicator_values_tmp] = Vector{Float64}(undef, dg.n_elements)
    end

    alpha     = dg.element_variables[:amr_indicator_values]
    alpha_tmp = dg.element_variables[:amr_indicator_values_tmp]
    calc_blending_factors!(alpha, alpha_tmp, dg.elements.u, dg.amr_alpha_max, dg.amr_alpha_min, dg.amr_alpha_smooth,
                           density_pressure, dg.thread_cache, dg)

    # Iterate over all elements
    for element_id in 1:dg.n_elements
      if alpha[element_id] > blending_factor_threshold
        target_level = max_level
      else
        target_level = base_level
      end

      # Compare target level with actual level to set indicator
      cell_id = dg.elements.cell_ids[element_id]
      actual_level = mesh.tree.levels[cell_id]
      if actual_level < target_level
        lambda[element_id] = 1.0
      elseif actual_level > target_level
        lambda[element_id] = -1.0
      else
        lambda[element_id] = 0.0
      end
    end
  else
    error("unknown AMR indicator '$(dg.amr_indicator)'")
  end

  return lambda
end
