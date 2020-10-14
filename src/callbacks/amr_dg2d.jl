
# TODO: Taal, dimension agnostic
# Refine elements in the DG solver based on a list of cell_ids that should be refined
function refine!(u_ode::AbstractVector, adaptor, mesh::TreeMesh{2},
                 equations, dg::DGSEM, cache, cells_to_refine)
  # Return early if there is nothing to do
  if isempty(cells_to_refine)
    return
  end

  # Determine for each existing element whether it needs to be refined
  needs_refinement = falses(nelements(dg, cache))

  # The "Ref(...)" is such that we can vectorize the search but not the array that is searched
  elements_to_refine = searchsortedfirst.(Ref(cache.elements.cell_ids[1:nelements(dg, cache)]),
                                          cells_to_refine)
  needs_refinement[elements_to_refine] .= true

  # Retain current solution data
  old_n_elements = nelements(dg, cache)
  old_u_ode = copy(u_ode)
  GC.@preserve old_u_ode begin # OBS! If we don't GC.@preserve old_u_ode, it might be GC'ed
    old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

    # Get new list of leaf cells
    leaf_cell_ids = leaf_cells(mesh.tree)

    # re-initialize elements container
    @unpack elements = cache
    resize!(elements, length(leaf_cell_ids))
    init_elements!(elements, leaf_cell_ids, mesh, dg.basis.nodes)
    @assert nelements(dg, cache) > old_n_elements

    resize!(u_ode, nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
    u = wrap_array(u_ode, mesh, equations, dg, cache)

    # Loop over all elements in old container and either copy them or refine them
    element_id = 1
    for old_element_id in 1:old_n_elements
      if needs_refinement[old_element_id]
        # Refine element and store solution directly in new data structure
        refine_element!(u, element_id, old_u, old_element_id,
                        adaptor, equations, dg)
        element_id += 2^ndims(mesh)
      else
        # Copy old element data to new element container
        @views u[:, .., element_id] .= old_u[:, .., old_element_id]
        element_id += 1
      end
    end
    # If everything is correct, we should have processed all elements.
    # Depending on whether the last element processed above had to be refined or not,
    # the counter `element_id` can have two different values at the end.
    @assert element_id == nelements(dg, cache) + 1 || element_id == nelements(dg, cache) + 2^ndims(mesh) "element_id = $element_id, nelements(dg, cache) = $(nelements(dg, cache))"
  end # GC.@preserve old_u_ode

  # re-initialize interfaces container
  @unpack interfaces = cache
  resize!(interfaces, count_required_interfaces(mesh, leaf_cell_ids))
  init_interfaces!(interfaces, elements, mesh)

  # TODO: Taal performance, allow initializing the stuff in place, making use of resize!
  # Initialize boundaries
  boundaries, _ = init_boundaries(leaf_cell_ids, mesh, elements,
                                  real(dg), nvariables(equations), polydeg(dg))
  copy!(cache.boundaries, boundaries)

  # TODO: Taal performance, allow initializing the stuff in place, making use of resize!
  # Initialize new mortar containers
  mortars = init_mortars(leaf_cell_ids, mesh, elements,
                         real(dg), nvariables(equations), polydeg(dg), dg.mortar)
  copy!(cache.mortars, mortars)

  # Sanity check
  if isperiodic(mesh.tree) && nmortars(mortars) == 0
    @assert ninterfaces(interfaces) == 2 * nelements(dg, cache) ("For 2D and periodic domains and conforming elements, the number of interfaces must be twice the number of elements")
  end

  return nothing
end


# TODO: Taal compare performance of different implementations
# Refine solution data u for an element, using L2 projection (interpolation)
function refine_element!(u::AbstractArray{<:Any,4}, element_id,
                         old_u, old_element_id,
                         adaptor::LobattoLegendreAdaptorL2, equations, dg)
  @unpack forward_upper, forward_lower = adaptor

  # Store new element ids
  lower_left_id  = element_id
  lower_right_id = element_id + 1
  upper_left_id  = element_id + 2
  upper_right_id = element_id + 3

  @boundscheck begin
    @assert old_element_id >= 1
    @assert size(old_u, 1) == nvariables(equations)
    @assert size(old_u, 2) == nnodes(dg)
    @assert size(old_u, 3) == nnodes(dg)
    @assert size(old_u, 4) >= old_element_id
    @assert     element_id >= 1
    @assert size(    u, 1) == nvariables(equations)
    @assert size(    u, 2) == nnodes(dg)
    @assert size(    u, 3) == nnodes(dg)
    @assert size(    u, 4) >= element_id + 3
  end

  # Interpolate to lower left element
  for j in eachnode(dg), i in eachnode(dg)
    acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) * forward_lower[i, k] * forward_lower[j, l]
    end
    set_node_vars!(u, acc, equations, dg, i, j, lower_left_id)
  end

  # Interpolate to lower right element
  for j in eachnode(dg), i in eachnode(dg)
    acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) * forward_upper[i, k] * forward_lower[j, l]
    end
    set_node_vars!(u, acc, equations, dg, i, j, lower_right_id)
  end

  # Interpolate to upper left element
  for j in eachnode(dg), i in eachnode(dg)
    acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) * forward_lower[i, k] * forward_upper[j, l]
    end
    set_node_vars!(u, acc, equations, dg, i, j, upper_left_id)
  end

  # Interpolate to upper right element
  for j in eachnode(dg), i in eachnode(dg)
    acc = zero(get_node_vars(u, equations, dg, i, j, element_id))
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, old_element_id) * forward_upper[i, k] * forward_upper[j, l]
    end
    set_node_vars!(u, acc, equations, dg, i, j, upper_right_id)
  end

  return nothing
end



# TODO: Taal, dimension agnostic
# Coarsen elements in the DG solver based on a list of cell_ids that should be removed
function coarsen!(u_ode::AbstractVector, adaptor, mesh::TreeMesh{2},
                  equations, dg::DGSEM, cache, child_cells_to_coarsen)
  # Return early if there is nothing to do
  if isempty(child_cells_to_coarsen)
    return
  end

  # Determine for each old element whether it needs to be removed
  to_be_removed = falses(nelements(dg, cache))
  # The "Ref(...)" is such that we can vectorize the search but not the array that is searched
  elements_to_remove = searchsortedfirst.(Ref(cache.elements.cell_ids[1:nelements(dg, cache)]),
                                          child_cells_to_coarsen)
  to_be_removed[elements_to_remove] .= true

  # Retain current solution data
  old_n_elements = nelements(dg, cache)
  old_u_ode = copy(u_ode)
  GC.@preserve old_u_ode begin # OBS! If we don't GC.@preserve old_u_ode, it might be GC'ed
    old_u = wrap_array(old_u_ode, mesh, equations, dg, cache)

    # Get new list of leaf cells
    leaf_cell_ids = leaf_cells(mesh.tree)

    # re-initialize elements container
    @unpack elements = cache
    resize!(elements, length(leaf_cell_ids))
    init_elements!(elements, leaf_cell_ids, mesh, dg.basis.nodes)
    @assert nelements(dg, cache) < old_n_elements

    resize!(u_ode, nvariables(equations) * nnodes(dg)^ndims(mesh) * nelements(dg, cache))
    u = wrap_array(u_ode, mesh, equations, dg, cache)

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
        @assert all(to_be_removed[old_element_id:(old_element_id+2^ndims(mesh)-1)]) "bad cell/element order"

        # Coarsen elements and store solution directly in new data structure
        coarsen_elements!(u, element_id, old_u, old_element_id,
                          adaptor, equations, dg)
        element_id += 1
        skip = 2^ndims(mesh) - 1
      else
        # Copy old element data to new element container
        @views u[:, .., element_id] .= old_u[:, .., old_element_id]
        element_id += 1
      end
    end
    # If everything is correct, we should have processed all elements.
    @assert element_id == nelements(dg, cache) + 1 "element_id = $element_id, nelements(dg, cache) = $(nelements(dg, cache))"
  end # GC.@preserve old_u_ode

  # re-initialize interfaces container
  @unpack interfaces = cache
  resize!(interfaces, count_required_interfaces(mesh, leaf_cell_ids))
  init_interfaces!(interfaces, elements, mesh)

  # TODO: Taal performance, allow initializing the stuff in place, making use of resize!
  # Initialize boundaries
  boundaries, _ = init_boundaries(leaf_cell_ids, mesh, elements,
                                  real(dg), nvariables(equations), polydeg(dg))
  copy!(cache.boundaries, boundaries)

  # TODO: Taal performance, allow initializing the stuff in place, making use of resize!
  # Initialize new mortar containers
  mortars = init_mortars(leaf_cell_ids, mesh, elements,
                         real(dg), nvariables(equations), polydeg(dg), dg.mortar)
  copy!(cache.mortars, mortars)

  # Sanity check
  if isperiodic(mesh.tree) && nmortars(mortars) == 0
    @assert ninterfaces(interfaces) == 2 * nelements(dg, cache) ("For 2D and periodic domains and conforming elements, the number of interfaces must be twice the number of elements")
  end

  return nothing
end


# TODO: Taal compare performance of different implementations
# Coarsen solution data u for four elements, using L2 projection
function coarsen_elements!(u::AbstractArray{<:Any,4}, element_id, old_u, old_element_id,
                           adaptor::LobattoLegendreAdaptorL2, equations, dg)
  @unpack reverse_upper, reverse_lower = adaptor

  # Store old element ids
  lower_left_id  = old_element_id
  lower_right_id = old_element_id + 1
  upper_left_id  = old_element_id + 2
  upper_right_id = old_element_id + 3

  @boundscheck begin
    @assert old_element_id >= 1
    @assert size(old_u, 1) == nvariables(equations)
    @assert size(old_u, 2) == nnodes(dg)
    @assert size(old_u, 3) == nnodes(dg)
    @assert size(old_u, 4) >= old_element_id + 3
    @assert     element_id >= 1
    @assert size(    u, 1) == nvariables(equations)
    @assert size(    u, 2) == nnodes(dg)
    @assert size(    u, 3) == nnodes(dg)
    @assert size(    u, 4) >= element_id
  end

  for j in eachnode(dg), i in eachnode(dg)
    acc = zero(get_node_vars(u, equations, dg, i, j, element_id))

    # Project from lower left element
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, lower_left_id) * reverse_lower[i, k] * reverse_lower[j, l]
    end

    # Project from lower right element
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, lower_right_id) * reverse_upper[i, k] * reverse_lower[j, l]
    end

    # Project from upper left element
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, upper_left_id) * reverse_lower[i, k] * reverse_upper[j, l]
    end

    # Project from upper right element
    for l in eachnode(dg), k in eachnode(dg)
      acc += get_node_vars(old_u, equations, dg, k, l, upper_right_id) * reverse_upper[i, k] * reverse_upper[j, l]
    end

    # Update value
    set_node_vars!(u, acc, equations, dg, i, j, element_id)
  end
end


# this method is called when an `ControllerThreeLevel` is constructed
function create_cache(::Type{ControllerThreeLevel}, mesh::TreeMesh{2}, equations, dg::DG, cache)

  controller_value = Vector{Int}(undef, nelements(dg, cache))
  return (; controller_value)
end

