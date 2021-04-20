
function create_cache_timeseries(point_coordinates, mesh::TreeMesh{2}, dg, cache)
  # Determine element ids for point coordinates
  element_ids = get_elements_by_coordinates(point_coordinates, mesh, dg, cache)

  # Calculate & store Lagrange interpolation polynomials
  interpolating_polynomials = calc_interpolating_polynomials(point_coordinates, element_ids, mesh,
                                                             dg, cache)

  timeseries_cache = (; element_ids, interpolating_polynomials)

  return timeseries_cache
end


function get_elements_by_coordinates!(element_ids, coordinates, mesh::TreeMesh, dg, cache)
  @assert length(element_ids) == size(coordinates, 2) "size mismatch"

  @unpack cell_ids = cache.elements
  @unpack tree = mesh

  # Reset element ids - 0 indicates "not (yet) found"
  element_ids .= 0
  found_elements = 0

  # Iterate over all elements
  for element in eachelement(dg, cache)
    # Get cell id
    cell_id = cell_ids[element]

    # Iterate over coordinates
    for index in 1:length(element_ids)
      # Skip coordinates for which an element has already been found
      if element_ids[index] > 0
        continue
      end

      # Construct point
      x = SVector(ntuple(i -> coordinates[i, index], ndims(mesh)))

      # Skip if point is not in cell
      if !is_point_in_cell(tree, x, cell_id)
        continue
      end

      # Otherwise point is in cell and thus in element
      element_ids[index] = element
      found_elements += 1
    end

    # Exit loop if all elements have already been found
    if found_elements == length(element_ids)
      break
    end
  end

  return element_ids
end


function get_elements_by_coordinates(coordinates, mesh, dg, cache)
  element_ids = Vector{Int}(undef, size(coordinates, 2))
  get_elements_by_coordinates!(element_ids, coordinates, mesh, dg, cache)

  return element_ids
end


function calc_interpolating_polynomials!(interpolating_polynomials, coordinates, element_ids,
                                         mesh::TreeMesh, dg, cache)
  @unpack tree = mesh
  @unpack nodes = dg.basis

  wbary = barycentric_weights(nodes)

  for index in 1:length(element_ids)
    # Construct point
    x = SVector(ntuple(i -> coordinates[i, index], ndims(mesh)))

    # Convert to unit coordinates
    cell_id = cache.elements.cell_ids[element_ids[index]]
    cell_coordinates_ = cell_coordinates(tree, cell_id)
    cell_length = length_at_cell(tree, cell_id)
    unit_coordinates = (x .- cell_coordinates_) * 2 / cell_length

    for d in 1:ndims(mesh)
      interpolating_polynomials[:, d, index] .= lagrange_interpolating_polynomials(x[d], nodes, wbary)
    end
  end

  return interpolating_polynomials
end


function calc_interpolating_polynomials(coordinates, element_ids, mesh::TreeMesh, dg, cache)
  interpolating_polynomials = Array{real(dg), 3}(undef,
                                                 nnodes(dg), ndims(mesh), length(element_ids))
  calc_interpolating_polynomials!(interpolating_polynomials, coordinates, element_ids, mesh, dg,
                                  cache)

  return interpolating_polynomials
end


function record_state_at_points!(point_data, u, solution_variables, n_solution_variables,
                                 mesh::TreeMesh{2}, equations, dg::DG, timeseries_cache)
  @unpack element_ids, interpolating_polynomials = timeseries_cache
  old_length = length(first(point_data))
  new_length = old_length + n_solution_variables

  for index in 1:length(element_ids)
    data = point_data[index]
    element_id = element_ids[index]

    resize!(data, new_length)
    data[(old_length+1):new_length] .= zero(eltype(data))

    for j in eachnode(dg), i in eachnode(dg)
      u_node = solution_variables(get_node_vars(u, equations, dg, i, j, element_id), equations)

      for v in 1:length(u_node)
        data[old_length + v] += (u_node[v]
                                * interpolating_polynomials[i, 1, index]
                                * interpolating_polynomials[j, 2, index])
      end
    end
  end
end
