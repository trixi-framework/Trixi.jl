# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Find element ids containing coordinates given as a matrix [ndims, npoints]
function get_elements_by_coordinates!(element_ids, coordinates, mesh::TreeMesh, dg,
                                      cache)
    if length(element_ids) != size(coordinates, 2)
        throw(DimensionMismatch("storage length for element ids does not match the number of coordinates"))
    end

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
        for index in eachindex(element_ids)
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

# Calculate the interpolating polynomials to extract data at the given coordinates
# The coordinates are known to be located in the respective element in `element_ids`
function calc_interpolating_polynomials!(interpolating_polynomials, coordinates,
                                         element_ids,
                                         mesh::TreeMesh, dg::DGSEM, cache)
    @unpack tree = mesh
    @unpack nodes = dg.basis

    wbary = barycentric_weights(nodes)

    for index in eachindex(element_ids)
        # Construct point
        x = SVector(ntuple(i -> coordinates[i, index], ndims(mesh)))

        # Convert to unit coordinates
        cell_id = cache.elements.cell_ids[element_ids[index]]
        cell_coordinates_ = cell_coordinates(tree, cell_id)
        cell_length = length_at_cell(tree, cell_id)
        unit_coordinates = (x .- cell_coordinates_) * 2 / cell_length

        # Calculate interpolating polynomial for each dimension, making use of tensor product structure
        for d in 1:ndims(mesh)
            interpolating_polynomials[:, d, index] .= lagrange_interpolating_polynomials(unit_coordinates[d],
                                                                                         nodes,
                                                                                         wbary)
        end
    end

    return interpolating_polynomials
end

# Record the solution variables at each given point for the 1D case
function record_state_at_points!(point_data, u, solution_variables,
                                 n_solution_variables,
                                 mesh::TreeMesh{1}, equations, dg::DG,
                                 time_series_cache)
    @unpack element_ids, interpolating_polynomials = time_series_cache
    old_length = length(first(point_data))
    new_length = old_length + n_solution_variables

    # Loop over all points/elements that should be recorded
    for index in eachindex(element_ids)
        # Extract data array and element id
        data = point_data[index]
        element_id = element_ids[index]

        # Make room for new data to be recorded
        resize!(data, new_length)
        data[(old_length + 1):new_length] .= zero(eltype(data))

        # Loop over all nodes to compute their contribution to the interpolated values
        for i in eachnode(dg)
            u_node = solution_variables(get_node_vars(u, equations, dg, i,
                                                      element_id), equations)

            for v in eachindex(u_node)
                data[old_length + v] += (u_node[v] *
                                         interpolating_polynomials[i, 1, index])
            end
        end
    end
end

# Record the solution variables at each given point for the 2D case
function record_state_at_points!(point_data, u, solution_variables,
                                 n_solution_variables,
                                 mesh::TreeMesh{2},
                                 equations, dg::DG, time_series_cache)
    @unpack element_ids, interpolating_polynomials = time_series_cache
    old_length = length(first(point_data))
    new_length = old_length + n_solution_variables

    # Loop over all points/elements that should be recorded
    for index in eachindex(element_ids)
        # Extract data array and element id
        data = point_data[index]
        element_id = element_ids[index]

        # Make room for new data to be recorded
        resize!(data, new_length)
        data[(old_length + 1):new_length] .= zero(eltype(data))

        # Loop over all nodes to compute their contribution to the interpolated values
        for j in eachnode(dg), i in eachnode(dg)
            u_node = solution_variables(get_node_vars(u, equations, dg, i, j,
                                                      element_id), equations)

            for v in eachindex(u_node)
                data[old_length + v] += (u_node[v]
                                         * interpolating_polynomials[i, 1, index]
                                         * interpolating_polynomials[j, 2, index])
            end
        end
    end
end

# Record the solution variables at each given point for the 3D case
function record_state_at_points!(point_data, u, solution_variables,
                                 n_solution_variables,
                                 mesh::TreeMesh{3}, equations, dg::DG,
                                 time_series_cache)
    @unpack element_ids, interpolating_polynomials = time_series_cache
    old_length = length(first(point_data))
    new_length = old_length + n_solution_variables

    # Loop over all points/elements that should be recorded
    for index in eachindex(element_ids)
        # Extract data array and element id
        data = point_data[index]
        element_id = element_ids[index]

        # Make room for new data to be recorded
        resize!(data, new_length)
        data[(old_length + 1):new_length] .= zero(eltype(data))

        # Loop over all nodes to compute their contribution to the interpolated values
        for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
            u_node = solution_variables(get_node_vars(u, equations, dg, i, j, k,
                                                      element_id), equations)

            for v in eachindex(u_node)
                data[old_length + v] += (u_node[v]
                                         * interpolating_polynomials[i, 1, index]
                                         * interpolating_polynomials[j, 2, index]
                                         * interpolating_polynomials[k, 3, index])
            end
        end
    end
end
end # @muladd
