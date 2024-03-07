# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Creates cache for time series callback
function create_cache_time_series(point_coordinates,
                                  mesh::Union{TreeMesh{2}, UnstructuredMesh2D},
                                  dg, cache)
    # Determine element ids for point coordinates
    element_ids = get_elements_by_coordinates(point_coordinates, mesh, dg, cache)

    # Calculate & store Lagrange interpolation polynomials
    interpolating_polynomials = calc_interpolating_polynomials(point_coordinates,
                                                               element_ids, mesh,
                                                               dg, cache)

    time_series_cache = (; element_ids, interpolating_polynomials)

    return time_series_cache
end

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

function get_elements_by_coordinates!(element_ids, coordinates,
                                      mesh::UnstructuredMesh2D,
                                      dg, cache)
    if length(element_ids) != size(coordinates, 2)
        throw(DimensionMismatch("storage length for element ids does not match the number of coordinates"))
    end

    # Reset element ids - 0 indicates "not (yet) found"
    element_ids .= 0
    found_elements = 0

    # Iterate over all elements
    for element in eachelement(dg, cache)

        # Iterate over coordinates
        for index in 1:length(element_ids)
            # Skip coordinates for which an element has already been found
            if element_ids[index] > 0
                continue
            end

            # Construct point
            x = SVector(ntuple(i -> coordinates[i, index], ndims(mesh)))

            # Skip if point is not in the current element
            if !is_point_in_quad(mesh, x, element)
                continue
            end

            # Otherwise point is in the current element
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

# For the `UnstructuredMesh2D` this uses a simple method assuming a convex
# quadrilateral. It simply checks that the point lies on the "correct" side
# of each of the quadrilateral's edges (basically a ray casting strategy).
# Does not account for element curvature.
# OBS! One possibility for a more robust (and maybe faster) algorithm would
# be to replace this function with `inpolygon` from PolygonOps.jl
@inline function is_point_in_quad(mesh::UnstructuredMesh2D, point, element)
    # Helper array for the current quadrilateral element
    corners = zeros(eltype(mesh.corners), 2, 4)

    # Grab the four corners
    for j in 1:2, i in 1:4
        # pull the (x,y) values of these corners out of the global corners array
        corners[j, i] = mesh.corners[j, mesh.element_node_ids[i, element]]
    end

    if cross_product_2d(corners[:, 2] .- corners[:, 1], point .- corners[:, 1]) > 0 &&
       cross_product_2d(corners[:, 3] .- corners[:, 2], point .- corners[:, 2]) > 0 &&
       cross_product_2d(corners[:, 4] .- corners[:, 3], point .- corners[:, 3]) > 0 &&
       cross_product_2d(corners[:, 1] .- corners[:, 4], point .- corners[:, 4]) > 0
        return true
    else
        return false
    end
end

# 2D cross product
@inline function cross_product_2d(u, v)
    return u[1] * v[2] - u[2] * v[1]
end

function get_elements_by_coordinates(coordinates, mesh, dg, cache)
    element_ids = Vector{Int}(undef, size(coordinates, 2))
    get_elements_by_coordinates!(element_ids, coordinates, mesh, dg, cache)

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

    for index in 1:length(element_ids)
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

function calc_interpolating_polynomials!(interpolating_polynomials, coordinates,
                                         element_ids,
                                         mesh::UnstructuredMesh2D, dg::DGSEM, cache)
    @unpack nodes = dg.basis

    wbary = barycentric_weights(nodes)

    # Helper array for a straight-sided quadrilateral element
    corners = zeros(eltype(mesh.corners), 4, 2)

    for index in 1:length(element_ids)
        # Construct point
        x = SVector(ntuple(i -> coordinates[i, index], ndims(mesh)))

        # Convert to unit coordinates; procedure differs for straight-sided
        # versus curvilinear elements
        element = element_ids[index]
        if !mesh.element_is_curved[element]
            for j in 1:2, i in 1:4
                # Pull the (x,y) values of the element corners from the global corners array
                corners[i, j] = mesh.corners[j, mesh.element_node_ids[i, element]]
            end
            unit_coordinates = invert_bilinear_interpolation(mesh, x, corners)

            # Sanity check that the computed `unit_coordinates` indeed recover the desired point `x`
            x_check = straight_side_quad_map(unit_coordinates[1], unit_coordinates[2], corners)
            if !isapprox(x[1], x_check[1], atol = 1e-13) || !isapprox(x[2], x_check[2], atol = 1e-13)
                error("failed to compute computational coordinates for the time series point $(x)")
            end
        else # mesh.element_is_curved[element]
            unit_coordinates = invert_transfinite_interpolation(mesh, x, view(mesh.surface_curves, :, element))

            # Sanity check that the computed `unit_coordinates` indeed recover the desired point `x`
            x_check = transfinite_quad_map(unit_coordinates[1], unit_coordinates[2], view(mesh.surface_curves, :, element))
            if !isapprox(x[1], x_check[1], atol = 1e-13) || !isapprox(x[2], x_check[2], atol = 1e-13)
                error("failed to compute computational coordinates for the time series point $(x)")
            end
        end

        # TODO: debug statment for removal
        println("point ", x, " has unit coordinates ", unit_coordinates, " in element ",
                element_ids[index])

        # Calculate interpolating polynomial for each dimension, making use of tensor product structure
        for d in 1:ndims(mesh)
            interpolating_polynomials[:, d, index] .= lagrange_interpolating_polynomials(unit_coordinates[d],
                                                                                         nodes,
                                                                                         wbary)
        end
    end

    return interpolating_polynomials
end

# Use a Newton iteration to determine the computational coordinates
# (xi, eta) of given (x,y) `point` that is given in physical coordinates
# by inverting the transformation. For straight-sided elements this
# amounts to inverting a bi-linear interpolation. For curved
# elements we invert the transfinite interpolation with linear blending.
# The residual function for the Newton iteration is
#    r(xi,eta) = X(xi,eta) - point
# and the Jacobian entries are computed accordingly from either
# `straight_side_quad_map_metrics` or `transfinite_quad_map_metrics`.
# We exploit the 2x2 nature of the problem and directly compute the matrix
# inverse to make things faster. The implementations below are inspired by
# an answer on Stack Overflow (https://stackoverflow.com/a/18332009) where
# the author explicitly states that their code is released to the public domain.
@inline function invert_bilinear_interpolation(mesh::UnstructuredMesh2D, point, element_corners)
    # Initial guess for the point (center of the reference element)
    xi = zero(eltype(point))
    eta = zero(eltype(point))
    for k in 1:5 # Newton's method should converge quickly
        # Compute current x and y coordinate and the Jacobian matrix
        # J = (X_xi, X_eta; Y_xi, Y_eta)
        x, y = straight_side_quad_map(xi, eta, element_corners)
        J11, J12, J21, J22 = straight_side_quad_map_metrics(xi, eta, element_corners)

        # Compute residuals for the Newton teration for the current (x, y) coordinate
        r1 = x - point[1]
        r2 = y - point[2]

        # Newton update that directly applies the inverse of the 2x2 Jacobian matrix
        inv_detJ = inv(J11 * J22 - J12 * J21)

        xi = xi - inv_detJ * (J22 * r1 - J12 * r2)
        eta = eta - inv_detJ * (-J21 * r1 + J11 * r2)

        # Ensure updated point is in the reference element
        xi = min(max(xi, -1), 1)
        eta = min(max(eta, -1), 1)
    end

    return SVector(xi, eta)
end

@inline function invert_transfinite_interpolation(mesh::UnstructuredMesh2D, point, surface_curves::AbstractVector{<:CurvedSurface})
    # Initial guess for the point (center of the reference element)
    xi = zero(eltype(point))
    eta = zero(eltype(point))
    for k in 1:5 # Newton's method should converge quickly
        # Compute current x and y coordinate and the Jacobian matrix
        # J = (X_xi, X_eta; Y_xi, Y_eta)
        x, y = transfinite_quad_map(xi, eta, surface_curves)
        J11, J12, J21, J22 = transfinite_quad_map_metrics(xi, eta, surface_curves)

        # Compute residuals for the Newton teration for the current (x,y) coordinate
        r1 = x - point[1]
        r2 = y - point[2]

        # Newton update that directly applies the inverse of the 2x2 Jacobian matrix
        inv_detJ = inv(J11 * J22 - J12 * J21)

        xi = xi - inv_detJ * (J22 * r1 - J12 * r2)
        eta = eta - inv_detJ * (-J21 * r1 + J11 * r2)

        # Ensure updated point is in the reference element
        xi = min(max(xi, -1), 1)
        eta = min(max(eta, -1), 1)
    end

    return SVector(xi, eta)
end

function calc_interpolating_polynomials(coordinates, element_ids,
                                        mesh::Union{TreeMesh, UnstructuredMesh2D},
                                        dg, cache)
    interpolating_polynomials = Array{real(dg), 3}(undef,
                                                   nnodes(dg), ndims(mesh),
                                                   length(element_ids))
    calc_interpolating_polynomials!(interpolating_polynomials, coordinates, element_ids,
                                    mesh, dg,
                                    cache)

    return interpolating_polynomials
end

# Record the solution variables at each given point
function record_state_at_points!(point_data, u, solution_variables,
                                 n_solution_variables,
                                 mesh::Union{TreeMesh{2}, UnstructuredMesh2D},
                                 equations, dg::DG, time_series_cache)
    @unpack element_ids, interpolating_polynomials = time_series_cache
    old_length = length(first(point_data))
    new_length = old_length + n_solution_variables

    # Loop over all points/elements that should be recorded
    for index in 1:length(element_ids)
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

            for v in 1:length(u_node)
                data[old_length + v] += (u_node[v]
                                         * interpolating_polynomials[i, 1, index]
                                         * interpolating_polynomials[j, 2, index])
            end
        end
    end
end
end # @muladd
