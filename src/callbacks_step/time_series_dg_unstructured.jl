# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

# Elements on an `UnstructuredMesh2D` are possibly curved. Assume that each
# element is convex, i.e., all interior angles are less than 180 degrees.
# This routine computes the shortest distance from a given point to each element
# surface in the mesh. These distances then indicate possible candidate elements.
# From these candidates we (essentially) apply a ray casting strategy and identify
# the element in which the point lies by comparing the ray formed by the point to
# the nearest boundary to the rays cast by the candidate element barycenters to the
# boundary. If these rays point in the same direction, then we have identified the
# desired element location.
function get_elements_by_coordinates!(element_ids, coordinates,
                                      mesh::UnstructuredMesh2D,
                                      dg, cache)
    if length(element_ids) != size(coordinates, 2)
        throw(DimensionMismatch("storage length for element ids does not match the number of coordinates"))
    end

    # Reset element ids - 0 indicates "not (yet) found"
    element_ids .= 0

    # Compute and save the barycentric coordinate on each element
    bary_centers = zeros(eltype(mesh.corners), 2, mesh.n_elements)
    calc_bary_centers!(bary_centers, dg, cache)

    # Iterate over coordinates
    distances = zeros(eltype(mesh.corners), mesh.n_elements)
    indices = zeros(Int, mesh.n_elements, 2)
    for index in eachindex(element_ids)
        # Grab the current point for which the element needs found
        point = SVector(coordinates[1, index],
                        coordinates[2, index])

        # Compute the minimum distance between the `point` and all the element surfaces
        # saved into `distances`. The point in `node_coordinates` that gives said minimum
        # distance on each element is saved in `indices`
        distances, indices = calc_minimum_surface_distance(point,
                                                           cache.elements.node_coordinates,
                                                           dg, mesh)

        # Get the candidate elements where the `point` might live
        candidates = findall(abs.(minimum(distances) .- distances) .<
                             500 * eps(eltype(point)))

        # The minimal surface point is on a boundary so it plays no role which candidate
        # we use to grab it. So just use the first one
        surface_point = SVector(cache.elements.node_coordinates[1,
                                                                indices[candidates[1],
                                                                        1],
                                                                indices[candidates[1],
                                                                        2],
                                                                candidates[1]],
                                cache.elements.node_coordinates[2,
                                                                indices[candidates[1],
                                                                        1],
                                                                indices[candidates[1],
                                                                        2],
                                                                candidates[1]])

        # Compute the vector pointing from the current `point` toward the surface
        P = surface_point - point

        # If the vector `P` is the zero vector then this `point` is at an element corner or
        # on a surface. In this case the choice of a candidate element is ambiguous and
        # we just use the first candidate. However, solutions might differ at discontinuous
        # interfaces such that this choice may influence the result.
        if sum(P .* P) < 500 * eps(eltype(point))
            element_ids[index] = candidates[1]
            continue
        end

        # Loop through all the element candidates until we find a vector from the barycenter
        # to the surface that points in the same direction as the current `point` vector.
        # This then gives us the correct element.
        for element in eachindex(candidates)
            bary_center = SVector(bary_centers[1, candidates[element]],
                                  bary_centers[2, candidates[element]])
            # Vector pointing from the barycenter toward the minimal `surface_point`
            B = surface_point - bary_center
            if sum(P .* B) > zero(eltype(bary_center))
                element_ids[index] = candidates[element]
                break
            end
        end
    end

    return element_ids
end

# Use the available `node_coordinates` on each element to compute and save the barycenter.
# In essence, the barycenter is like an average where all the x and y node coordinates are
# summed and then we divide by the total number of degrees of freedom on the element, i.e.,
# the value of `n^2` in two spatial dimensions.
@inline function calc_bary_centers!(bary_centers, dg, cache)
    n = nnodes(dg)
    @views for element in eachelement(dg, cache)
        bary_centers[1, element] = sum(cache.elements.node_coordinates[1, :, :,
                                                                       element]) / n^2
        bary_centers[2, element] = sum(cache.elements.node_coordinates[2, :, :,
                                                                       element]) / n^2
    end
    return nothing
end

# Compute the shortest distance from a `point` to the surface of each element
# using the available `node_coordinates`. Also return the index pair of this
# minimum surface point location. We compute and store in `min_distance`
# the squared norm to avoid computing computationally more expensive square roots.
# Note! Could be made more accurate if the `node_coordinates` were super-sampled
# and reinterpolated onto a higher polynomial degree before this computation.
function calc_minimum_surface_distance(point, node_coordinates,
                                       dg, mesh::UnstructuredMesh2D)
    n = nnodes(dg)
    min_distance2 = Inf * ones(eltype(mesh.corners), length(mesh))
    indices = zeros(Int, length(mesh), 2)
    for k in 1:length(mesh)
        # used to ensure that only boundary points are used
        on_surface = MVector(false, false)
        for j in 1:n
            on_surface[2] = (j == 1) || (j == n)
            for i in 1:n
                on_surface[1] = (i == 1) || (i == n)
                if !any(on_surface)
                    continue
                end
                node = SVector(node_coordinates[1, i, j, k],
                               node_coordinates[2, i, j, k])
                distance2 = sum(abs2, node - point)
                if distance2 < min_distance2[k]
                    min_distance2[k] = distance2
                    indices[k, 1] = i
                    indices[k, 2] = j
                end
            end
        end
    end

    return min_distance2, indices
end

function calc_interpolating_polynomials!(interpolating_polynomials, coordinates,
                                         element_ids,
                                         mesh::UnstructuredMesh2D, dg::DGSEM, cache)
    @unpack nodes = dg.basis

    wbary = barycentric_weights(nodes)

    # Helper array for a straight-sided quadrilateral element
    corners = zeros(eltype(mesh.corners), 4, 2)

    for index in eachindex(element_ids)
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
            # Compute coordinates in reference system
            unit_coordinates = invert_bilinear_interpolation(mesh, x, corners)

            # Sanity check that the computed `unit_coordinates` indeed recover the desired point `x`
            x_check = straight_side_quad_map(unit_coordinates[1], unit_coordinates[2],
                                             corners)
            if !isapprox(x[1], x_check[1]) || !isapprox(x[2], x_check[2])
                error("failed to compute computational coordinates for the time series point $(x), closet candidate was $(x_check)")
            end
        else # mesh.element_is_curved[element]
            unit_coordinates = invert_transfinite_interpolation(mesh, x,
                                                                view(mesh.surface_curves,
                                                                     :, element))

            # Sanity check that the computed `unit_coordinates` indeed recover the desired point `x`
            x_check = transfinite_quad_map(unit_coordinates[1], unit_coordinates[2],
                                           view(mesh.surface_curves, :, element))
            if !isapprox(x[1], x_check[1]) || !isapprox(x[2], x_check[2])
                error("failed to compute computational coordinates for the time series point $(x), closet candidate was $(x_check)")
            end
        end

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
# (xi, eta) of an (x,y) `point` that is given in physical coordinates
# by inverting the transformation. For straight-sided elements this
# amounts to inverting a bi-linear interpolation. For curved
# elements we invert the transfinite interpolation with linear blending.
# The residual function for the Newton iteration is
#    r(xi, eta) = X(xi, eta) - point
# and the Jacobian entries are computed accordingly from either
# `straight_side_quad_map_metrics` or `transfinite_quad_map_metrics`.
# We exploit the 2x2 nature of the problem and directly compute the matrix
# inverse to make things faster. The implementations below are inspired by
# an answer on Stack Overflow (https://stackoverflow.com/a/18332009) where
# the author explicitly states that their code is released to the public domain.
@inline function invert_bilinear_interpolation(mesh::UnstructuredMesh2D, point,
                                               element_corners)
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

        # Update with explicitly inverted Jacobian
        xi = xi - inv_detJ * (J22 * r1 - J12 * r2)
        eta = eta - inv_detJ * (-J21 * r1 + J11 * r2)

        # Ensure updated point is in the reference element
        xi = min(max(xi, -1), 1)
        eta = min(max(eta, -1), 1)
    end

    return SVector(xi, eta)
end

@inline function invert_transfinite_interpolation(mesh::UnstructuredMesh2D, point,
                                                  surface_curves::AbstractVector{<:CurvedSurface})
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

        # Update with explicitly inverted Jacobian
        xi = xi - inv_detJ * (J22 * r1 - J12 * r2)
        eta = eta - inv_detJ * (-J21 * r1 + J11 * r2)

        # Ensure updated point is in the reference element
        xi = min(max(xi, -1), 1)
        eta = min(max(eta, -1), 1)
    end

    return SVector(xi, eta)
end

function record_state_at_points!(point_data, u, solution_variables,
                                 n_solution_variables,
                                 mesh::UnstructuredMesh2D,
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
end # @muladd
