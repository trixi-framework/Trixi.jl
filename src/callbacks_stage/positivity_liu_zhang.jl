
struct PositivityPreservingLimiterLiuZhang{LocalLimiter,
                                           CellAverages <: AbstractVector,
                                           DavisYinZ <: AbstractVector,
                                           ProjectedCellAverages <: AbstractVector,
                                           PinvCellVolumesVector <: AbstractVector{<:Real},
                                           RealT <: Real}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    davis_yin_Z::DavisYinZ
    projected_cell_averages::ProjectedCellAverages
    pseudo_inverse_cell_volumes_vector::PinvCellVolumesVector
    global_limiter_tol::RealT
    max_davis_yin_iterations::Int
end

# TODO: choose parameter values more rigorously
function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             semi::AbstractSemidiscretization;
                                             global_limiter_tol = 1e3 * eps(real(semi)), 
                                             max_davis_yin_iterations = 500)
    return PositivityPreservingLimiterLiuZhang(local_limiter!,
                                               mesh_equations_solver_cache(semi)...;
                                               global_limiter_tol, max_davis_yin_iterations)
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             mesh::AbstractMesh, equations, dg::DGSEM, cache;
                                             global_limiter_tol, max_davis_yin_iterations)

    # vector of cell volumes and their pseudo-inverse; these are used 
    # within the optimization-based cell average limiter to enforce conservation.
    uEltype = real(dg)
    cell_volumes = [get_cell_volume(element, mesh, equations, dg, cache)
                    for element in eachelement(dg, cache)]

    pseudo_inverse_cell_volumes_vector = vec(pinv(cell_volumes))

    n_elements = nelements(dg, cache)
    T = SVector{nvariables(equations), uEltype}

    # resizable storage for cell averages and Davis-Yin workspace (same layout as cell_averages)
    cell_averages = Vector{T}(undef, n_elements)
    davis_yin_Z = Vector{T}(undef, n_elements)
    projected_cell_averages = Vector{T}(undef, n_elements)

    return PositivityPreservingLimiterLiuZhang(local_limiter!, cell_averages,
                                               davis_yin_Z, projected_cell_averages,
                                               pseudo_inverse_cell_volumes_vector,
                                               global_limiter_tol, max_davis_yin_iterations)
end

function (global_limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                                semi::AbstractSemidiscretization,
                                                                t)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; local_limiter!, cell_averages, davis_yin_Z, projected_cell_averages,
    pseudo_inverse_cell_volumes_vector,
    global_limiter_tol, max_davis_yin_iterations) = global_limiter!

    u = wrap_array(u_ode, semi)

    # calculate cell averages for each element
    n_elements = nelements(dg, cache)
    if length(cell_averages) != n_elements
        resize!(cell_averages, n_elements)
        resize!(davis_yin_Z, n_elements)
        resize!(projected_cell_averages, n_elements)
    end
    @threaded for element in eachelement(dg, cache)
        cell_averages[element] = compute_u_mean(u, element, mesh, equations, dg, cache)
    end

    # check lower bounds of cell averages
    variable = local_limiter!.variables[1] # TODO: fix
    cell_average_bounds_violated = false
    for element in eachelement(dg, cache)
        # loop through all positivity bounds enforced by the local limiter,
        # and check if the cell average violates any of them
        for (index, variable) in enumerate(local_limiter!.variables)
            if variable(cell_averages[element], equations) < local_limiter!.thresholds[index]
                cell_average_bounds_violated = true
                @show variable(cell_averages[element], equations), local_limiter!.thresholds[index]
                break
            end
        end
    end

    if cell_average_bounds_violated == true
        # apply global optimization-based cell averagelimiter + local limiters
        @trixi_timeit timer() "positivity-preserving limiter" begin

            # apply global optimization-based cell average limiter
            global_cell_average_limiter!(u, cell_averages,
                                         davis_yin_Z, projected_cell_averages,
                                         pseudo_inverse_cell_volumes_vector,
                                         local_limiter!.thresholds,
                                         global_limiter_tol, max_davis_yin_iterations,
                                         mesh_equations_solver_cache(semi)...)
        end
    end

    # call a local (e.g., Zhang-Shu type) limiter to enforce pointwise positivity 
    local_limiter!(u_ode, integrator, semi, t)

    return nothing
end

# pointwise projection to the admissible set for any scalar equation 
function project_to_admissible_set(cell_average, lower_bound,
                                   equations::AbstractEquations{NDIMS, 1}) where {NDIMS}
    return SVector(max(lower_bound[1], cell_average[1]))
end

# compute the cell volume (up to the size of the reference element) for TreeMesh.
function get_cell_volume(element, mesh::TreeMesh{NDIMS}, equations, dg, cache) where {NDIMS}
    return 2^NDIMS / (cache.elements.inverse_jacobian[element])
end

function global_cell_average_limiter!(u, cell_averages, 
                                      davis_yin_Z, projected_cell_averages,
                                      pseudo_inverse_cell_volumes_vector,
                                      lower_bound, 
                                      global_limiter_tol, max_davis_yin_iterations,
                                      mesh, equations, dg, cache)

    # calculate the global average
    global_integral = zero(eltype(cell_averages))
    for element in eachelement(dg, cache)
        cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
        global_integral = global_integral + cell_averages[element] * cell_volume
    end

    residual = floatmax(real(mesh))

    # Davis-Yin dual iterate (same Z as in the step list below); start from unconstrained means.
    @threaded for element in eachelement(dg, cache)
        davis_yin_Z[element] = cell_averages[element]
    end
    # Per-iteration proj(Z); same as former X_half, not the final limited cell_averages.

    # the Davis-Yin splitting method uses variables X, Y, Z.
    # Z is initialized to the solution cell averages. The iteration is then:
    # 1. Project to admissible set: X_{1/2} = proj(Z)
    # 2. Enforce conservation: X = Y + (global_integral - dot(cell_volumes, u_avg)) * pinv(cell_volumes)
    # 3. Update dual variable: Z = Z + (X - X_{1/2})
    # and is repeated until (X - X_{1/2}) is smaller than the tolerance or the maximum number of
    # iterations is reached. Below, passes A-C implement the same algebra with two buffers only.
    num_davis_yin_iterations = 0
    while residual > global_limiter_tol &&
        num_davis_yin_iterations < max_davis_yin_iterations

        # Pass A: same as former X_half[element] = project_to_admissible_set(Z[element], ...).
        @threaded for element in eachelement(dg, cache)
            projected_cell_averages[element] = project_to_admissible_set(davis_yin_Z[element],
                                                                         lower_bound,
                                                                         equations)
        end

        # Pass B: same dot(cell_volumes, Y) as the former serial loop over Y[element], without
        # storing Y; only the sum is needed for the conservation map on X.
        gamma = 1.0 # TODO: update to a better value
        cell_volumes_dot_Y = zero(first(davis_yin_Z))
        for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            grad_least_squares_objective = 2 * cell_volume *
                     (projected_cell_averages[element] - cell_averages[element])
            Y = 2 * projected_cell_averages[element] - davis_yin_Z[element] - gamma * grad_least_squares_objective
            cell_volumes_dot_Y += cell_volume * Y
        end

        conservation_residual = global_integral - cell_volumes_dot_Y

        # Pass C: same per element as former Y, then X = Y + conservation_term * pinv, residual
        # from norm(X - X_half), then Z += X - X_half; Y and X are not stored because Y is cheap
        # to recompute once conservation_term is known.
        residual_squared = zero(real(mesh))
        for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            z_old = davis_yin_Z[element]
            P = projected_cell_averages[element]
            grad_least_squares_objective = 2 * cell_volume * (P - cell_averages[element])
            Y = 2 * P - z_old - gamma * grad_least_squares_objective
            X = Y + conservation_residual * pseudo_inverse_cell_volumes_vector[element]
            residual_squared += sum(abs2, X - P) * cell_volume
            davis_yin_Z[element] = z_old + (X - P)
        end
        residual = sqrt(residual_squared)

        num_davis_yin_iterations += 1
    end

    @show num_davis_yin_iterations
    if num_davis_yin_iterations >= max_davis_yin_iterations
        @show cell_averages
        cell_volume = get_cell_volume(1, mesh, equations, dg, cache)
        @show cell_volume
    end

    # replace solution cell averages with the projected cell averages
    @threaded for element in eachelement(dg, cache)
        old_cell_average = cell_averages[element]
        new_cell_average = project_to_admissible_set(davis_yin_Z[element], lower_bound, equations)               
        set_u_mean!(u, new_cell_average, old_cell_average, element, mesh, equations, dg, cache)
        cell_averages[element] = new_cell_average # for debugging and testing
    end

    return nothing
end

# setter functions for the solution cell averages
function set_u_mean!(u, new_cell_average, old_cell_average, element, 
                     mesh::AbstractMesh{1}, equations, dg, cache)
    for i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, element)
        new_u_node = u_node + (new_cell_average - old_cell_average)
        set_node_vars!(u, new_u_node, equations, dg, i, element)
    end
end

function set_u_mean!(u, new_cell_average, old_cell_average, element, 
                     mesh::AbstractMesh{2}, equations, dg, cache)
    for j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, element)
        new_u_node = u_node + (new_cell_average - old_cell_average)
        set_node_vars!(u, new_u_node, equations, dg, i, j, element)
    end
end

function set_u_mean!(u, new_cell_average, old_cell_average, element, 
                     mesh::AbstractMesh{3}, equations, dg, cache)
    for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
        u_node = get_node_vars(u, equations, dg, i, j, k, element)
        new_u_node = u_node + (new_cell_average - old_cell_average)
        set_node_vars!(u, new_u_node, equations, dg, i, j, k, element)
    end
end