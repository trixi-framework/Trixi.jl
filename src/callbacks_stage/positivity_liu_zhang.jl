
mutable struct PositivityPreservingLimiterLiuZhang{LocalLimiter,
                                                   CellAverages <: AbstractVector,
                                                   DavisYinZ <: AbstractVector,
                                                   ProjectedCellAverages <: AbstractVector,
                                                   SqrtCellVolumes <: AbstractVector{<:Real},
                                                   RealT <: Real}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    davis_yin_Z::DavisYinZ
    projected_cell_averages::ProjectedCellAverages
    sqrt_cell_volumes::SqrtCellVolumes
    total_volume::RealT
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

    uEltype = real(dg)
    n_elements = nelements(dg, cache)
    T = SVector{nvariables(equations), uEltype}

    cell_volumes = [get_cell_volume(element, mesh, equations, dg, cache) 
                    for element in eachelement(dg, cache)]
    sqrt_cell_volumes = sqrt.(cell_volumes)
    total_volume = sum(cell_volumes)

    cell_averages = Vector{T}(undef, n_elements)
    davis_yin_Z = Vector{T}(undef, n_elements)
    projected_cell_averages = Vector{T}(undef, n_elements)

    return PositivityPreservingLimiterLiuZhang(local_limiter!, cell_averages,
                                               davis_yin_Z, projected_cell_averages,
                                               sqrt_cell_volumes, total_volume,
                                               global_limiter_tol, max_davis_yin_iterations)
end

function (global_limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                                semi::AbstractSemidiscretization,
                                                                t)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; local_limiter!, cell_averages, davis_yin_Z, projected_cell_averages, sqrt_cell_volumes, 
       total_volume, global_limiter_tol, max_davis_yin_iterations) = global_limiter!

    u = wrap_array(u_ode, semi)

    # resize all arrays if the number of elements has changed (e.g., due to AMR)
    n_elements = nelements(dg, cache)
    if length(cell_averages) != n_elements
        resize!(cell_averages, n_elements)
        resize!(davis_yin_Z, n_elements)
        resize!(projected_cell_averages, n_elements)
        resize!(sqrt_cell_volumes, n_elements)

        # recalculate total volume and sqrt of cell volumes
        global_limiter!.total_volume = zero(typeof(global_limiter!.total_volume))
        for e in eachelement(dg, cache)
            cell_volume = get_cell_volume(e, mesh, equations, dg, cache)
            sqrt_cell_volumes[e] = sqrt(cell_volume)
            global_limiter!.total_volume += cell_volume
        end
    end

    # calculate cell averages of all variables
    @threaded for element in eachelement(dg, cache)
        cell_averages[element] = compute_u_mean(u, element, mesh, equations, dg, cache)
    end

    # loop through all positivity bounds enforced by the local limiter,
    # and check if the cell average violates any of them
    cell_average_bounds_violated = false
    for element in eachelement(dg, cache)
        for (index, variable) in enumerate(local_limiter!.variables)
            if variable(cell_averages[element], equations) < local_limiter!.thresholds[index]
                cell_average_bounds_violated = true
                break
            end
        end
    end
    
    # if any cell average violates a positivity bound, apply the global limiter
    if cell_average_bounds_violated == true
        @trixi_timeit timer() "positivity-preserving limiter" begin
            global_cell_average_limiter!(u, cell_averages,
                                         davis_yin_Z, projected_cell_averages,
                                         sqrt_cell_volumes, total_volume,
                                         local_limiter!.thresholds,
                                         global_limiter_tol, max_davis_yin_iterations,
                                         mesh_equations_solver_cache(semi)...)
        end
    end

    # after the global limiter, call a local (e.g., Zhang-Shu type) limiter to 
    # enforce pointwise positivity 
    local_limiter!(u_ode, integrator, semi, t)

    return nothing
end

# for any scalar equation, projection to the admissible set is simply a clipping operation
@inline function project_to_admissible_set(cell_average, lower_bound,
                                   equations::AbstractEquations{NDIMS, 1}) where {NDIMS}
    return SVector(max(lower_bound[1], cell_average[1]))
end

@inline function get_cell_volume(element, mesh::TreeMesh{NDIMS}, equations, dg, cache) where {NDIMS}
    return 2^NDIMS / (cache.elements.inverse_jacobian[element])
end

function global_cell_average_limiter!(u, cell_averages, 
                                      davis_yin_Z, projected_cell_averages,
                                      sqrt_cell_volumes, total_volume,
                                      lower_bound, 
                                      global_limiter_tol, max_davis_yin_iterations,
                                      mesh, equations, dg, cache)

    global_integral = zero(eltype(cell_averages))
    for element in eachelement(dg, cache)
        cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
        global_integral = global_integral + cell_averages[element] * cell_volume
    end

    residual = floatmax(real(mesh))

    @threaded for element in eachelement(dg, cache)
        sqrt_cell_volume = sqrt_cell_volumes[element]
        davis_yin_Z[element] = cell_averages[element] * sqrt_cell_volume
    end

    # the Davis-Yin splitting method uses variables X, Y, Z.
    # Z is initialized to the solution cell averages "u_avg_original" scaled by the square root 
    # of the cell volumes. The iteration is then:
    # 1. Project to admissible set: X_{1/2} = proj(Z)
    # 2. Enforce conservation: 
    #      X = Y + (global_integral - dot(sqrt_cell_volumes, u_avg_original)) * pinv(sqrt_cell_volumes)
    # 3. Update dual variable: Z = Z + (X - X_{1/2})
    # and is repeated until (X - X_{1/2}) is smaller than the tolerance or the maximum number of
    # iterations is reached. 
    # 
    # The implementation implements this with only two buffers: X (projected_cell_averages) and Z.
    num_davis_yin_iterations = 0
    while residual > global_limiter_tol &&
        num_davis_yin_iterations < max_davis_yin_iterations

        @threaded for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            projected_cell_averages[element] = project_to_admissible_set(davis_yin_Z[element] / sqrt_cell_volume,
                                                                         lower_bound,
                                                                         equations) * sqrt_cell_volume
        end

        sqrt_weighted_sum_Y = zero(first(davis_yin_Z))
        for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            P = projected_cell_averages[element]
            u_weighted_target = cell_averages[element] * sqrt_cell_volume
            Y = P - davis_yin_Z[element] + u_weighted_target
            sqrt_weighted_sum_Y = sqrt_weighted_sum_Y + sqrt_cell_volume * Y
        end

        conservation_residual = global_integral - sqrt_weighted_sum_Y

        residual_squared = zero(real(mesh))
        for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            z_old = davis_yin_Z[element]
            P = projected_cell_averages[element]
            u_weighted_target = cell_averages[element] * sqrt_cell_volume
            Y = P - z_old + u_weighted_target
            coeff = sqrt_cell_volume / total_volume
            X = Y + coeff * conservation_residual
            davis_yin_Z[element] = z_old + (X - P)
            residual_squared += sum(abs2, X - P)
        end
        residual = sqrt(residual_squared)

        num_davis_yin_iterations += 1
    end    

    # replace solution cell averages with the new cell averages
    @threaded for element in eachelement(dg, cache)
        old_cell_average = cell_averages[element]
        sqrt_cell_volume = sqrt_cell_volumes[element]
        new_cell_average = project_to_admissible_set(davis_yin_Z[element] / sqrt_cell_volume,
                                                     lower_bound, equations)
        set_u_mean!(u, new_cell_average, old_cell_average, element, mesh, equations, dg, cache)
        cell_averages[element] = new_cell_average
    end

    return nothing
end

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
