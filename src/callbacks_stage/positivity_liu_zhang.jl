
struct PositivityPreservingLimiterLiuZhang{N, Thresholds <: NTuple{N, <:Real},
                                           Variables <: NTuple{N, Any},
                                           LocalLimiter, # PositivityPreservingLimiterZhangShu{N} or adaptive filtering
                                           CellAverages <: AbstractVector{<:Real},
                                           PinvCellVolumesVector <: AbstractVector{<:Real},
                                           RealT <: Real}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    pseudo_inverse_cell_volumes_vector::PinvCellVolumesVector
    global_limiter_tol::RealT
    max_davis_yin_iterations::Int
end

function PositivityPreservingLimiterLiuZhang(local_limiter!, semi::AbstractSemidiscretization;
                                             global_limiter_tol = minimum(local_limiter!.thresholds),
                                             max_davis_yin_iterations = 50)
    return PositivityPreservingLimiterLiuZhang(local_limiter!, mesh_equations_solver_cache(semi)...;
                                               global_limiter_tol, max_davis_yin_iterations)
end

function PositivityPreservingLimiterLiuZhang(local_limiter!, 
                                             mesh, equations, dg, cache;
                                             global_limiter_tol, max_davis_yin_iterations)

    # vector of cell volumes and their pseudo-inverse     
    uEltype = real(dg)
    cell_volumes = [get_cell_volume(element, mesh, equations, dg, cache) for element in eachelement(dg, cache)]
    pseudo_inverse_cell_volumes_vector = vec(pinv(cell_volumes))

    # resizable storage for cell averages
    cell_averages = Vector{SVector{nvariables(equations), uEltype}}(undef, nelements(dg, cache))

    return PositivityPreservingLimiterLiuZhang(local_limiter!, cell_averages, 
                                               pseudo_inverse_cell_volumes_vector,
                                               global_limiter_tol, max_davis_yin_iterations)
end

function (limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                         semi::AbstractSemidiscretization,
                                                         t)
    (; local_limiter!, cell_averages) = limiter!
    u = wrap_array(u_ode, semi)

    # calculate cell averages for each element
    if length(cell_averages) != nelements(dg, cache)
        resize!(cell_averages, nelements(dg, cache))
    end
    @threaded for element in eachelement(dg, cache)
        cell_averages[element] = compute_u_mean(u, element, mesh, equations, dg, cache)
    end

    # apply global optimization-based cell averagelimiter + local limiters
    @trixi_timeit timer() "positivity-preserving limiter" begin

        # apply global optimization-based cell average limiter
        global_cell_average_limiter!(cell_averages,
                                     mesh_equations_solver_cache(semi)...,
                                     global_limiter_tol)

        # this initial implementation recomputes the cell averages
        local_limiter!(u, local_limiter!.thresholds, local_limiter!.variables,
                       mesh_equations_solver_cache(semi)...)
    end

    return nothing
end

# pointwise version
function project_to_admissible_set(cell_average, lower_bound, equations::LinearScalarAdvectionEquation1D)
    return max(lower_bound, cell_average)
end

function get_cell_volume(element, mesh::AbstractMesh{1}, equations, dg, cache)
    return cache.elements.inverse_jacobian[element]
end

function global_cell_average_limiter!(cell_averages, mesh::AbstractMesh{1}, equations, dg, cache;
                                      global_limiter_tol, max_davis_yin_iterations=50)
    @unpack inverse_jacobian = cache.elements

    # calculate the global average
    global_integral = zero(real(mesh))
    for element in eachelement(dg, cache)   
        cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
        global_integral = global_integral + cell_averages[element] * cell_volume
    end

    num_davis_yin_iterations = 0
    residual = floatmax(real(mesh))    
    X = copy(cell_averages) # Davis-Yin primal variable
    Y = copy(cell_averages) # Davis-Yin dual variable
    Z = copy(cell_averages) # Davis-Yin dual variable

    # the Davis-Yin splitting method uses variables X, Y, Z. 
    # Z = initialized to the cell averages u_avg
    # Project to admissible set: X_{1/2} = proj(Z)
    # Enforce conservation: X = Y + (global_integral - dot(cell_volumes, u_avg)) * pinv(cell_volumes) 
    # Update dual variable: Z = Z + (X - X_{1/2})
    while residual > global_limiter_tol && num_davis_yin_iterations < max_davis_yin_iterations

        # project the dual variable to the admissible set
        @threaded for element in eachelement(dg, cache)
            X_half[element] = project_to_admissible_set(Z[element], lower_bound, upper_bound)
        end

        # update the primal variable
        gamma = 1.0 # TODO: update to a better value
        @threaded for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            grad_h = 2 * cell_volume * (X_half[element] - cell_averages[element])
            Y[element] = 2 * X_half[element] - Z[element] - gamma * grad_h
        end

        # enforce the constraint that the sum of the cell averages is equal to the total volume
        cell_volumes_dot_Y = zero(real(mesh))
        for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            cell_volumes_dot_Y += cell_volume * Y[element]
        end
        
        @threaded for element in eachelement(dg, cache)            
            X[element] = Y[element] + (global_integral - cell_volumes_dot_Y) * pseudo_inverse_cell_averages[element]
        end

        # calculate norm(Z_new .- Z_old) 
        residual = norm((X - X_half) .* sqrt.(cell_volumes))

        # update the dual variable
        @threaded for element in eachelement(dg, cache)
            Z[element] = Z[element] + (X[element] - X_half[element])
        end

        num_davis_yin_iterations += 1
    end
end
