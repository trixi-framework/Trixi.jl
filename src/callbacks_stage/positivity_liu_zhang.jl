
struct PositivityPreservingLimiterLiuZhang{LocalLimiter,
                                           CellAverages <: AbstractVector,
                                           PinvCellVolumesVector <: AbstractVector{<:Real},
                                           RealT <: Real}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    pseudo_inverse_cell_volumes_vector::PinvCellVolumesVector
    global_limiter_tol::RealT
    max_davis_yin_iterations::Int
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             semi::AbstractSemidiscretization;
                                             global_limiter_tol = minimum(local_limiter!.thresholds),
                                             max_davis_yin_iterations = 50)
    return PositivityPreservingLimiterLiuZhang(local_limiter!,
                                               mesh_equations_solver_cache(semi)...;
                                               global_limiter_tol, max_davis_yin_iterations)
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             mesh::AbstractMesh, equations, dg::DGSEM,
                                             cache;
                                             global_limiter_tol, max_davis_yin_iterations)

    # vector of cell volumes and their pseudo-inverse; these are used 
    # within the optimization-based cell average limiter to enforce conservation.
    uEltype = real(dg)
    cell_volumes = [get_cell_volume(element, mesh, equations, dg, cache)
                    for element in eachelement(dg, cache)]

    pseudo_inverse_cell_volumes_vector = vec(pinv(cell_volumes))

    # resizable storage for cell averages
    cell_averages = Vector{SVector{nvariables(equations), uEltype}}(undef,
                                                                    nelements(dg, cache))

    return PositivityPreservingLimiterLiuZhang(local_limiter!, cell_averages,
                                               pseudo_inverse_cell_volumes_vector,
                                               global_limiter_tol, max_davis_yin_iterations)
end

function (global_limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                                semi::AbstractSemidiscretization,
                                                                t)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; local_limiter!, cell_averages,
    pseudo_inverse_cell_volumes_vector,
    global_limiter_tol, max_davis_yin_iterations) = global_limiter!

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
        global_cell_average_limiter!(cell_averages, local_limiter!.thresholds,
                                     pseudo_inverse_cell_volumes_vector,
                                     global_limiter_tol, max_davis_yin_iterations,
                                     mesh_equations_solver_cache(semi)...)
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

function get_cell_volume(element, mesh::AbstractMesh{1}, equations, dg, cache)
    # size of reference element is 2 in the 1D case
    return 2 / cache.elements.inverse_jacobian[element]
end

function global_cell_average_limiter!(cell_averages, lower_bound,
                                      pseudo_inverse_cell_volumes_vector,
                                      global_limiter_tol, max_davis_yin_iterations,
                                      mesh, equations, dg, cache)

    # calculate the global average
    global_integral = zero(eltype(cell_averages))
    for element in eachelement(dg, cache)
        cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
        global_integral = global_integral + cell_averages[element] * cell_volume
    end

    residual = floatmax(real(mesh))

    # TODO: avoid allocations each time
    X = copy(cell_averages) # Davis-Yin primal variable
    Y, Z, X_half = ntuple(_ -> similar(cell_averages), 3) # Davis-Yin dual variables

    # the Davis-Yin splitting method uses variables X, Y, Z. 
    # Z is initialized to the solution cell averages. The iteration is then:
    # 1. Project to admissible set: X_{1/2} = proj(Z)
    # 2. Enforce conservation: X = Y + (global_integral - dot(cell_volumes, u_avg)) * pinv(cell_volumes) 
    # 3. Update dual variable: Z = Z + (X - X_{1/2})
    # and is repeated until (X - X_{1/2}) is smaller than the tolerance or the maximum number of iterations is reached.
    num_davis_yin_iterations = 0
    while residual > global_limiter_tol &&
        num_davis_yin_iterations < max_davis_yin_iterations

        # project the dual variable to the admissible set
        @threaded for element in eachelement(dg, cache)
            X_half[element] = project_to_admissible_set(Z[element], lower_bound, equations)
        end

        # update the primal variable
        gamma = 1.0 # TODO: update to a better value
        @threaded for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            grad_h = 2 * cell_volume * (X_half[element] - cell_averages[element])
            Y[element] = 2 * X_half[element] - Z[element] - gamma * grad_h
        end

        # enforce the constraint that the sum of the cell averages is equal to the total volume
        cell_volumes_dot_Y = zero(first(Y))
        for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            cell_volumes_dot_Y += cell_volume * Y[element]
        end

        @threaded for element in eachelement(dg, cache)
            X[element] = Y[element] +
                         (global_integral - cell_volumes_dot_Y) *
                         pseudo_inverse_cell_volumes_vector[element]
        end

        # calculate residual = norm(Z_new .- Z_old) (same weighting as the scalar reference script)
        residual_squared = zero(real(mesh))
        for element in eachelement(dg, cache)
            cell_volume = get_cell_volume(element, mesh, equations, dg, cache)
            residual_squared += sum(abs2, X[element] - X_half[element]) * cell_volume
        end
        residual = sqrt(residual_squared)

        # update the dual variable
        @threaded for element in eachelement(dg, cache)
            Z[element] = Z[element] + (X[element] - X_half[element])
        end

        num_davis_yin_iterations += 1
    end

    @threaded for element in eachelement(dg, cache)
        cell_averages[element] = project_to_admissible_set(Z[element], lower_bound,
                                                           equations)
    end
end
