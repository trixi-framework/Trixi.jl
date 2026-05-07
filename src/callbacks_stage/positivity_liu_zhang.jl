
struct PositivityPreservingLimiterLiuZhang{N, Thresholds <: NTuple{N, <:Real},
                                           Variables <: NTuple{N, Any}, 
                                           LocalLimiter <: PositivityPreservingLimiterZhangShu{N},
                                           CellAverages <: AbstractVector{<:Real}, 
                                           RealT <: Real}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    global_limiter_tol::RealT
end

function (limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                         semi::AbstractSemidiscretization,
                                                         t)

    (; local_limiter!, cell_averages) = limiter!                                                         
    u = wrap_array(u_ode, semi)

    # calculate cell averages for each element
    resize!(cell_averages, nelements(dg, cache))
    @threaded for element in eachelement(dg, cache)
        cell_averages[element] = compute_u_mean(u, element, mesh, equations, dg, cache)
    end

    # apply global optimization-based cell averagelimiter + local limiters
    @trixi_timeit timer() "positivity-preserving limiter" begin
        global_cell_average_limiter!(cell_averages, 
                                     mesh_equations_solver_cache(semi)...,
                                     global_limiter_tol)

        # this initial implementation recomputes the cell averages
        local_limiter!(u, local_limiter!.thresholds, local_limiter!.variables,
                       mesh_equations_solver_cache(semi)...)                           
    end

    return nothing
end

function global_cell_average_limiter!(cell_averages, 
                                      mesh::AbstractMesh{1}, 
                                      equations, dg, cache; global_limiter_tol=1e-14)

    @unpack inverse_jacobian = cache.elements
    volume_jacobian = abs(inv(get_inverse_jacobian(inverse_jacobian, mesh, 1, 1, element)))

    global_integral = zero(real(mesh))
    for element in eachelement(dg, cache)
        global_integral += cell_averages[element] * volume_jacobian
    end

    diff = zero(real(mesh))
    while abs(diff) > global_limiter_tol
        # solve the optimization problem using Davis-Yin splitting
        # TODO: project onto invariant domain
        # 
    end
end
