
"""
    PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                        global_limiter_tol = 1e3 * eps(real(semi)),
                                        max_davis_yin_iterations = 500,
                                        record_davis_yin_iterations = false)

Positivity-preserving limiter which combines a global cell-average limiter 
with a local limiter such as [`PositivityPreservingLimiterZhangShu`](@ref).
The global cell-average limiter is from:
- Liu, Milesis, Shu, Zhang (2026)
  Efficient optimization-based invariant-domain-preserving limiters in solving gas dynamics equations
  [doi: 10.1016/j.jcp.2026.114839](https://doi.org/10.1016/j.jcp.2026.114839)

Currently, admissibility is enforced via projection onto lower bounds only for
scalar equations (`nvariables == 1`).

The keyword argument `global_limiter_tol` is the convergence tolerance for the Davis-Yin
splitting iteration in the global cell-average limiter.
`max_davis_yin_iterations` sets the maximum number of Davis-Yin iterations per global
limiting step.
If `record_davis_yin_iterations` is `true`, the number of Davis-Yin iterations used at each
global limiting step is appended to `history_davis_yin_iterations`.
"""
mutable struct PositivityPreservingLimiterLiuZhang{LocalLimiter,
                                                   CellAverages <: AbstractVector,
                                                   DavisYinZ <: AbstractVector,
                                                   ProjectedCellAverages <: AbstractVector,
                                                   SqrtCellVolumes <:
                                                   AbstractVector{<:Real},
                                                   RealT <: Real,
                                                   HistoryDavisYinIterations}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    davis_yin_Z::DavisYinZ
    projected_cell_averages::ProjectedCellAverages
    sqrt_cell_volumes::SqrtCellVolumes
    total_volume::RealT
    global_limiter_tol::RealT
    max_davis_yin_iterations::Int
    history_davis_yin_iterations::HistoryDavisYinIterations
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             semi::AbstractSemidiscretization;
                                             global_limiter_tol = 1e2 * eps(real(semi)),
                                             max_davis_yin_iterations = 500,
                                             record_davis_yin_iterations = false)
    return PositivityPreservingLimiterLiuZhang(local_limiter!,
                                               mesh_equations_solver_cache(semi)...;
                                               global_limiter_tol, max_davis_yin_iterations,
                                               record_davis_yin_iterations)
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             mesh::AbstractMesh, equations, dg::DGSEM,
                                             cache;
                                             global_limiter_tol, max_davis_yin_iterations,
                                             record_davis_yin_iterations)
    uEltype = real(dg)

    n_elements = nelements(dg, cache)
    cell_volumes = [get_cell_volume(element, mesh, equations, dg, cache)
                    for element in eachelement(dg, cache)]
    sqrt_cell_volumes = sqrt.(cell_volumes)
    total_volume = sum(cell_volumes)

    # create resizable arrays
    T = SVector{nvariables(equations), uEltype}
    cell_averages = Vector{T}(undef, n_elements)
    davis_yin_Z = Vector{T}(undef, n_elements)
    projected_cell_averages = Vector{T}(undef, n_elements)

    history_davis_yin_iterations = record_davis_yin_iterations ? Int[] : nothing

    return PositivityPreservingLimiterLiuZhang(local_limiter!, cell_averages,
                                               davis_yin_Z, projected_cell_averages,
                                               sqrt_cell_volumes, total_volume,
                                               global_limiter_tol, max_davis_yin_iterations,
                                               history_davis_yin_iterations)
end

function Base.show(io::IO, limiter::PositivityPreservingLimiterLiuZhang)
    @nospecialize limiter # reduce precompilation time
    (; global_limiter_tol, max_davis_yin_iterations, history_davis_yin_iterations) = limiter

    print(io, "PositivityPreservingLimiterLiuZhang(local_limiter!=",
          Base.typename(typeof(limiter.local_limiter!)).name)
    print(io, ", global_limiter_tol=", global_limiter_tol)
    print(io, ", max_davis_yin_iterations=", max_davis_yin_iterations)
    if history_davis_yin_iterations !== nothing
        print(io, ", history_davis_yin_iterations=", history_davis_yin_iterations)
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", limiter::PositivityPreservingLimiterLiuZhang)
    @nospecialize limiter # reduce precompilation time
    (; global_limiter_tol, max_davis_yin_iterations, history_davis_yin_iterations) = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        setup = Pair{String, Any}["local_limiter!" => Base.typename(typeof(limiter.local_limiter!)).name,
                                  "global_limiter_tol" => global_limiter_tol,
                                  "max_davis_yin_iterations" => max_davis_yin_iterations]
        if history_davis_yin_iterations !== nothing
            push!(setup, "history_davis_yin_iterations" => history_davis_yin_iterations)
        end
        summary_box(io, "PositivityPreservingLimiterLiuZhang", setup)
    end
end

function (global_limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                                semi::AbstractSemidiscretization,
                                                                t)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; local_limiter!, cell_averages, davis_yin_Z, projected_cell_averages,
    sqrt_cell_volumes, total_volume, global_limiter_tol,
    max_davis_yin_iterations, history_davis_yin_iterations) = global_limiter!

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
            if variable(cell_averages[element], equations) <
               local_limiter!.thresholds[index]
                cell_average_bounds_violated = true
                break
            end
        end
        cell_average_bounds_violated && break
    end

    # if any cell average violates a positivity bound, apply the global limiter
    if cell_average_bounds_violated
        @trixi_timeit timer() "positivity-preserving limiter" begin
            global_cell_average_limiter!(u, cell_averages,
                                         davis_yin_Z, projected_cell_averages,
                                         sqrt_cell_volumes, total_volume,
                                         local_limiter!.thresholds,
                                         local_limiter!.variables,
                                         global_limiter_tol, max_davis_yin_iterations,
                                         history_davis_yin_iterations,
                                         mesh_equations_solver_cache(semi)...)
        end
    end

    # # TODO: remove this once we're done debugging
    # for element in eachelement(dg, cache)
    #     u_mean = compute_u_mean(u, element, mesh, equations, dg, cache)
    #     constraint_tolerance = minimum(local_limiter!.thresholds) * 100 * eps(eltype(u_mean))
    #     # constraint_tolerance = zero(eltype(u_mean))

    #     if !satisfies_constraints(u_mean, local_limiter!.thresholds,
    #                               local_limiter!.variables, equations;
    #                               tolerance = constraint_tolerance)           
    #         println("variable[1] = $(local_limiter!.variables[1](u_mean, equations))")
    #         println("variable[2] = $(local_limiter!.variables[2](u_mean, equations))")
    #         println("constraint_residual = $(constraint_residual(u_mean, local_limiter!.thresholds, local_limiter!.variables, equations))")
    #         error("Before local limiter, element mean violates positivity constraints; " *
    #               "adaptive filter cannot recover a constraint-satisfying state")
    #     end
    # end

    # after the global limiter, call a local (e.g., Zhang-Shu type) limiter to 
    # enforce pointwise positivity 
    local_limiter!(u_ode, integrator, semi, t)

    return nothing
end

@inline function get_cell_volume(element, mesh::TreeMesh{NDIMS}, equations, dg,
                                 cache) where {NDIMS}
    return 2^NDIMS / (cache.elements.inverse_jacobian[element])
end

function global_cell_average_limiter!(u, cell_averages,
                                      davis_yin_Z, projected_cell_averages,
                                      sqrt_cell_volumes, total_volume,
                                      lower_bound, variables,
                                      global_limiter_tol,
                                      max_davis_yin_iterations,
                                      history_davis_yin_iterations,
                                      mesh, equations, dg, cache)
    global_integral = zero(eltype(cell_averages))
    for element in eachelement(dg, cache)
        cell_volume = sqrt_cell_volumes[element]^2
        global_integral += cell_averages[element] * cell_volume
    end

    # residual ||X^{k+1} - X^k||_{L^2} for the Davis-Yin iteration
    residual = floatmax(real(mesh))

    @threaded for element in eachelement(dg, cache)
        sqrt_cell_volume = sqrt_cell_volumes[element]
        davis_yin_Z[element] = cell_averages[element] * sqrt_cell_volume
    end

    # Davis-Yin splitting minimizes the cell average L2 error 
    #           ||Z/sqrt(cell_volume) - U_avg||_{L^2}^2 = ||Z - U_avg * sqrt(cell_volume)||_{L^2}^2 
    # Here, Z ≈ U_avg * sqrt(cell_volume). This reformulation significantly accelerates convergence 
    # of the Davis-Yin iteration for non-uniform meshes. 

    # Davis-Yin splitting uses variables X, Y, Z, where 
    # - Z is the "dual variable" and solution that is returned by the iteration.
    # - X is the projection of Z onto the admissible set.
    # - Y is the primal variable, through which the conservation and admissibility constraints are coupled.
    # 
    # The iteration then proceeds as follows: given cell averages u_avg,
    # 1. Project to admissible set: X_{1/2} = proj(Z / sqrt(cell_volume)) * sqrt(cell_volume)
    # 2. Update the primal variable Y: Y = 2 * X_{1/2} - Z - gamma * grad_h
    #    Here, gamma = 1 and grad_h = 2 * cell_volumes .* (X_{1/2} .- u_avg * sqrt(cell_volume)), 
    #    so this step simplifies to 
    #                       Y = X_{1/2} - Z + u_avg * sqrt(cell_volume)
    # 3. Enforce conservation: 
    #      X = Y + (global_integral - dot(sqrt_cell_volumes, u_avg)) * pinv(sqrt_cell_volumes)
    # 4. Update dual variable: Z = Z + (X - X_{1/2})
    # This is repeated until (X - X_{1/2}) is smaller than the tolerance.
    # 
    # The implementation implements this with only two buffers: X (projected_cell_averages) and Z.
    # Y is not stored explicitly, but is recalculated once in step 3.
    num_davis_yin_iterations = 0
    while residual > global_limiter_tol &&
        num_davis_yin_iterations < max_davis_yin_iterations

        # Step 1: projection to admissible set
        @threaded for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            projected_cell_averages[element] = project_to_admissible_set(davis_yin_Z[element] /
                                                                         sqrt_cell_volume,
                                                                         lower_bound,
                                                                         variables,
                                                                         equations) *
                                               sqrt_cell_volume
        end

        # Step 2: calculate primal variable Y and conservation residual
        global_integral_Y = zero(first(davis_yin_Z))
        for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            u_weighted_target = cell_averages[element] * sqrt_cell_volume
            Y = projected_cell_averages[element] - davis_yin_Z[element] + u_weighted_target
            global_integral_Y += sqrt_cell_volume * Y
        end
        conservation_residual = global_integral - global_integral_Y

        # Step 3: enforce conservation on Y and update dual variable Z
        residual_squared = zero(real(mesh))
        for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            Z_old = davis_yin_Z[element]
            P = projected_cell_averages[element]
            u_weighted_target = cell_averages[element] * sqrt_cell_volume

            # recalculate Y and enforce global conservation on X 
            Y = P - Z_old + u_weighted_target
            X = Y + (sqrt_cell_volume / total_volume) * conservation_residual

            # update dual variable Z
            davis_yin_Z[element] = Z_old + (X - P)

            # calculate residual
            residual_squared += sum(abs2, X - P)
        end
        residual = sqrt(residual_squared)

        num_davis_yin_iterations += 1
    end

    if num_davis_yin_iterations == max_davis_yin_iterations
        @warn "Davis-Yin iteration did not converge in $(max_davis_yin_iterations) iterations; " *
              "residual = $(residual) while tolerance = $(global_limiter_tol)."
    end

    if history_davis_yin_iterations !== nothing
        push!(history_davis_yin_iterations, num_davis_yin_iterations)
    end

    # replace solution cell averages with projections of the new cell averages.
    # convergence of the Davis-Yin iteration ensures that conservation is satisfied
    # up to the iteration tolerance.
    @threaded for element in eachelement(dg, cache)
        old_cell_average = cell_averages[element]
        sqrt_cell_volume = sqrt_cell_volumes[element]
        new_cell_average = project_to_admissible_set(davis_yin_Z[element] /
                                                     sqrt_cell_volume,
                                                     lower_bound, variables, equations)

        set_u_mean!(u, new_cell_average, old_cell_average, element, mesh, equations, dg,
                    cache)

        # for visualization and debugging only                    
        cell_averages[element] = new_cell_average
    end

    return nothing
end

include("admissible_projection.jl")
