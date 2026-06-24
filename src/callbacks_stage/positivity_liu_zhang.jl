# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
                                        global_limiter_tol = 1e3 * eps(real(semi)),
                                        max_davis_yin_iterations = 500,
                                        record_davis_yin_iterations = false)

Positivity-preserving limiter which combines a global cell-average limiter 
with a local limiter such as [`PositivityPreservingLimiterZhangShu`](@ref).
The Davis-Yin splitting implementation of the global cell-average limiter is from:
- Liu, Milesis, Shu, Zhang (2026)
  Efficient optimization-based invariant-domain-preserving limiters in solving gas dynamics equations
  [doi: 10.1016/j.jcp.2026.114839](https://doi.org/10.1016/j.jcp.2026.114839)

The "Liu-Zhang" naming convention reflects that, while other co-authors have been involved, 
C. Liu and X. Zhang are the main developers of the optimization-based limiter, and are the 
two authors who are on all of the other optimization-based limiter papers.
- Liu, Shu, Zhang (2026)
  Efficient admissible set projection in optimization-based invariant-domain-preserving limiters for ideal MHD
  [arXiv: 2605.10929](https://arxiv.org/abs/2605.10929)
- Liu, Hu, Taitano, Zhang (2025)
  An optimization-based positivity-preserving limiter in semi-implicit discontinuous Galerkin schemes solving Fokker-Planck equations
  [doi: 10.1016/j.camwa.2025.05.008](https://doi.org/10.1016/j.camwa.2025.05.008)
- Liu, Riviere, Shen, Zhang (2024)
  A simple and efficient convex optimization based bound-preserving high order accurate limiter for Cahn-Hilliard-Navier-Stokes system
  [doi: 10.1137/23M1587853](https://doi.org/10.1137/23M1587853)
- Liu, Buzzard, Zhang (2024)
  An optimization based limiter for enforcing positivity in a semi-implicit discontinuous Galerkin scheme for compressible Navier-Stokes equations
  [doi: 10.1016/j.jcp.2024.113440](https://doi.org/10.1016/j.jcp.2024.113440)

The keyword argument `global_limiter_tol` is the convergence tolerance for the Davis-Yin
splitting iteration in the global cell-average limiter, and `max_davis_yin_iterations` sets 
the maximum number of Davis-Yin iterations per global limiting step.

If `record_davis_yin_iterations` is `true`, the number of Davis-Yin iterations used at each
global limiting step is saved to the field `history_davis_yin_iterations` of the limiter.
"""
mutable struct PositivityPreservingLimiterLiuZhang{LocalLimiter,
                                                   CellAverages <: AbstractVector,
                                                   DavisYinDualVars <: AbstractVector,
                                                   ProjectedCellAverages <:
                                                   AbstractVector,
                                                   SqrtCellVolumes <:
                                                   AbstractVector{<:Real},
                                                   RealT <: Real,
                                                   ProjectionThresholds,
                                                   ProjectionVariables}
    local_limiter!::LocalLimiter
    cell_averages::CellAverages
    davis_yin_dual_vars::DavisYinDualVars
    projected_cell_averages::ProjectedCellAverages
    sqrt_cell_volumes::SqrtCellVolumes
    global_limiter_tol::RealT
    max_davis_yin_iterations::Int
    projection_thresholds::ProjectionThresholds
    projection_variables::ProjectionVariables
    record_davis_yin_iterations::Bool
    history_davis_yin_iterations::Vector{Int}
end

# For compressible Euler, convert local limiter variables and thresholds 
# to `(rho_floor, rho_e_floor)` with variables `(Trixi.density, energy_internal)`.
function convert_variables_and_thresholds(thresholds, variables,
                                          equations::Union{CompressibleEulerEquations1D,
                                                           CompressibleEulerEquations2D})
    if length(thresholds) != 2 || length(variables) != 2
        error("PositivityPreservingLimiterLiuZhang for compressible Euler requires exactly ",
              "two limiter variables: one for density and one for internal energy or pressure.")
    end

    rho_floor = nothing
    rho_e_floor = nothing
    for (threshold, variable) in zip(thresholds, variables)
        if variable === Trixi.density
            rho_floor = threshold
        elseif variable === energy_internal
            rho_e_floor = threshold
        elseif variable === pressure
            # convert pressure floor to internal energy floor; 
            # for ideal gas, p / (gamma - 1) = rho_e
            rho_e_floor = threshold / (equations.gamma - 1)
        else
            error("PositivityPreservingLimiterLiuZhang for compressible Euler requires ",
                  "variables = (density, energy_internal) or (density, pressure) ",
                  "(in either order); got unsupported variable.")
        end
    end
    if rho_floor === nothing || rho_e_floor === nothing
        error("PositivityPreservingLimiterLiuZhang for compressible Euler requires exactly ",
              "one limiter variable for density and one for internal energy or pressure.")
    end

    # return sorted thresholds and variables
    return (rho_floor, rho_e_floor), (Trixi.density, energy_internal)
end

# generic fallback: copy over the local limiter variables and thresholds as-is.
function convert_variables_and_thresholds(thresholds, variables, equations)
    return thresholds, variables
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             semi::AbstractSemidiscretization;
                                             global_limiter_tol = 1e2 * eps(real(semi)),
                                             max_davis_yin_iterations = 500,
                                             record_davis_yin_iterations = false)
    return PositivityPreservingLimiterLiuZhang(local_limiter!,
                                               mesh_equations_solver_cache(semi)...;
                                               global_limiter_tol,
                                               max_davis_yin_iterations,
                                               record_davis_yin_iterations)
end

function PositivityPreservingLimiterLiuZhang(local_limiter!,
                                             mesh::AbstractMesh, equations, dg::DGSEM,
                                             cache;
                                             global_limiter_tol,
                                             max_davis_yin_iterations,
                                             record_davis_yin_iterations)
    uEltype = real(dg)

    n_elements = nelements(dg, cache)
    sqrt_cell_volumes = [sqrt(get_cell_volume(element, mesh, equations, dg, cache))
                         for element in eachelement(dg, cache)]

    # create resizable arrays
    T = SVector{nvariables(equations), uEltype}
    cell_averages = Vector{T}(undef, n_elements)
    davis_yin_dual_vars = Vector{T}(undef, n_elements)
    projected_cell_averages = Vector{T}(undef, n_elements)

    # initialize empty length-0 history of Davis-Yin iterations 
    history_davis_yin_iterations = Vector{Int}(undef, 0)

    # convert local limiter variables and thresholds to the format expected by the global limiter
    projection_thresholds, projection_variables = convert_variables_and_thresholds(local_limiter!.thresholds,
                                                                                   local_limiter!.variables,
                                                                                   equations)

    return PositivityPreservingLimiterLiuZhang(local_limiter!, cell_averages,
                                               davis_yin_dual_vars,
                                               projected_cell_averages,
                                               sqrt_cell_volumes,
                                               global_limiter_tol,
                                               max_davis_yin_iterations,
                                               projection_thresholds,
                                               projection_variables,
                                               record_davis_yin_iterations,
                                               history_davis_yin_iterations)
end

function Base.show(io::IO, limiter::PositivityPreservingLimiterLiuZhang)
    @nospecialize limiter # reduce precompilation time
    (; global_limiter_tol, max_davis_yin_iterations, history_davis_yin_iterations) = limiter

    print(io, "PositivityPreservingLimiterLiuZhang(local_limiter!=",
          Base.typename(typeof(limiter.local_limiter!)).name)
    print(io, ", global_limiter_tol=", global_limiter_tol)
    print(io, ", max_davis_yin_iterations=", max_davis_yin_iterations)
    print(io, ", history_davis_yin_iterations=", history_davis_yin_iterations)
    print(io, ")")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain",
                   limiter::PositivityPreservingLimiterLiuZhang)
    @nospecialize limiter # reduce precompilation time
    (; global_limiter_tol, max_davis_yin_iterations, history_davis_yin_iterations) = limiter

    if get(io, :compact, false)
        show(io, limiter)
    else
        setup = Pair{String, Any}["local_limiter!" => Base.typename(typeof(limiter.local_limiter!)).name,
                                  "global_limiter_tol" => global_limiter_tol,
                                  "max_davis_yin_iterations" => max_davis_yin_iterations]
        push!(setup, "history_davis_yin_iterations" => history_davis_yin_iterations)
        summary_box(io, "PositivityPreservingLimiterLiuZhang", setup)
    end
end

function (global_limiter!::PositivityPreservingLimiterLiuZhang)(u_ode, integrator,
                                                                semi::AbstractSemidiscretization,
                                                                t)
    mesh, equations, dg, cache = mesh_equations_solver_cache(semi)
    (; local_limiter!, cell_averages, davis_yin_dual_vars, projected_cell_averages,
    sqrt_cell_volumes, global_limiter_tol, max_davis_yin_iterations,
    projection_thresholds, projection_variables,
    record_davis_yin_iterations, history_davis_yin_iterations) = global_limiter!

    @trixi_timeit timer() "Liu-Zhang positivity limiter" begin
        u = wrap_array(u_ode, semi)

        # resize all arrays if the number of elements has changed (e.g., due to AMR)
        n_elements = nelements(dg, cache)
        if length(cell_averages) != n_elements
            resize!(cell_averages, n_elements)
            resize!(davis_yin_dual_vars, n_elements)
            resize!(projected_cell_averages, n_elements)
            resize!(sqrt_cell_volumes, n_elements)
        end

        @trixi_timeit timer() "calc cell averages" begin
            # calculate cell averages of all variables
            @threaded for element in eachelement(dg, cache)
                cell_averages[element] = compute_u_mean(u, element, mesh, equations, dg,
                                                        cache)
            end
        end

        # check if the cell average is admissible
        cell_average_bounds_violated = false
        for element in eachelement(dg, cache)
            if !state_is_admissible(cell_averages[element],
                                    projection_thresholds,
                                    projection_variables,
                                    equations)
                cell_average_bounds_violated = true
                break
            end
        end

        # if any cell average violates a positivity bound, apply the global limiter
        if cell_average_bounds_violated

            # Recalculate total volume and sqrt of cell volumes. 
            # Note: this can be avoided by detecting when AMR occurs; however, 
            # the check `length(cell_averages) != n_elements` used to resize arrays 
            # is insufficient to detect this, since AMR can refine/coarsen while 
            # keeping the total number of elements constant.
            total_volume = zero(eltype(sqrt_cell_volumes))
            for e in eachelement(dg, cache)
                cell_volume = get_cell_volume(e, mesh, equations, dg, cache)
                sqrt_cell_volumes[e] = sqrt(cell_volume)
                total_volume += cell_volume
            end

            @trixi_timeit timer() "global cell-average limiter" begin
                global_cell_average_limiter!(u, cell_averages,
                                             davis_yin_dual_vars,
                                             projected_cell_averages,
                                             sqrt_cell_volumes, total_volume,
                                             projection_thresholds,
                                             projection_variables,
                                             global_limiter_tol,
                                             max_davis_yin_iterations,
                                             record_davis_yin_iterations,
                                             history_davis_yin_iterations,
                                             mesh_equations_solver_cache(semi)...)
            end
        end

        # after the global limiter, call a local (e.g., Zhang-Shu type) limiter to 
        # enforce pointwise positivity 
        local_limiter!(u_ode, integrator, semi, t)
    end # @trixi_timeit

    return nothing
end

function global_cell_average_limiter!(u, cell_averages,
                                      davis_yin_dual_vars, projected_cell_averages,
                                      sqrt_cell_volumes, total_volume,
                                      lower_bounds, variables,
                                      global_limiter_tol,
                                      max_davis_yin_iterations,
                                      record_davis_yin_iterations,
                                      history_davis_yin_iterations,
                                      mesh, equations, dg, cache)
    global_integral = zero(eltype(cell_averages))
    for element in eachelement(dg, cache)
        cell_volume = sqrt_cell_volumes[element]^2
        # explicit a = a + b * c instead of a += b * c to enable @muladd
        global_integral = global_integral + cell_averages[element] * cell_volume
    end

    # residual ||X^{k+1} - X^k||_{L^2} for the Davis-Yin iteration
    residual = floatmax(eltype(sqrt_cell_volumes))

    # Davis-Yin splitting minimizes the cell average L2 error 
    #           ||Z/sqrt(cell_volume) - U_avg||_{L^2}^2 = ||Z - U_avg * sqrt(cell_volume)||_{L^2}^2 
    # Here, Z ≈ U_avg * sqrt(cell_volume). This reformulation significantly accelerates convergence 
    # of the Davis-Yin iteration for non-uniform meshes. 
    # 
    # Davis-Yin splitting uses variables X (stored in `projected_cell_averages`), Y, and 
    # Z (stored in `davis_yin_dual_vars`), where 
    # - Z is the "dual variable" and solution that is returned by the iteration.
    # - X is the projection of Z onto the admissible set.
    # - Y is the primal variable, through which conservation and admissibility constraints are coupled.
    # 
    # The iteration then proceeds as follows: given DG cell averages u_avg, 
    # 0. Initialize Z = u_avg * sqrt(cell_volume)
    # 1. If u_avg violates positivity, project u_avg = Z / sqrt(cell_volume) to the admissible set: 
    #                       X_{1/2} = proj(Z / sqrt(cell_volume)) * sqrt(cell_volume)
    #    where "proj" denotes pointwise projection of a solution state to the admissible set.
    # 2. Update the primal variable 
    #                       Y: Y = 2 * X_{1/2} - Z - gamma * grad_h
    #    Here, gamma = 1 is known to be an optimal step size, and grad_h is the gradient of the 
    #    conservation constraint:
    #             grad_h = 2 * cell_volumes .* (X_{1/2} .- u_avg * sqrt(cell_volume)) 
    #    so Step 2 simplifies to 
    #                       Y = X_{1/2} - Z + u_avg * sqrt(cell_volume)
    # 3. Enforce conservation: 
    #      X = Y + (global_integral - dot(sqrt_cell_volumes, u_avg)) * pinv(sqrt_cell_volumes)
    # 4. Update dual variable: Z = Z + (X - X_{1/2})
    # 
    # Step 1-4 are repeated until ||X - X_{1/2}||_{L^2} is smaller than the tolerance.
    # 
    # The implementation uses only two buffers: X (projected_cell_averages) and Z (davis_yin_dual_vars).
    # The vector Y is not stored explicitly, but is recalculated step 3.

    # Step 0: initialize dual variable Z = u_avg * sqrt(cell_volume)
    @threaded for element in eachelement(dg, cache)
        sqrt_cell_volume = sqrt_cell_volumes[element]
        davis_yin_dual_vars[element] = cell_averages[element] * sqrt_cell_volume
    end

    num_davis_yin_iterations = 0
    while residual > global_limiter_tol &&
        num_davis_yin_iterations < max_davis_yin_iterations

        # Step 1: projection to admissible set
        @threaded for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            unweighted_cell_average = davis_yin_dual_vars[element] / sqrt_cell_volume
            unweighted_projected_cell_average = project_to_admissible_set(unweighted_cell_average,
                                                                          lower_bounds,
                                                                          variables,
                                                                          equations)
            projected_cell_averages[element] = unweighted_projected_cell_average *
                                               sqrt_cell_volume
        end

        # Step 2: calculate primal variable Y and conservation residual
        global_integral_Y = zero(first(davis_yin_dual_vars))
        for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            u_weighted_target = cell_averages[element] * sqrt_cell_volume
            Y = projected_cell_averages[element] - davis_yin_dual_vars[element] +
                u_weighted_target
            global_integral_Y = global_integral_Y + sqrt_cell_volume * Y
        end
        conservation_residual = global_integral - global_integral_Y

        # Step 3: enforce conservation on Y and update dual variable Z
        residual_squared = zero(real(mesh))
        for element in eachelement(dg, cache)
            sqrt_cell_volume = sqrt_cell_volumes[element]
            Z_old = davis_yin_dual_vars[element]
            P = projected_cell_averages[element]
            u_weighted_target = cell_averages[element] * sqrt_cell_volume

            # recalculate Y and enforce global conservation on X 
            Y = P - Z_old + u_weighted_target
            X = Y + (sqrt_cell_volume / total_volume) * conservation_residual

            # update dual variable Z
            davis_yin_dual_vars[element] = Z_old + (X - P)

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

    if record_davis_yin_iterations == true
        push!(history_davis_yin_iterations, num_davis_yin_iterations)
    end

    # replace solution cell averages with projections of the new cell averages.
    # convergence of the Davis-Yin iteration ensures that conservation is satisfied
    # up to the iteration tolerance.
    @threaded for element in eachelement(dg, cache)
        old_cell_average = cell_averages[element]
        sqrt_cell_volume = sqrt_cell_volumes[element]
        new_cell_average = project_to_admissible_set(davis_yin_dual_vars[element] /
                                                     sqrt_cell_volume,
                                                     lower_bounds, variables, equations)

        set_u_mean!(u, new_cell_average, old_cell_average, element, mesh, equations, dg,
                    cache)
    end

    return nothing
end
end # @muladd

include("admissible_projection.jl")
