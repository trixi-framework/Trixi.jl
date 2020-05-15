module TimeDisc

include("pairedrk.jl")

using ..Trixi
using ..Solvers: AbstractSolver, rhs!, update_level_info!, set_acc_level_id!
using ..Auxiliary: timer, parameter
using ..Mesh: TreeMesh
using ..Mesh.Trees: minimum_level, maximum_level
using .PairedRk: calc_coefficients, calc_c, calc_a_multilevel, acc_level_ids_by_stage
using TimerOutputs: @timeit

export timestep!


# Second-order paired Runge-Kutta method (multilevel version)
function timestep!(solver::AbstractSolver, mesh::TreeMesh,
                   ::Val{:paired_rk_2_multi}, t::Float64, dt::Float64)
  @timeit timer() "time integration" begin
    # Get parameters
    n_stages = parameter("n_stages", 16, valid=(2, 4, 8, 16))
    derivative_evaluations = parameter("derivative_evaluations", n_stages, valid=(2, 4, 8, 16))
    optimized_paired_rk = parameter("optimized_paired_rk", true)
    polyDeg = parameter("N")

    # Determine Runge-Kutta coefficients "c"
    c = calc_c(n_stages)

    # Store for convenience
    u   = solver.elements.u
    k   = solver.elements.u_t
    k1 = solver.elements.u_rungekutta
    un  = similar(k)

    # Implement general Runge-Kutta method (not storage-optimized) for paired RK schemes, where
    # aᵢⱼ= 0 except for j = 1 or j = i - 1
    # bₛ = 1, bᵢ = 0  for i ≠ s
    # c₁ = 0
    #
    #                 s
    # uⁿ⁺¹ = uⁿ + Δt  ∑ bᵢkᵢ = uⁿ + Δt kₛ
    #                i=1
    # k₁ = rhs(tⁿ, uⁿ)
    # k₂ = rhs(tⁿ + c₂Δt, uⁿ + Δt(a₂₁ k₁))
    # k₃ = rhs(tⁿ + c₃Δt, uⁿ + Δt(a₃₁ k₁ + a₃₂ k₂))
    # k₄ = rhs(tⁿ + c₄Δt, uⁿ + Δt(a₄₁ k₁ + a₄₃ k₃))
    # ...
    # kₛ = rhs(tⁿ + cₛΔt, uⁿ + Δt(aₛ₁ k₁ + aₛ,ₛ₋₁ kₛ₋₁))

    # Update level info for each element
    @timeit timer() "update level info" update_level_info!(solver, mesh)

    # Determine number of levels and obtain accumulated level id for each stage
    n_levels = length(solver.level_info_elements)
    acc_level_ids = acc_level_ids_by_stage(n_stages, n_levels)
  end

  # Stage 1
  @timeit timer() "time integration" begin
    stage = 1
    t_stage = t + dt * c[stage]
    optimized_paired_rk && set_acc_level_id!(solver, acc_level_ids[stage])
  end
  @timeit timer() "rhs" rhs!(solver, t_stage, stage, acc_level_ids[stage])

  # Store permanently
  @timeit timer() "time integration" begin
    @timeit timer() "Runge-Kutta step" begin
      @. un = u
      @. k1 = k
    end
  end

  # Stage 2
  @timeit timer() "time integration" begin
    stage = 2
    t_stage = t + dt * c[stage]
    @timeit timer() "calc_a_multilevel" a_1, _ = calc_a_multilevel(polyDeg, n_stages,
                                                                   stage,
                                                                   derivative_evaluations,
                                                                   solver.n_elements,
                                                                   solver.level_info_elements)
    a_1_rs = reshape(a_1, 1, 1, 1, :)
    @timeit timer() "Runge-Kutta step 1" @. u = un + dt * a_1_rs * k1
    optimized_paired_rk && set_acc_level_id!(solver, acc_level_ids[stage])
  end
  @timeit timer() "rhs" rhs!(solver, t_stage, stage, acc_level_ids[stage])

  # Stages 3-n_stages
  for stage in 3:n_stages
    @timeit timer() "time integration" begin
      t_stage = t + dt * c[stage]
      @timeit timer() "calc_a_multilevel" a_1, a_2 = calc_a_multilevel(polyDeg, n_stages,
                                                                       stage,
                                                                       derivative_evaluations,
                                                                       solver.n_elements,
                                                                       solver.level_info_elements)
      #=a_1_rs = reshape(a_1, 1, 1, 1, :)=#
      #=a_2_rs = reshape(a_2, 1, 1, 1, :)=#
      #=@timeit timer() "Runge-Kutta step 2" @. u = un + dt * (a_1_rs * k1 + a_2_rs * k)=#
      @timeit timer() "Runge-Kutta step 2" begin
        substep!(u, un, dt, a_1, k1, a_2, k, Val(size(u, 2)), Val(size(u, 1)))
      end
      optimized_paired_rk && set_acc_level_id!(solver, acc_level_ids[stage])
    end
    @timeit timer() "rhs" rhs!(solver, t_stage, stage, acc_level_ids[stage])
  end

  # Final update to u
  @timeit timer() "time integration" begin
    @timeit timer() "Runge-Kutta step 3" @. u = un + dt * k
  end
end

# Stages 3-n
function substep!(u, un, dt, a_1, k1, a_2, k, ::Val{NNODES}, ::Val{NVARS}) where {NNODES, NVARS}
  n_elements = size(u, 4)
  @inbounds for element_id in 1:n_elements
    @inbounds for j in 1:NNODES, i in 1:NNODES
      @inbounds @simd for v in 1:NVARS
        u[v, i, j, element_id] = (un[v, i, j, element_id] + dt *
                                  (a_1[element_id] * k1[v, i, j, element_id] +
                                   a_2[element_id] * k[v, i, j, element_id]))
      end
    end
  end
end


# Second-order paired Runge-Kutta method
function timestep!(solver::AbstractSolver, mesh::TreeMesh,
                   ::Val{:paired_rk_2_s}, t::Float64, dt::Float64)
  @timeit timer() "time integration" begin
    # Get parameters
    n_stages = parameter("n_stages", valid=(2, 4, 8, 16))
    derivative_evaluations = parameter("derivative_evaluations", valid=(2, 3, 4, 8, 16))
    polyDeg = parameter("N")

    # Determine Runge-Kutta coefficients
    a, c = calc_coefficients(polyDeg, n_stages, derivative_evaluations)

    # Store for convenience
    u   = solver.elements.u
    k   = solver.elements.u_t
    k1 = solver.elements.u_rungekutta
    un  = similar(k)
  end

  # Implement general Runge-Kutta method (not storage-optimized) for paired RK schemes, where
  # aᵢⱼ= 0 except for j = 1 or j = i - 1
  # bₛ = 1, bᵢ = 0  for i ≠ s
  # c₁ = 0
  #
  #                 s
  # uⁿ⁺¹ = uⁿ + Δt  ∑ bᵢkᵢ = uⁿ + Δt kₛ
  #                i=1
  # k₁ = rhs(tⁿ, uⁿ)
  # k₂ = rhs(tⁿ + c₂Δt, uⁿ + Δt(a₂₁ k₁))
  # k₃ = rhs(tⁿ + c₃Δt, uⁿ + Δt(a₃₁ k₁ + a₃₂ k₂))
  # k₄ = rhs(tⁿ + c₄Δt, uⁿ + Δt(a₄₁ k₁ + a₄₃ k₃))
  # ...
  # kₛ = rhs(tⁿ + cₛΔt, uⁿ + Δt(aₛ₁ k₁ + aₛ,ₛ₋₁ kₛ₋₁))

  # Stage 1
  stage = 1
  t_stage = t + dt * c[stage]
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Store permanently
  @timeit timer() "time integration" begin
    @. un = u
    @. k1 = k
  end

  # Stage 2
  stage = 2
  t_stage = t + dt * c[stage]
  @timeit timer() "time integration" @. u = un + dt * a[ 2, 1] * k1
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Stages 3-n_stages
  for stage in 3:n_stages
    t_stage = t + dt * c[stage]
    @timeit timer() "time integration" @. u = un + dt * (a[stage, 1] * k1 + a[stage, stage-1] * k)
    @timeit timer() "rhs" rhs!(solver, t_stage, stage)
  end

  # Final update to u
  @timeit timer() "time integration" @. u = un + dt * k
end


# Carpenter's 4th-order 5-stage low-storage Runge-Kutta method
function timestep!(solver::AbstractSolver, mesh::TreeMesh,
                   ::Val{:carpenter_4_5}, t::Float64, dt::Float64)
  a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  for stage = 1:5
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)
    @timeit timer() "time integration" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end


end
