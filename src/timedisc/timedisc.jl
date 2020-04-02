module TimeDisc

include("pairedrk.jl")

using ..Trixi
using ..Solvers: AbstractSolver, rhs!
using ..Auxiliary: timer, parameter
using .PairedRk: calc_coefficients
using TimerOutputs: @timeit

export timestep!


# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver::AbstractSolver, t::Float64, dt::Float64)
  time_integration_scheme = Symbol(parameter("time_integration_scheme", "carpenter_4_5",
                                             valid=("carpenter_4_5", "paired_rk_2_16", "paired_rk_2_8")))
  timestep!(solver, Val(time_integration_scheme), t, dt)
end


# Second-order, 8-stage paired Runge-Kutta method
function timestep!(solver::AbstractSolver, ::Val{:paired_rk_2_8}, t::Float64, dt::Float64)
  # Determine Runge-Kutta coefficients
  a, c = calc_coefficients(8, 8)

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

  # Stage 1
  stage = 1
  t_stage = t + dt * c[stage]
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Store permanently
  @timeit timer() "Runge-Kutta step" begin
    @. un = u
    @. k1 = k
  end

  # Stage 2
  stage = 2
  t_stage = t + dt * c[stage]
  @timeit timer() "Runge-Kutta step" @. u = un + dt * a[ 2, 1] * k1
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Stages 3-8
  for stage in 3:8
    t_stage = t + dt * c[stage]
    @timeit timer() "Runge-Kutta step" @. u = un + dt * (a[stage, 1] * k1 + a[stage, stage-1] * k)
    @timeit timer() "rhs" rhs!(solver, t_stage, stage)
  end

  # Final update to u
  @timeit timer() "Runge-Kutta step" @. u = un + dt * k
end


# Second-order, 16-stage paired Runge-Kutta method
function timestep!(solver::AbstractSolver, ::Val{:paired_rk_2_16}, t::Float64, dt::Float64)
  # Determine Runge-Kutta coefficients
  a, c = calc_coefficients(16, 16)

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

  # Stage 1
  stage = 1
  t_stage = t + dt * c[stage]
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Store permanently
  @timeit timer() "Runge-Kutta step" begin
    @. un = u
    @. k1 = k
  end

  # Stage 2
  stage = 2
  t_stage = t + dt * c[stage]
  @timeit timer() "Runge-Kutta step" @. u = un + dt * a[ 2, 1] * k1
  @timeit timer() "rhs" rhs!(solver, t_stage, stage)

  # Stages 3-16
  for stage in 3:16
    t_stage = t + dt * c[stage]
    @timeit timer() "Runge-Kutta step" @. u = un + dt * (a[stage, 1] * k1 + a[stage, stage-1] * k)
    @timeit timer() "rhs" rhs!(solver, t_stage, stage)
  end

  # Final update to u
  @timeit timer() "Runge-Kutta step" @. u = un + dt * k
end


# Carpenter's 4th-order 5-stage low-storage Runge-Kutta method
function timestep!(solver::AbstractSolver, ::Val{:carpenter_4_5}, t::Float64, dt::Float64)
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
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end


end
