module TimeDisc

using ..Trixi
using ..Solvers: AbstractSolver, rhs!
using ..Auxiliary: timer, parameter
using TimerOutputs: @timeit

export timestep!


# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver::AbstractSolver, t::Float64, dt::Float64)
  time_integration_scheme = Symbol(parameter("time_integration_scheme", "carpenter_4_5",
                                             valid=("carpenter_4_5", "paired_rk_2_16")))
  timestep!(solver, Val(time_integration_scheme), t, dt)
end


# Second-order, 16-stage paired Runge-Kutta method
function timestep!(solver::AbstractSolver, ::Val{:paired_rk_2_16}, t::Float64, dt::Float64)
  # c is equidistant from zero to 1/2
  c = zeros(16)
  c[ 1] =  0/30
  c[ 2] =  1/30
  c[ 3] =  2/30
  c[ 4] =  3/30
  c[ 5] =  4/30
  c[ 6] =  5/30
  c[ 7] =  6/30
  c[ 8] =  7/30
  c[ 9] =  8/30
  c[10] =  9/30
  c[11] = 10/30
  c[12] = 11/30
  c[13] = 12/30
  c[14] = 13/30
  c[15] = 14/30
  c[16] = 15/30

  # Initialize storage for a
  a = fill(NaN, 16, 16)

  # a is from RKVToolbox.f95:coeff_myRKV() for N = 3, e = 16
  #=a[ 3,  2] = 0.165166669337488=#
  #=a[ 4,  3] = 0.0401668050701765=#
  #=a[ 5,  4] = 0.00759455436520174=#
  #=a[ 6,  5] = 0.00115122631486203=#
  #=a[ 7,  6] = 0.000142357578873289=#
  #=a[ 8,  7] = 1.44841485339422E-05=#
  #=a[ 9,  8] = 1.21471109398537E-06=#
  #=a[10,  9] = 8.3591475032839E-08=#
  #=a[11, 10] = 4.66643183666034E-09=#
  #=a[12, 11] = 2.07030960191563E-10=#
  #=a[13, 12] = 7.05381938694148E-12=#
  #=a[14, 13] = 1.74014502604551E-13=#
  #=a[15, 14] = 2.77667689318645E-15=#
  #=a[16, 15] = 2.15877727024596E-17=#

  # a is from RKVToolbox.f95:coeff_myRKV() for N = 4, e = 16
  a[ 3,  2] = 0.165170249013769
  a[ 4,  3] = 0.0401708925915785
  a[ 5,  4] = 0.00759453418765444
  a[ 6,  5] = 0.00115041317941212
  a[ 7,  6] = 0.000142008703633599
  a[ 8,  7] = 1.44012500747081E-05
  a[ 9,  8] = 1.20129989690388E-06
  a[10,  9] = 8.20080636558013E-08
  a[11, 10] = 4.52651381950371E-09
  a[12, 11] = 1.97772183056185E-10
  a[13, 12] = 6.60458879939973E-12
  a[14, 13] = 1.58815369931921E-13
  a[15, 14] = 2.45444021408439E-15
  a[16, 15] = 1.83497672067887E-17

  # Compute first column of Butcher tableau
  a[ 2,  1] = c[ 2]
  a[ 3,  1] = c[ 3] - a[ 3, 2]
  a[ 4,  1] = c[ 4] - a[ 4, 3]
  a[ 5,  1] = c[ 5] - a[ 5, 4]
  a[ 6,  1] = c[ 6] - a[ 6, 5]
  a[ 7,  1] = c[ 7] - a[ 7, 6]
  a[ 8,  1] = c[ 8] - a[ 8, 7]
  a[ 9,  1] = c[ 9] - a[ 9, 8]
  a[10,  1] = c[10] - a[10, 9]
  a[11,  1] = c[11] - a[11,10]
  a[12,  1] = c[12] - a[12,11]
  a[13,  1] = c[13] - a[13,12]
  a[14,  1] = c[14] - a[14,13]
  a[15,  1] = c[15] - a[15,14]
  a[16,  1] = c[16] - a[16,15]

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
