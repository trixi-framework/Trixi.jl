
# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver::AbstractSolver, t, dt)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
  a = @SVector [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = @SVector [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = @SVector [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  for stage = 1:5
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)

    a_stage    = a[stage]
    b_stage_dt = b[stage] * dt
    @timeit timer() "Runge-Kutta step" begin
      Threads.@threads for i in eachindex(solver.elements.u_rungekutta)
        solver.elements.u_rungekutta[i] = solver.elements.u_t[i] - solver.elements.u_rungekutta[i] * a_stage
        solver.elements.u[i] += solver.elements.u_rungekutta[i] * b_stage_dt
      end
    end
  end
end
