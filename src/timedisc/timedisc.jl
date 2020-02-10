module TimeDisc

using ..Jul1dge
using ..Solvers: rhs!
using ..Auxiliary: timer
using TimerOutputs: @timeit

export timestep!


# Integrate solution by repeatedly calling the rhs! method on the DG solution.
function timestep!(dg, t, dt)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
  a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  for stage = 1:5
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(dg, t_stage)
    @timeit timer() "RK" begin
      @. dg.urk = dg.ut - dg.urk * a[stage]
      @. dg.u += dg.urk * b[stage] * dt
    end
  end
end


end
