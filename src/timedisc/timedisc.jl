
# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver::AbstractSolver, t, dt)
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
    @timeit timer() "rhs" rhs!(solver, t_stage)
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end


# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver_euler, solver_gravity, t::Float64, dt::Float64)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
  a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  # Store for convenience
  solver = solver_euler

  for stage = 1:5
    # Update gravity
    @timeit "gravity" update_gravity!(solver_gravity, solver_euler.elements.u)

    # Update stage time
    t_stage = t + dt * c[stage]

    # FIXME: Update gravity in Euler solver

    @timeit timer() "rhs" rhs!(solver, t_stage)
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end


function update_gravity!(solver, u_euler)
  # FIXME: Update sources in gravity solver with density from Euler solver

  # Set up main loop
  finalstep = false
  first_loop_iteration = true
  n_iterations_max = parameter("n_iterations_max", 0)
  iteration = 0
  time = 0.0

  # Iterate gravity solver until convergence or maximum number of iterations are reached
  while !finalstep
    # Calculate time step size
    @timeit timer() "calc_dt" dt = calc_dt(solver, cfl)

    # Evolve solution by one pseudo-time step
    timestep!(solver, time, dt, u_euler[1,:,:,:])
    time += dt

    # Update iteration counter
    iteration += 1

    # Check if we reached the maximum number of iterations
    if n_iterations_max > 0 && iteration >= n_iterations_max
      finalstep = true
    end

    if maximum(abs.(solver.elements.u_t[1, :, :, :])) <= solver.equations.resid_tol
      println()
      println("-"^80)
      println("  Steady state tolerance of ",solver.equations.resid_tol," reached at time ",time)
      println("-"^80)
      println()
      finalstep = true
    end

    if first_loop_iteration
      clear_malloc_data()
      first_loop_iteration = false
    end
  end
end

#=
# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver::AbstractSolver, t, dt, fixed_dens)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
  a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]
  # Newton's gravitational constant (cgs units)
  G = 6.674e-8 # cm^3/(g⋅s^2)

  for stage = 1:5
    t_stage = t + dt * c[stage]
    # rhs! has the source term for the harmonic problem
    @timeit timer() "rhs" rhs!(solver, t_stage)
    # put in gravity source term proportional to Euler density
    @views solver.elements.u_t[1,:,:,:] -= 4*pi*G*fixed_dens
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end
=#
