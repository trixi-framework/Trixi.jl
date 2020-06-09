
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


# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep!(solver_euler, solver_gravity, t::Float64, dt::Float64, time_parameters)
  @unpack cfl = time_parameters

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
    # FIXME: Hack to use different CFL number for the gravity solver
#    gravity_cfl = 0.8 # works for CK LSRK45         (≈97% of solve)
    gravity_cfl = 0.435 # works for Williamson LSRK3 (≈95% of solve)
    @timeit timer() "gravity solver" update_gravity!(solver_gravity, solver_euler.elements.u, gravity_cfl)

    # Update stage time
    t_stage = t + dt * c[stage]

    # computes compressible Euler w/o any sources
    @timeit timer() "Euler solver" rhs!(solver, t_stage)
    # add in gravitational source terms from update_gravity! call
    # OBS! u_gravity[2] contains ∂ϕ/∂x and u_gravity[3] contains ∂ϕ/∂y
    u_euler = solver_euler.elements.u
    u_t_euler = solver_euler.elements.u_t
    u_gravity = solver_gravity.elements.u
    @views @. u_t_euler[2,:,:,:] -= u_euler[1,:,:,:]*u_gravity[2,:,:,:]
    @views @. u_t_euler[3,:,:,:] -= u_euler[1,:,:,:]*u_gravity[3,:,:,:]
    @views @. u_t_euler[4,:,:,:] -= (u_euler[2,:,:,:]*u_gravity[2,:,:,:]
                                    +u_euler[3,:,:,:]*u_gravity[3,:,:,:])
    # take RK step for compressible Euler
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end

# Update the gravity potential variable(s) a la hyperbolic diffusion coupled to Euler through
# the density passed within u_euler
function update_gravity!(solver, u_euler, cfl)
  # FIXME: Outputs a lot of data to the terminal, could be improved

  # Set up main loop
  finalstep = false
  n_iterations_max = parameter("n_iterations_max", 0)
  iteration = 0
  time = 0.0

  # Iterate gravity solver until convergence or maximum number of iterations are reached
  while !finalstep
    # Calculate time step size
    @timeit timer() "calc_dt" dt = calc_dt(solver, cfl)

    # Evolve solution by one pseudo-time step
    timestep!(solver, time, dt, u_euler)
    time += dt

    # Update iteration counter
    iteration += 1

    # Check if we reached the maximum number of iterations
    if n_iterations_max > 0 && iteration >= n_iterations_max
      finalstep = true
    end
    if maximum(abs.(solver.elements.u_t[1, :, :, :])) <= solver.equations.resid_tol
      println("  Gravity solution tolerance ",solver.equations.resid_tol,
              " reached in iterations ",iteration)
      finalstep = true
    end

  end
end


# Integrate gravity solver
# OBS! coupling source term added outside the rhs! call
function timestep!(solver::AbstractSolver, t, dt, u_euler)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
#  a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
#       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
#  b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
#       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
#       2277821191437.0 / 14882151754819.0]
#  c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
#       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]
  # Coefficients for Williamson's 3-stage 3rd-order low-storage Runge-Kutta method
  a = [0.0, 5.0/9.0, 153.0/128.0]
  b = [1.0/3.0, 15.0/16.0, 8.0/15.0]
  c = [0.0, 1.0/3.0, 3.0/4.0]

  # Newton's gravitational constant (cgs units)
  G = 6.674e-8 # cm^3/(g⋅s^2)
  rho0 = 1.5e7 # background density
  grav_scale = -4.0*pi*G
#  for stage = 1:5 # for LSRK45
  for stage = 1:3 # for LSRK3
    t_stage = t + dt * c[stage]
    # rhs! has the source term for the harmonic problem
    @timeit timer() "rhs" rhs!(solver, t_stage)
    # put in gravity source term proportional to Euler density
    # OBS! subtract off the background density ρ_0 around which the Jeans instability is perturbed
    @views @. solver.elements.u_t[1,:,:,:] += grav_scale*(u_euler[1,:,:,:] - rho0)
    # now take the RK step
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_rungekutta = (solver.elements.u_t
                                         - solver.elements.u_rungekutta * a[stage])
      @. solver.elements.u += solver.elements.u_rungekutta * b[stage] * dt
    end
  end
end
