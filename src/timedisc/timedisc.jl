
# Integrate solution by repeatedly calling the rhs! method on the solver solution.
# function timestep_XYZ!(solver::AbstractSolver, t, dt)


# Integrate solution by repeatedly calling the rhs! method on the solver solution.
function timestep_euler_gravity!(solver_euler, solver_gravity, t::Float64, dt::Float64, time_parameters)
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

  # Update gravity in every time step
  # FIXME: Hack to use different CFL number for the gravity solver
  # Values for the Jeans instability test
  #gravity_cfl = 0.8 # works for CK LSRK45         (≈97% of solve) N = 3
  #gravity_cfl = 0.55 # works for CK LSRK45         (≈97% of solve) N = 4
  #gravity_cfl = 0.4375 # works for Williamson LSRK3 (≈95% of solve) N = 3
  #gravity_cfl = 0.275 # works for Williamson LSRK3 (≈95% of solve) N = 4
  #gravity_cfl = 1.2 # works for Ranocha 3Sstar (≈??% of solve) N = 3
  #gravity_cfl = 0.7 # works for Ranocha 3Sstar (≈??% of solve) N = 4
  #@timeit timer() "gravity solver" update_gravity!(solver_gravity, solver_euler.elements.u, gravity_cfl)

  # Value for the polytrope test
  #gravity_cfl = 0.5 # for LSRK45
  gravity_cfl = 1.0 # for 3Sstar
  for stage = 1:5
    # Update gravity in every RK stage
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
      @. solver.elements.u_tmp2 = (solver.elements.u_t
                                         - solver.elements.u_tmp2 * a[stage])
      @. solver.elements.u += solver.elements.u_tmp2 * b[stage] * dt
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
    #timestep_gravity!(solver, time, dt, u_euler)
    timestep_gravity_3Sstar!(solver, time, dt, u_euler)
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
function timestep_gravity!(solver::AbstractSolver, t, dt, u_euler)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
 a = [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
      3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
 b = [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
      1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
      2277821191437.0 / 14882151754819.0]
 c = [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
      2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]
  # Coefficients for Williamson's 3-stage 3rd-order low-storage Runge-Kutta method
  # a = [0.0, 5.0/9.0, 153.0/128.0]
  # b = [1.0/3.0, 15.0/16.0, 8.0/15.0]
  # c = [0.0, 1.0/3.0, 3.0/4.0]

  # Newton's gravitational constant (cgs units) for Jeans instability
  #G = 6.674e-8 # cm^3/(g⋅s^2)
  #rho0 = 1.5e7 # background density
  # Newton's gravitational constant (normalized) for polytrope test
  G = 1.0
  rho0 = 0.0
  grav_scale = -4.0*pi*G
  for stage = 1:5 # for LSRK45
#  for stage = 1:3 # for LSRK3
    t_stage = t + dt * c[stage]
    # rhs! has the source term for the harmonic problem
    @timeit timer() "rhs" rhs!(solver, t_stage)
    # put in gravity source term proportional to Euler density
    # OBS! subtract off the background density ρ_0 around which the Jeans instability is perturbed
    @views @. solver.elements.u_t[1,:,:,:] += grav_scale*(u_euler[1,:,:,:] - rho0)
    # now take the RK step
    @timeit timer() "Runge-Kutta step" begin
      @. solver.elements.u_tmp2 = (solver.elements.u_t
                                         - solver.elements.u_tmp2 * a[stage])
      @. solver.elements.u += solver.elements.u_tmp2 * b[stage] * dt
    end
  end
end


# Integrate gravity solver
# OBS! coupling source term added outside the rhs! call
function timestep_gravity_3Sstar!(solver::AbstractSolver, t, dt, u_euler)
  # New 3Sstar coefficients optimized for polynomials of degree p=3
  # and examples/parameters_hyp_diff_llf.toml
  # 5 stages, order 1
  gamma1 = @SVector [0.0000000000000000E+00, 5.2910412316555866E-01, 2.8433964362349406E-01, -1.4467571130907027E+00, 7.5592215948661057E-02]
  gamma2 = @SVector [1.0000000000000000E+00, 2.6366970460864109E-01, 3.7423646095836322E-01, 7.8786901832431289E-01, 3.7754129043053775E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 8.0043329115077388E-01, 1.3550099149374278E-01]
  beta   = @SVector [1.9189497208340553E-01, 5.4506406707700059E-02, 1.2103893164085415E-01, 6.8582252490550921E-01, 8.7914657211972225E-01]
  delta  = @SVector [1.0000000000000000E+00, 7.8593091509463076E-01, 1.2639038717454840E-01, 1.7726945920209813E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 1.9189497208340553E-01, 1.9580448818599061E-01, 2.4241635859769023E-01, 5.0728347557552977E-01]

  # Jeans instability setup
  # Newton's gravitational constant (cgs units) for Jeans instability
  #G = 6.674e-8 # cm^3/(g⋅s^2)
  #rho0 = 1.5e7 # background density

  # Newton's gravitational constant (normalized) for polytrope test
  G = 1.0
  rho0 = 0.0
  grav_scale = -4.0*pi*G

  # Polytrope setup
  r_soft = 0.001 # must be the same as in initial conditions
  inicenter = [0.0, 0.0] # must be same as in initial conditions

  solver.elements.u_tmp2 .= zero(eltype(solver.elements.u_tmp2))
  solver.elements.u_tmp3 .= solver.elements.u
  for stage in eachindex(c)
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)

    # Source term: Jeans instability
    # put in gravity source term proportional to Euler density
    # OBS! subtract off the background density ρ_0 around which the Jeans instability is perturbed
    #@views @. solver.elements.u_t[1,:,:,:] += grav_scale*(u_euler[1,:,:,:] - rho0)

    # Source term: polytrope
    for element_id in axes(u_euler, 4)
      for j in axes(u_euler, 3)
        for i in axes(u_euler, 2)
          # Calculate radius and radius with Plummer's softening to avoid singularity at r == 0.0
          x1 = solver.elements.node_coordinates[1, i, j, element_id]
          x2 = solver.elements.node_coordinates[2, i, j, element_id]
          x_norm = x1 - inicenter[1]
          y_norm = x2 - inicenter[2]
          r = sqrt(x_norm^2 + y_norm^2)
          # r_plummer = (r^2 + r_soft^2) / r
          r_plummer = max(r, r_soft)

          C = -2.0
          alpha = sqrt(2.0*pi)
          rho = u_euler[1, i, j, element_id]
          term1 = C * (alpha^2*r_plummer^2 - 1.0) * rho / r_plummer^2
          term2 = C * cos(alpha*r_plummer) / r_plummer^2

          #println(u_euler[1, i, j, element_id] - sin(alpha*r_plummer)/(alpha*r_plummer))

          numerator = C*( (alpha^2*r_plummer^2 - 1.0)*sin(alpha*r_plummer) +
                         alpha*r_plummer*cos(alpha*r_plummer) )
          denominator = alpha * r_plummer^3
          #
          # this has a similar form to the RHS in hyperbolic diffusion polytrope EXCEPT it attempts
          # to use the Euler density as part of the f(x,y) function. In this sense it has "full"
          # coupling but it is unstable...do not understand why
          #solver.elements.u_t[1, i, j, element_id] += term1 + term2
          #
          # this uses the same RHS as the hyperbolic diffusion polytrope test (stable and converges)
          # BUT it does not use the Euler solution to update gravity potential, but the p = ϕ_x and
          # q = ϕ_y variables from HypDiff are used for the gravity source term of Euler
          solver.elements.u_t[1, i, j, element_id] += numerator / denominator
          #
          # this uses the density from the Euler solver for the gravity Poisson problem (unstable)
          #solver.elements.u_t[1, i, j, element_id] += grav_scale*rho
        end
      end
    end

    delta_stage   = delta[stage]
    gamma1_stage  = gamma1[stage]
    gamma2_stage  = gamma2[stage]
    gamma3_stage  = gamma3[stage]
    beta_stage_dt = beta[stage] * dt
    @timeit timer() "Runge-Kutta step" begin
      Threads.@threads for i in eachindex(solver.elements.u)
        solver.elements.u_tmp2[i] += delta_stage * solver.elements.u[i]
        solver.elements.u[i]       = (gamma1_stage * solver.elements.u[i] +
                                      gamma2_stage * solver.elements.u_tmp2[i] +
                                      gamma3_stage * solver.elements.u_tmp3[i] +
                                      beta_stage_dt * solver.elements.u_t[i])
      end
    end
  end
end


"""
    timestep_2N!(solver, t, dt, a, b, c)

Perform one timestep using an explicit Runge-Kutta method of the low-storage
class 2N of Williamson.
"""
@inline function timestep_2N!(solver, t, dt, a, b, c)
  for stage in eachindex(c)
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)

    a_stage    = a[stage]
    b_stage_dt = b[stage] * dt
    @timeit timer() "Runge-Kutta step" begin
      Threads.@threads for i in eachindex(solver.elements.u)
        solver.elements.u_tmp2[i] = solver.elements.u_t[i] - solver.elements.u_tmp2[i] * a_stage
        solver.elements.u[i] += solver.elements.u_tmp2[i] * b_stage_dt
      end
    end
  end
end

"""
    timestep_carpenter_kennedy_erk54_2N!(solver::AbstractSolver, t, dt)

Carpenter, Kennedy (1994) Fourth order 2N storage RK schemes, Solution 3
"""
function timestep_carpenter_kennedy_erk54_2N!(solver::AbstractSolver, t, dt)
  a = @SVector [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
       3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = @SVector [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
       1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
       2277821191437.0 / 14882151754819.0]
  c = @SVector [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
       2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  timestep_2N!(solver, t, dt, a, b, c)
end

"""
    timestep_carpenter_kennedy_erk43_2N!(solver::AbstractSolver, t, dt)

Carpenter, Kennedy (1994) Third order 2N storage RK schemes with error control
"""
function timestep_carpenter_kennedy_erk43_2N!(solver::AbstractSolver, t, dt)
  a = @SVector [0, 756391 / 934407, 36441873 / 15625000, 1953125 / 1085297]
  b = @SVector [8 / 141, 6627 / 2000, 609375 / 1085297, 198961 / 526383]
  c = @SVector [0, 8 / 141, 86 / 125, 1]

  timestep_2N!(solver, t, dt, a, b, c)
end


"""
    timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)

Perform one timestep using an explicit Runge-Kutta method of the low-storage
class 3Sstar of Ketcheson.
"""
@inline function timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
  solver.elements.u_tmp2 .= zero(eltype(solver.elements.u_tmp2))
  solver.elements.u_tmp3 .= solver.elements.u
  for stage in eachindex(c)
    t_stage = t + dt * c[stage]
    @timeit timer() "rhs" rhs!(solver, t_stage)

    delta_stage   = delta[stage]
    gamma1_stage  = gamma1[stage]
    gamma2_stage  = gamma2[stage]
    gamma3_stage  = gamma3[stage]
    beta_stage_dt = beta[stage] * dt
    @timeit timer() "Runge-Kutta step" begin
      Threads.@threads for i in eachindex(solver.elements.u)
        solver.elements.u_tmp2[i] += delta_stage * solver.elements.u[i]
        solver.elements.u[i]       = (gamma1_stage * solver.elements.u[i] +
                                      gamma2_stage * solver.elements.u_tmp2[i] +
                                      gamma3_stage * solver.elements.u_tmp3[i] +
                                      beta_stage_dt * solver.elements.u_t[i])
      end
    end
  end
end


"""
    timestep_hyp_diff_N3_3Sstar!(solver::AbstractSolver, t, dt)

Five stage, first order explicit Runge-Kutta scheme with stability region optimized for
the hyperbolic diffusion equation with LLF flux and polynomials of degree N=3.
"""
function timestep_hyp_diff_N3_erk51_3Sstar!(solver::AbstractSolver, t, dt)
  # New 3Sstar coefficients optimized for polynomials of degree p=3
  # and examples/parameters_hyp_diff_llf.toml
  # 5 stages, order 1
  gamma1 = @SVector [0.0000000000000000E+00, 5.2910412316555866E-01, 2.8433964362349406E-01, -1.4467571130907027E+00, 7.5592215948661057E-02]
  gamma2 = @SVector [1.0000000000000000E+00, 2.6366970460864109E-01, 3.7423646095836322E-01, 7.8786901832431289E-01, 3.7754129043053775E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 8.0043329115077388E-01, 1.3550099149374278E-01]
  beta   = @SVector [1.9189497208340553E-01, 5.4506406707700059E-02, 1.2103893164085415E-01, 6.8582252490550921E-01, 8.7914657211972225E-01]
  delta  = @SVector [1.0000000000000000E+00, 7.8593091509463076E-01, 1.2639038717454840E-01, 1.7726945920209813E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 1.9189497208340553E-01, 1.9580448818599061E-01, 2.4241635859769023E-01, 5.0728347557552977E-01]

  timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
end


"""
    timestep_parsani_ketcheson_deconinck_erk94_3Sstar!(solver::AbstractSolver, t, dt)

Parsani, Ketcheson, Deconinck (2013)
  Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
[DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
"""
function timestep_parsani_ketcheson_deconinck_erk94_3Sstar!(solver::AbstractSolver, t, dt)
  gamma1 = @SVector [0.0000000000000000E+00, -4.6556413837561301E+00, -7.7202649689034453E-01, -4.0244202720632174E+00, -2.1296873883702272E-02, -2.4350219407769953E+00, 1.9856336960249132E-02, -2.8107894116913812E-01, 1.6894354373677900E-01]
  gamma2 = @SVector [1.0000000000000000E+00, 2.4992627683300688E+00, 5.8668202764174726E-01, 1.2051419816240785E+00, 3.4747937498564541E-01, 1.3213458736302766E+00, 3.1196363453264964E-01, 4.3514189245414447E-01, 2.3596980658341213E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 7.6209857891449362E-01, -1.9811817832965520E-01, -6.2289587091629484E-01, -3.7522475499063573E-01, -3.3554373281046146E-01, -4.5609629702116454E-02]
  beta   = @SVector [2.8363432481011769E-01, 9.7364980747486463E-01, 3.3823592364196498E-01, -3.5849518935750763E-01, -4.1139587569859462E-03, 1.4279689871485013E+00, 1.8084680519536503E-02, 1.6057708856060501E-01, 2.9522267863254809E-01]
  delta  = @SVector [1.0000000000000000E+00, 1.2629238731608268E+00, 7.5749675232391733E-01, 5.1635907196195419E-01, -2.7463346616574083E-02, -4.3826743572318672E-01, 1.2735870231839268E+00, -6.2947382217730230E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 2.8363432481011769E-01, 5.4840742446661772E-01, 3.6872298094969475E-01, -6.8061183026103156E-01, 3.5185265855105619E-01, 1.6659419385562171E+00, 9.7152778807463247E-01, 9.0515694340066954E-01]

  timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
end


"""
    timestep_parsani_ketcheson_deconinck_erk32_3Sstar!(solver::AbstractSolver, t, dt)

Parsani, Ketcheson, Deconinck (2013)
  Optimized explicit RK schemes for the spectral difference method applied to wave propagation problems
[DOI: 10.1137/120885899](https://doi.org/10.1137/120885899)
"""
function timestep_parsani_ketcheson_deconinck_erk32_3Sstar!(solver::AbstractSolver, t, dt)
  gamma1 = @SVector [0.0000000000000000E+00, -1.2664395576322218E-01, 1.1426980685848858E+00]
  gamma2 = @SVector [1.0000000000000000E+00, 6.5427782599406470E-01, -8.2869287683723744E-02]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00]
  beta   = @SVector [7.2366074728360086E-01, 3.4217876502651023E-01, 3.6640216242653251E-01]
  delta  = @SVector [1.0000000000000000E+00, 7.2196567116037724E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 7.2366074728360086E-01, 5.9236433182015646E-01]

  timestep_3Sstar!(solver, t, dt, gamma1, gamma2, gamma3, beta, delta, c)
end
