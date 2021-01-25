
"""
    ParametersEulerGravity(; background_density=0.0,
                             gravitational_constant=1.0,
                             cfl=1.0,
                             resid_tol=1.0e-4,
                             n_iterations_max=10^4,
                             timestep_gravity=timestep_gravity_erk52_3Sstar!)

Set up parameters for the gravitational part of a [`SemidiscretizationEulerGravity`](@ref).
"""
struct ParametersEulerGravity{RealT<:Real, TimestepGravity}
  background_density    ::RealT # aka rho0
  gravitational_constant::RealT # aka G
  cfl                   ::RealT
  resid_tol             ::RealT
  resid_tol_type        ::Symbol
  n_iterations_max      ::Int
  timestep_gravity::TimestepGravity
end

function ParametersEulerGravity(; background_density=0.0,
                                  gravitational_constant=1.0,
                                  cfl=1.0,
                                  resid_tol=1.0e-4,
                                  resid_tol_type=:linf_phi, #  :linf_phi, :l2_full
                                  n_iterations_max=10^4,
                                  timestep_gravity=timestep_gravity_erk52_3Sstar!)
  background_density, gravitational_constant, cfl, resid_tol = promote(
    background_density, gravitational_constant, cfl, resid_tol)

  return ParametersEulerGravity(background_density, gravitational_constant, cfl,
    resid_tol, resid_tol_type, n_iterations_max, timestep_gravity)
end

function Base.show(io::IO, parameters::ParametersEulerGravity)
  print(io, "ParametersEulerGravity(")
  print(io,   "background_density=", parameters.background_density)
  print(io, ", gravitational_constant=", parameters.gravitational_constant)
  print(io, ", cfl=", parameters.cfl)
  print(io, ", n_iterations_max=", parameters.n_iterations_max)
  print(io, ", timestep_gravity=", parameters.timestep_gravity)
  print(io, ")")
end
function Base.show(io::IO, ::MIME"text/plain", parameters::ParametersEulerGravity)
  if get(io, :compact, false)
    show(io, parameters)
  else
    setup = [
             "background density (ρ₀)" => parameters.background_density,
             "gravitational constant (G)" => parameters.gravitational_constant,
             "CFL (gravity)" => parameters.cfl,
             "max. #iterations" => parameters.n_iterations_max,
             "time integrator" => parameters.timestep_gravity,
            ]
    summary_box(io, "ParametersEulerGravity", setup)
  end
end


"""
    SemidiscretizationEulerGravity

A struct containing everything needed to describe a spatial semidiscretization
of a the compressible Euler equations with self-gravity, reformulating the
Poisson equation for the gravitational potential as steady-state problem of
the hyperblic diffusion equations.
- Michael Schlottke-Lakemper, Andrew R. Winters, Hendrik Ranocha, Gregor J. Gassner (2020)
  "A purely hyperbolic discontinuous Galerkin approach for self-gravitating gas dynamics"
  [arXiv: 2008.10593](https://arXiv.org/abs/2008.10593)
"""
struct SemidiscretizationEulerGravity{SemiEuler, SemiGravity,
                                      Parameters<:ParametersEulerGravity, Cache} <: AbstractSemidiscretization
  semi_euler::SemiEuler
  semi_gravity::SemiGravity
  parameters::Parameters
  performance_counter::PerformanceCounter
  gravity_counter::PerformanceCounter
  cache::Cache

  function SemidiscretizationEulerGravity{SemiEuler, SemiGravity, Parameters, Cache}(
      semi_euler::SemiEuler, semi_gravity::SemiGravity,
      parameters::Parameters, cache::Cache) where {SemiEuler, SemiGravity,
                                                   Parameters<:ParametersEulerGravity, Cache}
    @assert ndims(semi_euler) == ndims(semi_gravity)
    @assert typeof(semi_euler.mesh) == typeof(semi_gravity.mesh)
    @assert polydeg(semi_euler.solver) == polydeg(semi_gravity.solver)

    performance_counter = PerformanceCounter()
    gravity_counter = PerformanceCounter()

    new(semi_euler, semi_gravity, parameters, performance_counter, gravity_counter, cache)
  end
end

"""
    SemidiscretizationEulerGravity(semi_euler::SemiEuler, semi_gravity::SemiGravity, parameters)

Construct a semidiscretization of the compressible Euler equations with self-gravity.
`parameters` should be given as [`ParametersEulerGravity`](@ref).
"""
function SemidiscretizationEulerGravity(semi_euler::SemiEuler, semi_gravity::SemiGravity, parameters) where
    {Mesh, SemiEuler<:SemidiscretizationHyperbolic{Mesh, <:AbstractCompressibleEulerEquations},
           SemiGravity<:SemidiscretizationHyperbolic{Mesh, <:AbstractHyperbolicDiffusionEquations}}

  u_ode = compute_coefficients(zero(real(semi_gravity)), semi_gravity)
  du_ode     = similar(u_ode)
  u_tmp1_ode = similar(u_ode)
  u_tmp2_ode = similar(u_ode)
  cache = (; u_ode, du_ode, u_tmp1_ode, u_tmp2_ode)

  SemidiscretizationEulerGravity{typeof(semi_euler), typeof(semi_gravity), typeof(parameters), typeof(cache)}(
    semi_euler, semi_gravity, parameters, cache)
end

function Base.show(io::IO, semi::SemidiscretizationEulerGravity)
  print(io, "SemidiscretizationEulerGravity using")
  print(io,       semi.semi_euler)
  print(io, ", ", semi.semi_gravity)
  print(io, ", ", semi.parameters)
  print(io, ", cache(")
  for (idx,key) in enumerate(keys(semi.cache))
    idx > 1 && print(io, " ")
    print(io, key)
  end
  print(io, "))")
end

function Base.show(io::IO, mime::MIME"text/plain", semi::SemidiscretizationEulerGravity)
  if get(io, :compact, false)
    show(io, semi)
  else
    summary_header(io, "SemidiscretizationEulerGravity")
    summary_line(io, "semidiscretization Euler", typeof(semi.semi_euler).name)
    show(increment_indent(io), mime, semi.semi_euler)
    summary_line(io, "semidiscretization gravity", typeof(semi.semi_gravity).name)
    show(increment_indent(io), mime, semi.semi_gravity)
    summary_line(io, "parameters", typeof(semi.parameters).name)
    show(increment_indent(io), mime, semi.parameters)
    summary_footer(io)
  end
end


# The compressible Euler semidiscretization is considered to be the main semidiscretization.
# The hyperbolic diffusion equations part is only used internally to update the gravitational
# potential during an rhs! evaluation of the flow solver.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationEulerGravity)
  mesh_equations_solver_cache(semi.semi_euler)
end

@inline Base.real(semi::SemidiscretizationEulerGravity) = real(semi.semi_euler)


# computes the coefficients of the initial condition
@inline function compute_coefficients(t, semi::SemidiscretizationEulerGravity)
  compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
  compute_coefficients(t, semi.semi_euler)
end

# computes the coefficients of the initial condition and stores the Euler part in `u_ode`
@inline function compute_coefficients!(u_ode, t, semi::SemidiscretizationEulerGravity)
  compute_coefficients!(semi.cache.u_ode, t, semi.semi_gravity)
  compute_coefficients!(u_ode, t, semi.semi_euler)
end


@inline function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationEulerGravity, cache_analysis)
  calc_error_norms(func, u, t, analyzer, semi.semi_euler, cache_analysis)
end


function rhs!(du_ode, u_ode, semi::SemidiscretizationEulerGravity, t)
  @unpack semi_euler, semi_gravity, cache = semi

  u_euler   = wrap_array(u_ode , semi_euler)
  du_euler  = wrap_array(du_ode, semi_euler)
  u_gravity = wrap_array(cache.u_ode, semi_gravity)

  time_start = time_ns()

  # standard semidiscretization of the compressible Euler equations
  @timeit_debug timer() "Euler solver" rhs!(du_ode, u_ode, semi_euler, t)

  # compute gravitational potential and forces
  @timeit_debug timer() "gravity solver" update_gravity!(semi, u_ode, semi.parameters.timestep_gravity)

  # add gravitational source source_terms to the Euler part
  if ndims(semi_euler) == 1
    @views @. du_euler[2, .., :] -= u_euler[1, .., :] * u_gravity[2, .., :]
    @views @. du_euler[3, .., :] -= u_euler[2, .., :] * u_gravity[2, .., :]
  elseif ndims(semi_euler) == 2
    @views @. du_euler[2, .., :] -=  u_euler[1, .., :] * u_gravity[2, .., :]
    @views @. du_euler[3, .., :] -=  u_euler[1, .., :] * u_gravity[3, .., :]
    @views @. du_euler[4, .., :] -= (u_euler[2, .., :] * u_gravity[2, .., :] +
                                     u_euler[3, .., :] * u_gravity[3, .., :])
  elseif ndims(semi_euler) == 3
    @views @. du_euler[2, .., :] -=  u_euler[1, .., :] * u_gravity[2, .., :]
    @views @. du_euler[3, .., :] -=  u_euler[1, .., :] * u_gravity[3, .., :]
    @views @. du_euler[4, .., :] -=  u_euler[1, .., :] * u_gravity[4, .., :]
    @views @. du_euler[5, .., :] -= (u_euler[2, .., :] * u_gravity[2, .., :] +
                                     u_euler[3, .., :] * u_gravity[3, .., :] +
                                     u_euler[4, .., :] * u_gravity[4, .., :])
  else
    error("Number of dimensions $(ndims(semi_euler)) not supported.")
  end

  runtime = time_ns() - time_start
  put!(semi.performance_counter, runtime)

  return nothing
end


function get_x_A_b(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector)
  @unpack semi_euler, semi_gravity, parameters, cache = semi

  # TODO: Clean-up
  # We can also use a more direct approach like
  #   A, b = linear_structure(semi_gravity)
  # for the following task. However, if we know that the hyperbolic diffusion system
  # we want to solve to steady state has a vanishing `b` at first. Hence, we can
  # use the more efficient variant below.
  x = cache.u_ode
  A, b = linear_structure(semi_gravity)
  # A = LinearMap(length(x), ismutating=true) do dest,src
  #   rhs!(dest, src, semi_gravity, 0)
  # end
  # b = zero(x)

  _b = wrap_array(b, semi_gravity)
  u_euler = wrap_array(u_ode, semi_euler)
  G    = parameters.gravitational_constant
  rho0 = parameters.background_density
  grav_scale = -4.0*pi*G
  @views @. _b[1, .., :] -= grav_scale * (u_euler[1, .., :] - rho0)

  return x, A, b
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, ::typeof(bicgstabl!))
  @unpack parameters, gravity_counter = semi
  @unpack resid_tol, resid_tol_type, n_iterations_max = parameters

  # TODO: Clean-up
  # let
  #   @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi

  #   # Can be changed by AMR
  #   resize!(cache.du_ode,     length(cache.u_ode))
  #   resize!(cache.u_tmp1_ode, length(cache.u_ode))
  #   resize!(cache.u_tmp2_ode, length(cache.u_ode))

  #   u_euler    = wrap_array(u_ode,        semi_euler)
  #   u_gravity  = wrap_array(cache.u_ode,  semi_gravity)
  #   du_gravity = wrap_array(cache.du_ode, semi_gravity)

  #   # set up main loop
  #   finalstep = false
  #   @unpack n_iterations_max, cfl, resid_tol, resid_tol_type = parameters
  #   iter = 0
  #   t = zero(real(semi_gravity.solver))
  #   dt = @timeit_debug timer() "calculate dt" cfl * max_dt(u_gravity, t, semi_gravity.mesh,
  #                                                           have_constant_speed(semi_gravity.equations),
  #                                                           semi_gravity.equations,
  #                                                           semi_gravity.solver, semi_gravity.cache)
  #   timestep_gravity_erk52_3Sstar!(cache, u_euler, t, dt, parameters, semi_gravity)
  # end

  @assert resid_tol_type === :l2_full
  x, A, b = get_x_A_b(semi, u_ode)

  abstol = resid_tol * length(x)
  reltol = 0.0
  time_start = time_ns()
  # TODO: We can also use bicgstabl!(x, A, b, l; kwargs...) instead of the default `l=2`.
  @timeit_debug timer() "linear solver" bicgstabl!(x, A, b; abstol, reltol, max_mv_products=n_iterations_max)
  runtime = time_ns() - time_start
  put!(gravity_counter, runtime)

  # TODO: Clean-up
  # @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi
  # @show norm(A * x - b)
  # @show mean_phi = integrate(first, cache.u_ode, semi_gravity)

  return nothing
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, ::typeof(gmres!))
  @unpack parameters, gravity_counter = semi
  @unpack resid_tol, resid_tol_type, n_iterations_max = parameters

  @assert resid_tol_type === :l2_full
  x, A, b = get_x_A_b(semi, u_ode)

  abstol = resid_tol * length(x)
  reltol = 0.0
  time_start = time_ns()
  @timeit_debug timer() "linear solver" gmres!(x, A, b; abstol, reltol, maxiter=n_iterations_max)
  runtime = time_ns() - time_start
  put!(gravity_counter, runtime)

  # TODO: Clean-up
  # @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi
  # @show norm(A * x - b)
  # @show mean_phi = integrate(first, cache.u_ode, semi_gravity)

  return nothing
end

function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, ::typeof(idrs!))
  @unpack parameters, gravity_counter = semi
  @unpack resid_tol, resid_tol_type, n_iterations_max = parameters

  @assert resid_tol_type === :l2_full
  x, A, b = get_x_A_b(semi, u_ode)

  abstol = resid_tol * length(x)
  reltol = 0.0
  time_start = time_ns()
  @timeit_debug timer() "linear solver" idrs!(x, A, b; abstol, reltol, maxiter=n_iterations_max)
  runtime = time_ns() - time_start
  put!(gravity_counter, runtime)

  # TODO: Clean-up
  # @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi
  # @show norm(A * x - b)
  # @show mean_phi = integrate(first, cache.u_ode, semi_gravity)

  return nothing
end



# TODO: Taal refactor, add some callbacks or so within the gravity update to allow investigating/optimizing it
function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector, timestep_gravity)
  @unpack semi_euler, semi_gravity, parameters, gravity_counter, cache = semi

  # Can be changed by AMR
  resize!(cache.du_ode,     length(cache.u_ode))
  resize!(cache.u_tmp1_ode, length(cache.u_ode))
  resize!(cache.u_tmp2_ode, length(cache.u_ode))

  u_euler    = wrap_array(u_ode,        semi_euler)
  u_gravity  = wrap_array(cache.u_ode,  semi_gravity)
  du_gravity = wrap_array(cache.du_ode, semi_gravity)

  # set up main loop
  finalstep = false
  @unpack n_iterations_max, cfl, resid_tol, resid_tol_type = parameters
  iter = 0
  t = zero(real(semi_gravity.solver))

  # calculate time step size sing a CFL condition once before integrating in time
  # since the mesh will not change and the linear equations have constant speeds
  dt = @timeit_debug timer() "calculate dt" cfl * max_dt(u_gravity, t, semi_gravity.mesh,
                                                          have_constant_speed(semi_gravity.equations),
                                                          semi_gravity.equations,
                                                          semi_gravity.solver, semi_gravity.cache)

  # Evaluate the RHS after computing a stage to check the termination criterion
  # correctly. Thus, we need to pre-start the RHS evaluations at the beginning
  # in some kind of FSAL-approach.
  rhs_gravity!(du_gravity, cache.du_ode, cache.u_ode, semi_gravity, t, u_euler, parameters)

  # TODO: Clean-up; this is one possible way of speeding-up the process by using
  # at first an RK method with few stages and another one with more stages later.
  # For Sedov, it can reduce the number of gravity RHS calls by ca. 20% compared
  # to using only `timestep_gravity_erk52_3Sstar!` if the first-order, three-stage
  # method is used to start here, depending on the chosen tolerances. For the
  # ≈ grid-converged resid_tol = 3.0e-11, the number of RHS evaluations is the
  # same as with using only the five-stage method. For tighter tolerances, using
  # only the five-stage method is better. For less tight tolerances, this approach
  # is better.
  # For Jeans, just using the five-stage method is better for the standard tolerance
  # and some tighter tolerances...
  # time_start = time_ns()
  # timestep_gravity_erk_test_3Sstar!(cache, u_euler, t, 0.5*dt, parameters, semi_gravity)
  # runtime = time_ns() - time_start
  # put!(gravity_counter, runtime)

  # iterate gravity solver until convergence or maximum number of iterations are reached
  while true
    # use an absolute residual tolerance check
    if resid_tol_type === :linf_phi
      residual = maximum(abs, @views du_gravity[1, .., :])
    elseif resid_tol_type === :l2_full
      residual = norm(du_gravity) / length(du_gravity)
    else # general fallback
      residual = convert(real(eltype(du_gravity)), NaN)
    end

    if residual <= resid_tol
      break
    end

    # check if we reached the maximum number of iterations
    if n_iterations_max > 0 && iter >= n_iterations_max
      @warn "Max iterations reached: Gravity solver failed to converge!" residual=residual t=t dt=dt
      break
    end

    # evolve solution by one pseudo-time step
    time_start = time_ns()
    timestep_gravity(cache, u_euler, t, dt, parameters, semi_gravity)
    runtime = time_ns() - time_start
    put!(gravity_counter, runtime)

    # update iteration counter
    iter += 1
    t += dt
  end

  # TODO: Clean-up
  # x, A, b = get_x_A_b(semi, u_ode)
  # @show norm(A * x - b)
  # @show mean_phi = integrate(first, cache.u_ode, semi_gravity)

  return nothing
end


@inline function rhs_gravity!(du_gravity, du_ode, u_ode, semi_gravity, t, u_euler, gravity_parameters)
  G    = gravity_parameters.gravitational_constant
  rho0 = gravity_parameters.background_density
  grav_scale = -4 * G * pi

  # rhs! has the source term for the harmonic problem
  # We don't need a `@timeit_debug timer() "rhs!"` here since that's already
  # included in the `rhs!` call.
  rhs!(du_ode, u_ode, semi_gravity, t)

  # Source term: Jeans instability OR coupling convergence test OR blast wave
  # put in gravity source term proportional to Euler density
  # OBS! subtract off the background density ρ_0 (spatial mean value)
  @views @. du_gravity[1, .., :] += grav_scale * (u_euler[1, .., :] - rho0)

  return nothing
end


# Integrate gravity solver for 2N-type low-storage schemes
function timestep_gravity_2N!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                              a, b, c)
  @unpack u_ode, du_ode, u_tmp1_ode = cache
  u_tmp1_ode .= zero(eltype(u_tmp1_ode))
  du_gravity = wrap_array(du_ode, semi_gravity)

  for stage in eachindex(c)
    a_stage = a[stage]
    b_stage_dt = b[stage] * dt
    @timeit_debug timer() "Runge-Kutta step" begin
      Threads.@threads for idx in eachindex(u_ode)
        u_tmp1_ode[idx] = du_ode[idx] - u_tmp1_ode[idx] * a_stage
        u_ode[idx] += u_tmp1_ode[idx] * b_stage_dt
      end
    end

    # Evaluate the RHS after computing a stage to check the termination criterion
    # correctly. Thus, we need to pre-start the RHS evaluations at the beginning
    # in some kind of FSAL-approach.
    # We do not need to set the time to the correct stage time since the gravity
    # system is autonomous.
    rhs_gravity!(du_gravity, du_ode, u_ode, semi_gravity, t, u_euler, gravity_parameters)
  end

  return nothing
end

function timestep_gravity_carpenter_kennedy_erk54_2N!(cache, u_euler, t, dt, gravity_parameters, semi_gravity)
  # Coefficients for Carpenter's 5-stage 4th-order low-storage Runge-Kutta method
  a = @SVector [0.0, 567301805773.0 / 1357537059087.0,2404267990393.0 / 2016746695238.0,
  3550918686646.0 / 2091501179385.0, 1275806237668.0 / 842570457699.0]
  b = @SVector [1432997174477.0 / 9575080441755.0, 5161836677717.0 / 13612068292357.0,
  1720146321549.0 / 2090206949498.0, 3134564353537.0 / 4481467310338.0,
  2277821191437.0 / 14882151754819.0]
  c = @SVector [0.0, 1432997174477.0 / 9575080441755.0, 2526269341429.0 / 6820363962896.0,
  2006345519317.0 / 3224310063776.0, 2802321613138.0 / 2924317926251.0]

  timestep_gravity_2N!(cache, u_euler, t, dt, gravity_parameters, semi_gravity, a, b, c)
end


# Integrate gravity solver for 3S*-type low-storage schemes
function timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                                  gamma1, gamma2, gamma3, beta, delta, c)
  @unpack u_ode, du_ode, u_tmp1_ode, u_tmp2_ode = cache
  u_tmp1_ode .= zero(eltype(u_tmp1_ode))
  u_tmp2_ode .= u_ode
  du_gravity = wrap_array(du_ode, semi_gravity)

  for stage in eachindex(c)
    delta_stage   = delta[stage]
    gamma1_stage  = gamma1[stage]
    gamma2_stage  = gamma2[stage]
    gamma3_stage  = gamma3[stage]
    beta_stage_dt = beta[stage] * dt
    @timeit_debug timer() "Runge-Kutta step" begin
      Threads.@threads for idx in eachindex(u_ode)
        u_tmp1_ode[idx] += delta_stage * u_ode[idx]
        u_ode[idx]       = (gamma1_stage * u_ode[idx] +
                            gamma2_stage * u_tmp1_ode[idx] +
                            gamma3_stage * u_tmp2_ode[idx] +
                            beta_stage_dt * du_ode[idx])
      end
    end

    # Evaluate the RHS after computing a stage to check the termination criterion
    # correctly. Thus, we need to pre-start the RHS evaluations at the beginning
    # in some kind of FSAL-approach.
    # We do not need to set the time to the correct stage time since the gravity
    # system is autonomous.
    rhs_gravity!(du_gravity, du_ode, u_ode, semi_gravity, t, u_euler, gravity_parameters)
  end

  return nothing
end

function timestep_gravity_erk_test_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity)
  # New 3Sstar coefficients optimized for
  # equations = HyperbolicDiffusionEquations2D()
  # initial_condition = initial_condition_poisson_periodic
  # surface_flux = flux_upwind
  # solver = DGSEM(3, surface_flux)
  # coordinates_min = (0, 0)
  # coordinates_max = (1, 1)
  # mesh = TreeMesh(coordinates_min, coordinates_max,
  #                 initial_refinement_level=3,
  #                 n_cells_max=30_000)
  # semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
  #                                     source_terms=source_terms_poisson_periodic)
  # 2 stages, order 1, erk-1-2_2021-01-21T16-59-55.txt
  # gamma1 = @SVector [0.0000000000000000E+00, -1.2180814892100231E+00]
  # gamma2 = @SVector [1.0000000000000000E+00, 2.2180814892100231E+00]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00]
  # beta   = @SVector [2.5570794725413060E-01, 1.3114731171941489E+00]
  # delta  = @SVector [1.0000000000000000E+00, 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, 2.5570794725413060E-01]
  # 3 stages, order 1
  # This scheme can be between ca. 5% and 20% better than `timestep_gravity_erk52_3Sstar!`
  # for EOC and Jeans, depending on the tolerances... However, it does not seem to be better
  # for Sedov...
  gamma1 = @SVector [0.0000000000000000E+00, 5.3542666596047617E-01, 9.1410889739925583E-01]
  gamma2 = @SVector [1.0000000000000000E+00, 4.3591819397582626E-01, 8.0593291910989059E-02]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00]
  beta   = @SVector [5.1434308417699570E-01, 5.0548475589200048E-01, 2.6999513995934876E-01]
  delta  = @SVector [1.0000000000000000E+00, 6.5735132095190080E-02, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 5.1434308417699570E-01, 7.9561633173060387E-01]
  # # 4 stages, order 1, 3Sstar-1-4_2021-01-21T18-19-22.txt
  # gamma1 = @SVector [0.0000000000000000E+00, -5.6625536579338731E-01, 1.3418587031785676E+00, 8.8890148834863436E-01]
  # gamma2 = @SVector [1.0000000000000000E+00, 1.1536881793250324E+00, -1.8570628751004167E-01, 3.8337868078920548E-02]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 4.0523982162793015E-02]
  # beta   = @SVector [2.3171237092062003E-01, 3.2627637023876249E-01, 1.9590115587227472E-01, 5.0746738742196784E-01]
  # delta  = @SVector [1.0000000000000000E+00, 3.5760718872037695E-01, 4.8324972443116271E-01, 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, 2.3171237092062003E-01, 2.9066491782488579E-01, 5.4445940944117321E-01]
  # # 4 stages, order 2, 3Sstar-2-4_2021-01-21T18-22-47.txt
  # gamma1 = @SVector [0.0000000000000000E+00, 3.9242586618503339E-01, 1.0035329389428556E+00, 5.5592551752986541E-01]
  # gamma2 = @SVector [1.0000000000000000E+00, 3.1722869401108084E-01, -1.6915015356303413E-03, 2.1306582216797171E-01]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, -9.4346440879594987E-04]
  # beta   = @SVector [2.9330285434493913E-01, 3.3265933704162054E-01, 2.1171710030945143E-01, 5.0844675353885938E-01]
  # delta  = @SVector [1.0000000000000000E+00, 9.1525591879070056E-01, 1.7338477614017425E-01, 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, 2.9330285434493913E-01, 5.3291810995299238E-01, 7.4590760404926193E-01]
  # # 5 stages, order 1, 3Sstar-1-5_2021-01-22T06-47-41.txt
  # gamma1 = @SVector [0.0000000000000000E+00, -1.8802729519221351E-01, 4.6604651511432649E-01, 8.8394183350193534E-01, -6.5068141753484565E-01]
  # gamma2 = @SVector [1.0000000000000000E+00, 5.4134480367690074E-01, 2.0790247136947018E-01, -7.5830656883294466E-04, 6.2410224487791299E-01]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 1.1833886026287863E-01, -2.2637748845630642E-01]
  # beta   = @SVector [1.4639181558742428E-01, 3.1071563744231162E-01, 8.7907721397119457E-02, 1.7847768337323872E-01, 1.0178754551387090E+00]
  # delta  = @SVector [1.0000000000000000E+00, 1.1945852017474661E+00, 3.7370309886040687E-01, 4.3932611096717716E-01, 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, 1.4639181558742428E-01, 3.7785900436065667E-01, 3.2972235839127506E-01, 4.6958353540042619E-01]
  # # 7 stages, order 2
  # gamma1 = @SVector [0.0000000000000000E+00, 7.4645855541196471E-01, 1.1933706288014441E+00, 4.4882143764278509E-01, 1.2597816001927011E+00, -1.9395359618180008E-01, -2.0529989434528106E-01]
  # gamma2 = @SVector [1.0000000000000000E+00, 1.9604567859122732E-01, -9.4228044406896444E-02, 1.8274409185201260E-01, -5.0215951766262042E-02, 1.6864474841417218E-01, 7.7591329451069480E-01]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 1.3273971095845494E-01, -1.2014683669108431E-01, 6.8162040983132111E-01, -1.1518807130939930E+00]
  # beta   = @SVector [2.9710300213012131E-01, 2.8004654286613695E-01, 1.8558682604908017E-01, 4.7849858779738258E-01, 4.6699827604810107E-01, -1.0894983405033070E-02, -2.0239599603544003E-01]
  # delta  = @SVector [1.0000000000000000E+00, 2.9327739540075143E-01, 7.5887841467240658E-01, 2.3759729306746027E-01, 4.9093228845702752E-01, 2.5725792515626500E-01, 0.0000000000000000E+00]
  # c      = zero(gamma1)
  # # xxx stages, order xxx
  # gamma1 = @SVector [0.0000000000000000E+00, ]
  # gamma2 = @SVector [1.0000000000000000E+00, ]
  # gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, ]
  # beta   = @SVector []
  # delta  = @SVector [1.0000000000000000E+00, , 0.0000000000000000E+00]
  # c      = @SVector [0.0000000000000000E+00, ]

  timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                           gamma1, gamma2, gamma3, beta, delta, c)
end

function timestep_gravity_erk51_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity)
  # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
  # and examples/parameters_hypdiff_lax_friedrichs.toml
  # 5 stages, order 1
  gamma1 = @SVector [0.0000000000000000E+00, 5.2910412316555866E-01, 2.8433964362349406E-01, -1.4467571130907027E+00, 7.5592215948661057E-02]
  gamma2 = @SVector [1.0000000000000000E+00, 2.6366970460864109E-01, 3.7423646095836322E-01, 7.8786901832431289E-01, 3.7754129043053775E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 8.0043329115077388E-01, 1.3550099149374278E-01]
  beta   = @SVector [1.9189497208340553E-01, 5.4506406707700059E-02, 1.2103893164085415E-01, 6.8582252490550921E-01, 8.7914657211972225E-01]
  delta  = @SVector [1.0000000000000000E+00, 7.8593091509463076E-01, 1.2639038717454840E-01, 1.7726945920209813E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 1.9189497208340553E-01, 1.9580448818599061E-01, 2.4241635859769023E-01, 5.0728347557552977E-01]

  timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                           gamma1, gamma2, gamma3, beta, delta, c)
end

function timestep_gravity_erk52_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity)
  # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
  # and examples/parameters_hypdiff_lax_friedrichs.toml
  # 5 stages, order 2
  gamma1 = @SVector [0.0000000000000000E+00, 5.2656474556752575E-01, 1.0385212774098265E+00, 3.6859755007388034E-01, -6.3350615190506088E-01]
  gamma2 = @SVector [1.0000000000000000E+00, 4.1892580153419307E-01, -2.7595818152587825E-02, 9.1271323651988631E-02, 6.8495995159465062E-01]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 4.1301005663300466E-01, -5.4537881202277507E-03]
  beta   = @SVector [4.5158640252832094E-01, 7.5974836561844006E-01, 3.7561630338850771E-01, 2.9356700007428856E-02, 2.5205285143494666E-01]
  delta  = @SVector [1.0000000000000000E+00, 1.3011720142005145E-01, 2.6579275844515687E-01, 9.9687218193685878E-01, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 4.5158640252832094E-01, 1.0221535725056414E+00, 1.4280257701954349E+00, 7.1581334196229851E-01]

  timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                           gamma1, gamma2, gamma3, beta, delta, c)
end

function timestep_gravity_erk53_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity)
  # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
  # and examples/parameters_hypdiff_lax_friedrichs.toml
  # 5 stages, order 3
  gamma1 = @SVector [0.0000000000000000E+00, 6.9362208054011210E-01, 9.1364483229179472E-01, 1.3129305757628569E+00, -1.4615811339132949E+00]
  gamma2 = @SVector [1.0000000000000000E+00, 1.3224582239681788E+00, 2.4213162353103135E-01, -3.8532017293685838E-01, 1.5603355704723714E+00]
  gamma3 = @SVector [0.0000000000000000E+00, 0.0000000000000000E+00, 0.0000000000000000E+00, 3.8306787039991996E-01, -3.5683121201711010E-01]
  beta   = @SVector [8.4476964977404881E-02, 3.0834660698015803E-01, 3.2131664733089232E-01, 2.8783574345390539E-01, 8.2199204703236073E-01]
  delta  = @SVector [1.0000000000000000E+00, -7.6832695815481578E-01, 1.2497251501714818E-01, 1.4496404749796306E+00, 0.0000000000000000E+00]
  c      = @SVector [0.0000000000000000E+00, 8.4476964977404881E-02, 2.8110631488732202E-01, 5.7093842145029405E-01, 7.2999896418559662E-01]

  timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                           gamma1, gamma2, gamma3, beta, delta, c)
end


# TODO: Taal decide, where should specific parts like these be?
@inline function save_solution_file(u_ode::AbstractVector, t, dt, iter,
                                    semi::SemidiscretizationEulerGravity, solution_callback,
                                    element_variables=Dict{Symbol,Any}())

  u_euler = wrap_array(u_ode, semi.semi_euler)
  filename_euler = save_solution_file(u_euler, t, dt, iter,
                                      mesh_equations_solver_cache(semi.semi_euler)...,
                                      solution_callback, element_variables, system="euler")

  u_gravity = wrap_array(semi.cache.u_ode, semi.semi_gravity)
  filename_gravity = save_solution_file(u_gravity, t, dt, iter,
                                        mesh_equations_solver_cache(semi.semi_gravity)...,
                                        solution_callback, element_variables, system="gravity")

  return filename_euler, filename_gravity
end


@inline function (amr_callback::AMRCallback)(u_ode::AbstractVector,
                                             semi::SemidiscretizationEulerGravity,
                                             t, iter; kwargs...)
  passive_args = ((semi.cache.u_ode, mesh_equations_solver_cache(semi.semi_gravity)...),)
  amr_callback(u_ode, mesh_equations_solver_cache(semi.semi_euler)..., t, iter;
               kwargs..., passive_args=passive_args)
end
