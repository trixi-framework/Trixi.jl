
"""
    ParametersEulerGravity(; background_density=0.0,
                             gravitational_constant=1.0,
                             cfl=1.0,
                             n_iterations_max=10^4,
                             timestep_gravity=timestep_gravity_erk52_3Sstar!)

Set up parameters for the gravitational part of a [`SemidiscretizationEulerGravity`](@ref).
"""
struct ParametersEulerGravity{RealT<:Real, TimestepGravity}
  background_density    ::RealT # aka rho0
  gravitational_constant::RealT # aka G
  cfl                   ::RealT
  n_iterations_max      ::Int
  timestep_gravity::TimestepGravity
end

function ParametersEulerGravity(; background_density=0.0,
                                  gravitational_constant=1.0,
                                  cfl=1.0,
                                  n_iterations_max=10^4,
                                  timestep_gravity=timestep_gravity_erk52_3Sstar!)
  background_density, gravitational_constant, cfl = promote(background_density, gravitational_constant, cfl)
  ParametersEulerGravity(background_density, gravitational_constant, cfl, n_iterations_max, timestep_gravity)
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
  println(io, "ParametersEulerGravity using")
  println(io, "- background_density:     ", parameters.background_density)
  println(io, "- gravitational_constant: ", parameters.gravitational_constant)
  println(io, "- cfl (gravity):    ", parameters.cfl)
  println(io, "- n_iterations_max: ", parameters.n_iterations_max)
  print(io,   "- timestep_gravity: ", parameters.timestep_gravity)
end


"""
    SemidiscretizationEulerGravity

A struct containing everything needed to describe a spatial semidiscretization
of a the compressible Euler equations with self-gravity, reformulating the
Poisson equation for the gravitational potential as steady-state problem of
the hyperblic diffusion equations.
- Schlottke-Lakemper, Winters, Ranocha, Gassner (2020)
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
  println(io, "SemidiscretizationEulerGravity using")
  print(io, "  "); show(io, mime, semi.semi_euler); println()
  print(io, "  "); show(io, mime, semi.semi_gravity); println()
  print(io, "  "); show(io, mime, semi.parameters); println()
  print(io, "  cache with fields:")
  for key in keys(semi.cache)
    print(io, " ", key)
  end
end


# The compressible Euler semidiscretization is considered to be the main semidiscretization.
# The hyperbolic diffusion equations part is only used internally to update the gravitational
# potential during an rhs! evaluation of the flow solver.
@inline function mesh_equations_solver_cache(semi::SemidiscretizationEulerGravity)
  mesh_equations_solver_cache(semi.semi_euler)
end


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


@inline function calc_error_norms(func, u, t, analyzer, semi::SemidiscretizationEulerGravity)
  calc_error_norms(func, u, t, analyzer, semi.semi_euler)
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
  @timeit_debug timer() "gravity solver" update_gravity!(semi, u_ode)

  # add gravitational source source_terms to the Euler part
  if ndims(semi_euler) == 2
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


# TODO: Taal refactor, add some callbacks or so within the gravity update to allow investigating/optimizing it
function update_gravity!(semi::SemidiscretizationEulerGravity, u_ode::AbstractVector)
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
  @unpack n_iterations_max, cfl, timestep_gravity = parameters
  @unpack resid_tol = semi_gravity.equations
  iter = 0
  t = zero(real(semi_gravity.solver))

  # iterate gravity solver until convergence or maximum number of iterations are reached
  @unpack equations = semi_gravity
  while !finalstep
    dt = @timeit_debug timer() "calculate dt" cfl * max_dt(u_gravity, t, semi_gravity.mesh,
                                                           have_constant_speed(equations), equations,
                                                           semi_gravity.solver, semi_gravity.cache)

    # evolve solution by one pseudo-time step
    time_start = time_ns()
    timestep_gravity(cache, u_euler, t, dt, parameters, semi_gravity)
    runtime = time_ns() - time_start
    put!(gravity_counter, runtime)

    # update iteration counter
    iter += 1
    t += dt

    # check if we reached the maximum number of iterations
    if n_iterations_max > 0 && iter >= n_iterations_max
      @warn "Max iterations reached: Gravity solver failed to converge!" residual=maximum(abs, @views du_gravity[1, .., :]) t=t dt=dt
      finalstep = true
    end

    # this is an absolute tolerance check
    if maximum(abs, @views du_gravity[1, .., :]) <= resid_tol
      finalstep = true
    end
  end

  return nothing
end


function timestep_gravity_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity,
                                  gamma1, gamma2, gamma3, beta, delta, c)
  G    = gravity_parameters.gravitational_constant
  rho0 = gravity_parameters.background_density
  grav_scale = -4 * G * pi

  @unpack u_ode, du_ode, u_tmp1_ode, u_tmp2_ode = cache
  u_tmp1_ode .= zero(eltype(u_tmp1_ode))
  u_tmp2_ode .= u_ode
  du_gravity = wrap_array(du_ode, semi_gravity)
  for stage in eachindex(c)
    t_stage = t + dt * c[stage]
    @timeit_debug timer() "rhs!" rhs!(du_ode, u_ode, semi_gravity, t_stage)

    # Source term: Jeans instability OR coupling convergence test OR blast wave
    # put in gravity source term proportional to Euler density
    # OBS! subtract off the background density ρ_0 around which the Jeans instability is perturbed
    @views @. du_gravity[1, .., :] += grav_scale * (u_euler[1, .., :] - rho0)

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
  end

  return nothing
end


function timestep_gravity_erk52_3Sstar!(cache, u_euler, t, dt, gravity_parameters, semi_gravity)
  # New 3Sstar coefficients optimized for polynomials of degree polydeg=3
  # and examples/parameters_hyp_diff_llf.toml
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
                                             semi::SemidiscretizationEulerGravity; kwargs...)
  passive_args = ((semi.cache.u_ode, mesh_equations_solver_cache(semi.semi_gravity)...),)
  amr_callback(u_ode, mesh_equations_solver_cache(semi.semi_euler)...;
               kwargs..., passive_args=passive_args)
end
