# TODO: Allow kwargs that are later used when constructing `integrator_euler`?
mutable struct EulerAcousticsCouplingCallback{RealT<:Real, MeanValues}
  stepsize_callback_acoustics::StepsizeCallback{RealT}
  stepsize_callback_euler::StepsizeCallback{RealT}
  mean_values::MeanValues
  integrator_euler::Union{Nothing, AbstractODEIntegrator}
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:EulerAcousticsCouplingCallback})
  @nospecialize cb # reduce precompilation time
  euler_acoustics_coupling = cb.affect!

  print(io, "EulerAcousticsCouplingCallback(")
  print(io,       euler_acoustics_coupling.stepsize_callback_acoustics)
  print(io, ", ", euler_acoustics_coupling.stepsize_callback_euler)
  print(io, ", ", euler_acoustics_coupling.mean_values, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:EulerAcousticsCouplingCallback})
  @nospecialize cb # reduce precompilation time
  euler_acoustics_coupling = cb.affect!

  summary_header(io, "EulerAcousticsCouplingCallback")
  summary_line(io, "acoustics StepsizeCallback", euler_acoustics_coupling.stepsize_callback_acoustics)
  summary_line(io, "Euler StepsizeCallback", euler_acoustics_coupling.stepsize_callback_euler)
  summary_line(io, "mean values", euler_acoustics_coupling.mean_values)
  summary_footer(io)
end


function EulerAcousticsCouplingCallback(cfl_acoustics::Real, cfl_euler::Real,
                                        averaging_callback::DiscreteCallback{<:Any, <:AveragingCallback})
  @unpack mean_values = averaging_callback.affect!
  return EulerAcousticsCouplingCallback(cfl_acoustics, cfl_euler, mean_values)
end

function EulerAcousticsCouplingCallback(cfl_acoustics::Real, cfl_euler::Real, averaging_file,
                                        semi_euler::SemidiscretizationHyperbolic)
  mean_values = load_averaging_file(averaging_file, mesh_equations_solver_cache(semi_euler)...)

  return EulerAcousticsCouplingCallback(cfl_acoustics, cfl_euler, mean_values)
end

function EulerAcousticsCouplingCallback(cfl_acoustics::Real, cfl_euler::Real, mean_values)

  euler_acoustics_coupling = EulerAcousticsCouplingCallback{typeof(cfl_acoustics), typeof(mean_values)}(
    StepsizeCallback(cfl_acoustics), StepsizeCallback(cfl_euler), mean_values, nothing)
  condition = (u, t, integrator) -> true

  return DiscreteCallback(condition, euler_acoustics_coupling, save_positions=(false, false),
                          initialize=initialize!)
end


# This is called before the main loop and initializes the flow solver and calculates the gradient
# of the squared mean speed of sound which is needed for the conservation source term
function initialize!(cb::DiscreteCallback{Condition,Affect!}, u_ode, t, integrator_acoustics) where {Condition, Affect!<:EulerAcousticsCouplingCallback}
  euler_acoustics_coupling = cb.affect!
  semi = integrator_acoustics.p
  @unpack semi_acoustics, semi_euler = semi

  # Set up ODE Integrator for Euler equations
  tspan = integrator_acoustics.sol.prob.tspan
  alg = integrator_acoustics.alg

  ode_euler = semidiscretize(semi_euler, tspan)
  euler_acoustics_coupling.integrator_euler = init(ode_euler, alg, save_everystep=false, dt=1.0) # dt will be overwritten

  # Set mean values in u_ode according to `AveragingCallback`
  u_acoustics = wrap_array(u_ode, semi_acoustics)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack mean_values = euler_acoustics_coupling
  @views @. u_acoustics[4:5, .., :] = mean_values.v_mean
  @views @. u_acoustics[6, .., :] = mean_values.c_mean
  @views @. u_acoustics[7, .., :] = mean_values.rho_mean

  # Calculate gradient of squared mean speed of sound for the q_cons source term
  @trixi_timeit timer() "calc conservation source term" calc_gradient_c_mean_square!(
    semi.cache.grad_c_mean_sq, u_acoustics, mesh, equations, solver, cache)

  # Adjust stepsize, advance the flow solver by one time step
  cb.affect!(integrator_acoustics)

  return nothing
end


# This function is called at the end of every time step and advances the Euler solution by one
# time step, # manages the time stepsize for both the acoustics and Euler solvers and calculates the
# acoustic sources for the next acoustics time step
function (euler_acoustics_coupling::EulerAcousticsCouplingCallback)(integrator_acoustics)
  @unpack stepsize_callback_acoustics, stepsize_callback_euler, integrator_euler = euler_acoustics_coupling

  @assert integrator_acoustics.t == integrator_euler.t

  # Use the the minimum of the acoustics and Euler stepsizes for both solvers
  stepsize_callback_acoustics(integrator_acoustics)
  stepsize_callback_euler(integrator_euler)
  dt = min(get_proposed_dt(integrator_acoustics), get_proposed_dt(integrator_euler))

  set_proposed_dt!(integrator_acoustics, dt)
  integrator_acoustics.opts.dtmax = dt
  integrator_acoustics.dtcache = dt

  set_proposed_dt!(integrator_euler, dt)
  integrator_euler.opts.dtmax = dt
  integrator_euler.dtcache = dt

  # Advance the Euler solution by one step and check for errors
  if !isfinished(integrator_euler)
    @trixi_timeit timer() "Euler solver" step!(integrator_euler)
    return_code = check_error(integrator_euler)
    if return_code != :Success && return_code != :Default
      error("Error during Euler time integration. Received return code $(return_code)")
    end
  end

  # Calculate acoustic sources based on linearized lamb vector
  semi = integrator_acoustics.p
  semi_euler = integrator_euler.p
  u_acoustics = wrap_array(integrator_acoustics.u, semi)
  u_euler = wrap_array(integrator_euler.u, semi_euler)
  @unpack acoustic_source_terms = semi.cache
  @unpack source_region, weights = semi
  @unpack vorticity_mean = euler_acoustics_coupling.mean_values

  @trixi_timeit timer() "calc acoustic source terms" calc_acoustic_sources!(
    acoustic_source_terms, u_euler, u_acoustics, vorticity_mean, source_region, weights,
    mesh_equations_solver_cache(semi_euler)...)

  # avoid re-evaluation possible FSAL stages
  u_modified!(integrator_acoustics, false)
  u_modified!(integrator_euler, false)

  return nothing
end

include("euler_acoustics_coupling_dg2d.jl")