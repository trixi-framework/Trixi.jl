# TODO: Allow kwargs that are later used when constructing `integrator_euler`?
mutable struct ApeEulerCouplingCallback{RealT<:Real}
  stepsize_callback_ape::StepsizeCallback{RealT}
  stepsize_callback_euler::StepsizeCallback{RealT}
  averaging_callback::DiscreteCallback{<:Any, <:AveragingCallback}
  integrator_euler::Union{Nothing, AbstractODEIntegrator}
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:ApeEulerCouplingCallback})

end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:ApeEulerCouplingCallback})

end


function ApeEulerCouplingCallback(cfl_ape::Real, cfl_euler::Real,
                                  averaging_callback::DiscreteCallback{<:Any, <:AveragingCallback})

  ape_euler_coupling = ApeEulerCouplingCallback{typeof(cfl_ape)}(StepsizeCallback(cfl_ape),
                                                                 StepsizeCallback(cfl_euler),
                                                                 averaging_callback, nothing)
  condition = (u, t, integrator) -> true

  return DiscreteCallback(condition, ape_euler_coupling, save_positions=(false, false),
                          initialize=initialize!)
end

function initialize!(cb::DiscreteCallback{Condition,Affect!}, u_ode, t, integrator) where {Condition, Affect!<:ApeEulerCouplingCallback}
  ape_euler_coupling = cb.affect!
  semi = integrator.p
  @unpack semi_ape, semi_euler = semi

  # Set up ODE Integrator for Euler equations
  tspan = integrator.sol.prob.tspan
  alg = integrator.alg

  ode_euler = semidiscretize(semi_euler, tspan)
  ape_euler_coupling.integrator_euler = init(ode_euler, alg, save_everystep=false, dt=1.0) # dt will be overwritten

  # Set mean values in u_ode according to `AveragingCallback`
  u_ape = wrap_array(u_ode, semi_ape)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  @unpack mean_values = ape_euler_coupling.averaging_callback.affect!
  @views @. u_ape[4:5, .., :] = mean_values.v_mean
  @views @. u_ape[6, .., :] = mean_values.c_mean
  @views @. u_ape[7, .., :] = mean_values.rho_mean

  # Calculate gradient of squared mean speed of sound for the q_cons source term
  @trixi_timeit timer() "calc conservation source term" calc_gradient_c_mean_square!(
    semi.cache.grad_c_mean_sq, u_ape, mesh, equations, solver, cache)

  # Adjust stepsize, advance the flow solver by one time step
  cb.affect!(integrator)

  return nothing
end


# This is called at the end of every time step and advances the Euler solution by one time step,
# manages the time stepsize for both the APE and Euler solvers and calculates the acoustic sources
# for the next APE time step
function (ape_euler_coupling::ApeEulerCouplingCallback)(integrator_ape)
  @unpack stepsize_callback_ape, stepsize_callback_euler, integrator_euler = ape_euler_coupling

  @assert integrator_ape.t == integrator_euler.t

  # Use the the minimum of the APE and Euler stepsizes for both solvers
  stepsize_callback_ape(integrator_ape)
  stepsize_callback_euler(integrator_euler)
  dt = min(get_proposed_dt(integrator_ape), get_proposed_dt(integrator_euler))

  set_proposed_dt!(integrator_ape, dt)
  integrator_ape.opts.dtmax = dt
  integrator_ape.dtcache = dt

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
  semi = integrator_ape.p
  semi_euler = integrator_euler.p
  u_ape = wrap_array(integrator_ape.u, semi)
  u_euler = wrap_array(integrator_euler.u, semi_euler)
  @unpack acoustic_source_terms = semi.cache
  @unpack source_region, weights = semi
  @unpack vorticity_mean = ape_euler_coupling.averaging_callback.affect!.mean_values

  @trixi_timeit timer() "calc acoustic source terms" calc_acoustic_sources!(
    acoustic_source_terms, u_euler, u_ape, vorticity_mean, source_region, weights,
    mesh_equations_solver_cache(semi_euler)...)

  # avoid re-evaluation possible FSAL stages
  u_modified!(integrator_ape, false)
  u_modified!(integrator_euler, false)

  return nothing
end

include("ape_euler_coupling_dg2d.jl")