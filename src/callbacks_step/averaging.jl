struct AveragingCallback{TSpan, MeanValues, Cache}
  tspan::TSpan
  mean_values::MeanValues
  cache::Cache
end


function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AveragingCallback})
  print(io, "AveragingCallback")
end

function Base.show(io::IO, ::MIME"text/plain", cb::DiscreteCallback{<:Any, <:AveragingCallback})
  if get(io, :compact, false)
    show(io, cb)
  else
    averaging_callback = cb.affect!

    setup = [
             "Start time" => first(averaging_callback.tspan),
             "Final time" => last(averaging_callback.tspan)
            ]
    summary_box(io, "AveragingCallback", setup)
  end
end

function AveragingCallback(semi::SemidiscretizationHyperbolic; tspan)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  mean_values = initialize_mean_values(mesh, equations, solver, cache)
  cache = create_cache(AveragingCallback, mesh, equations, solver, cache)

  averaging_callback = AveragingCallback(tspan, mean_values, cache)
  condition = (u, t, integrator) -> first(tspan) <= t <= last(tspan) ? true : false

  return DiscreteCallback(condition, averaging_callback, save_positions=(false,false),
                          initialize=initialize!)
end


function initialize!(cb::DiscreteCallback{Condition,Affect!}, u_ode, t, integrator) where {Condition, Affect!<:AveragingCallback}
  averaging_callback = cb.affect!
  semi = integrator.p
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)

  @trixi_timeit timer() "averaging" initialize_cache!(averaging_callback.cache, u, t,
                                                      mesh, equations, solver, cache)

  # avoid re-evaluating possible FSAL stages
  u_modified!(integrator, false)
  return nothing
end

# This function is called during time integration and updates the mean values according to the
# trapezoidal rule
function (averaging_callback::AveragingCallback)(integrator)
  @unpack v_mean, c_mean, rho_mean, vorticity_mean = averaging_callback.mean_values
  @unpack mean_values = averaging_callback

  u_ode = integrator.u
  u_prev_ode = integrator.uprev
  semi = integrator.p
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)
  u_prev = wrap_array(u_prev_ode, mesh, equations, solver, cache)

  dt = get_proposed_dt(integrator)
  tspan = averaging_callback.tspan

  integration_constant = 0.5 * dt / (tspan[2] - tspan[1]) # .5 due to trapezoidal rule

  @trixi_timeit timer() "averaging" calc_means!(mean_values, averaging_callback.cache, u, u_prev,
                                                integration_constant, mesh, equations, solver, cache)

  return nothing
end

include("averaging_dg2d.jl")