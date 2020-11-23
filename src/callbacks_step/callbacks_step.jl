
# overload this function for specific callbacks which use element element variables
# that should be saved
get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                       callback; kwargs...) = nothing

@inline function get_element_variables!(element_variables, u_ode::AbstractVector,
                                        semi::AbstractSemidiscretization, cb::DiscreteCallback;
                                        kwargs...)
  mesh, equations, solver, cache = mesh_equations_solver_cache(semi)
  u = wrap_array(u_ode, mesh, equations, solver, cache)
  get_element_variables!(element_variables, u, mesh, equations, solver, cache,
                         cb.affect!; kwargs...)
end


@inline function isfinished(integrator)
  # Checking for floating point equality is OK here as `DifferentialEquations.jl`
  # sets the time exactly to the final time in the last iteration
  return integrator.t == last(integrator.sol.prob.tspan) ||
         isempty(integrator.opts.tstops) ||
         integrator.iter == integrator.opts.maxiters
end


# `include` callback definitions in the order that we currently prefer
# when combining them into a `CallbackSet` which is called *after* a complete step
# The motivation is as follows:
# * `SummaryCallback` controls, among other things, timers and should thus be first
# * `SteadyStateCallback` may mark a time step as the last step, which is needed by other callbacks
# * `AnalysisCallback` may also do some checks that mark a step as the last one
# * `AliveCallback` belongs to `AnalysisCallback` and should thus be nearby
# * `SaveRestartCallback`/`SaveSolutionCallback` should save the current solution state before it is
#   potentially degraded by AMR
# * `AMRCallback` really belongs to the next time step already, as it should be the "first" callback
#   in a time step loop (however, callbacks are always executed *after* a step, thus it comes near
#   the end here)
# * `StepsizeCallback` must come after AMR to accomodate potential changes in the minimum cell size
# * `GlmSpeedCallback` must come after computing time step size because it affects the value of c_h
include("summary.jl")
include("steady_state.jl")
include("analysis.jl")
include("alive.jl")
include("save_restart.jl")
include("save_solution.jl")
include("amr.jl")
include("stepsize.jl")
include("glm_speed.jl")


# The `TrivialCallback` purposely does nothing: It allows to quickly disable specific callbacks
# when using `trixi_include` or `test_trixi_include`
include("trivial.jl")
