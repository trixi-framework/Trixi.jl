
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
# when combining them into a `CallbackSet` which is called after a complete step
include("summary.jl")
include("steady_state.jl")
include("amr.jl")
include("stepsize.jl")
include("save_restart.jl")
include("save_solution.jl")
include("analysis.jl")
include("alive.jl")
